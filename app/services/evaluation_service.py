"""RAG evaluation service.

Runs the test dataset against the RAG pipeline and computes automatic metrics
(BERTScore, BLEU-4, ROUGE-L, RAGAS), persisting every run and per-question result
to the database for a reproducible, auditable evaluation record.

Heavy metric dependencies (torch/bert-score/ragas) are imported lazily inside the
functions that need them, so the app can start even if they are not installed.
"""

import os
import re
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.database.session import SessionLocal
from app.models.evaluation import EvaluationRun, EvaluationItem

logger = logging.getLogger(__name__)

# Datasets live next to the existing evaluation scripts
DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def list_datasets() -> List[Dict[str, Any]]:
    """List available evaluation dataset JSON files in the scripts directory."""
    datasets = []
    if not DATASET_DIR.exists():
        return datasets
    for path in sorted(DATASET_DIR.glob("*.json")):
        try:
            questions = _load_dataset(path.name)
            datasets.append({
                "name": path.name,
                "num_questions": len(questions),
                "has_ground_truth": any(q.get("ground_truth") for q in questions),
            })
        except Exception as e:
            logger.warning(f"Skipping dataset {path.name}: {e}")
    return datasets


def _load_dataset(dataset_name: str) -> List[Dict[str, Any]]:
    """Load a dataset by file name from the scripts directory."""
    path = DATASET_DIR / dataset_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list):
        raise ValueError("Invalid dataset format (expected list or {'questions': [...]})")
    return data


def _validate_questions(data: Any) -> List[Dict[str, Any]]:
    """Validate parsed dataset content and return the list of questions."""
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if not isinstance(data, list) or not data:
        raise ValueError("Dataset harus berupa list pertanyaan (atau {'questions': [...]}) dan tidak boleh kosong")
    for i, q in enumerate(data, 1):
        if not isinstance(q, dict) or not (q.get("question") or q.get("query")):
            raise ValueError(f"Item ke-{i} tidak memiliki field 'question'")
    return data


def save_uploaded_dataset(filename: str, content: bytes) -> Dict[str, Any]:
    """Validate and persist an uploaded dataset JSON to the scripts directory.

    Returns DatasetInfo-compatible dict. Avoids clobbering existing files by
    appending a numeric suffix if the sanitized name already exists.
    """
    try:
        parsed = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"File bukan JSON valid: {e}")

    questions = _validate_questions(parsed)

    base = Path(filename or "dataset.json").name
    if not base.lower().endswith(".json"):
        base += ".json"
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    target = DATASET_DIR / base
    stem = target.stem
    idx = 1
    while target.exists():
        target = DATASET_DIR / f"{stem}_{idx}.json"
        idx += 1

    # Normalise to a plain list on disk
    target.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "name": target.name,
        "num_questions": len(questions),
        "has_ground_truth": any(q.get("ground_truth") for q in questions),
    }


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def _bleu4(generated: str, reference: str) -> float:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize

    if not reference or not generated:
        return 0.0
    try:
        ref_tokens = word_tokenize(reference.lower())
        gen_tokens = word_tokenize(generated.lower())
        if len(gen_tokens) < 4:
            weights = (0.5, 0.5, 0, 0) if len(gen_tokens) >= 2 else (1, 0, 0, 0)
        else:
            weights = (0.25, 0.25, 0.25, 0.25)
        smoothie = SmoothingFunction().method1
        return float(sentence_bleu([ref_tokens], gen_tokens, weights=weights, smoothing_function=smoothie))
    except Exception as e:
        logger.warning(f"BLEU error: {e}")
        return 0.0


def _ensure_nltk():
    import nltk
    for pkg in ("tokenizers/punkt", "tokenizers/punkt_tab"):
        try:
            nltk.data.find(pkg)
        except LookupError:
            nltk.download(pkg.split("/")[-1], quiet=True)


def _rouge_scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def _bertscore(generated_list: List[str], reference_list: List[str]) -> Dict[str, Any]:
    """Batch BERTScore. Returns aggregate P/R/F1 and per-sample F1 list (aligned to valid pairs)."""
    from bert_score import score as bert_score_fn

    idx_valid = [i for i, (g, r) in enumerate(zip(generated_list, reference_list)) if g and r]
    if not idx_valid:
        return {"skipped": True}

    gen = [generated_list[i] for i in idx_valid]
    ref = [reference_list[i] for i in idx_valid]
    P, R, F1 = bert_score_fn(
        gen, ref, lang="id", model_type="bert-base-multilingual-cased", verbose=False
    )
    per_sample = {i: float(f) for i, f in zip(idx_valid, F1.tolist())}
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item()),
        "per_sample_f1": per_sample,
    }


def _run_ragas(eval_data: Dict[str, List], use_ground_truth: bool) -> Dict[str, Any]:
    """Run RAGAS (Faithfulness, Answer Relevancy, Context Precision) using Gemini as judge.

    Returns aggregate scores + per-sample lists keyed by metric.
    """
    # Patch missing langchain_community.chat_models.vertexai in newer langchain versions
    import sys
    import types
    try:
        import langchain_community
        if not hasattr(langchain_community, "chat_models"):
            langchain_community.chat_models = types.ModuleType("langchain_community.chat_models")
            sys.modules["langchain_community.chat_models"] = langchain_community.chat_models
        vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
        vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai
    except Exception:
        pass

    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    from app.config import settings

    google_api_key = os.environ.get("GOOGLE_API_KEY") or getattr(settings, "GOOGLE_API_KEY", "")
    if not google_api_key:
        return {"error": "GOOGLE_API_KEY not set"}

    gemini_model = os.environ.get("GEMINI_MODEL") or getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash")

    # NOTE: RAGAS Answer Relevancy embeds text via langchain's GoogleGenerativeAIEmbeddings,
    # which uses the Generative Language API (v1beta). That API needs a "models/text-embedding-*"
    # name — NOT the app's Vertex model (e.g. text-multilingual-embedding-002), which 404s here.
    ragas_embedding_model = os.environ.get("RAGAS_EMBEDDING_MODEL") or "models/gemini-embedding-001"
    if not ragas_embedding_model.startswith("models/"):
        ragas_embedding_model = f"models/{ragas_embedding_model}"

    llm = ChatGoogleGenerativeAI(model=gemini_model, google_api_key=google_api_key, temperature=0.0)
    embeddings = GoogleGenerativeAIEmbeddings(model=ragas_embedding_model, google_api_key=google_api_key)

    dataset = Dataset.from_dict({
        "question": eval_data["question"],
        "answer": eval_data["answer"],
        "contexts": eval_data["contexts"],
        "ground_truth": eval_data["ground_truth"],
    })

    metrics = [faithfulness, answer_relevancy]
    if use_ground_truth:
        metrics.append(context_precision)

    results = ragas_evaluate(dataset, metrics=metrics, llm=llm, embeddings=embeddings)

    # Extract per-sample + mean via pandas (most reliable across RAGAS versions)
    out: Dict[str, Any] = {}
    try:
        df = results.to_pandas()
        for key in ("faithfulness", "answer_relevancy", "context_precision"):
            if key in df.columns:
                col = [None if v != v else float(v) for v in df[key].tolist()]  # NaN -> None
                out[key] = col
    except Exception as e:
        logger.warning(f"RAGAS to_pandas failed: {e}")
    return out


# --------------------------------------------------------------------------- #
# Main task
# --------------------------------------------------------------------------- #
def run_evaluation_task(run_id: int) -> None:
    """Background task: execute an evaluation run end-to-end and persist results."""
    db = SessionLocal()
    try:
        run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
        if not run:
            logger.error(f"Evaluation run {run_id} not found")
            return

        config = run.config or {}
        metrics = set(config.get("metrics", ["bertscore", "bleu", "rouge", "ragas"]))
        use_gt = bool(config.get("ragas_use_ground_truth", True))
        limit = config.get("limit")
        default_division_id = config.get("division_id")
        default_category_id = config.get("category_id")

        run.status = "running"
        db.commit()

        questions = _load_dataset(run.dataset_name)
        if limit:
            questions = questions[: int(limit)]
        run.num_samples = len(questions)
        db.commit()

        # --- 1. Collect RAG responses (creates item rows, updates progress) ---
        # Use the ASYNC pipeline (same path production uses) because only that path
        # honours the `filters` argument (division/category scope). The sync
        # generate_response() ignores filters. Reuse one event loop for all questions.
        from app.rag.chain import rag_chain

        loop = asyncio.new_event_loop()
        items: List[EvaluationItem] = []
        eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

        for i, q in enumerate(questions, 1):
            question = q.get("question") or q.get("query") or ""
            ground_truth = q.get("ground_truth", "") or ""

            # Retrieval scope: per-question override (dataset) falls back to run-level default.
            # Mirrors production, where retrieval is filtered by the user's division.
            q_division = q.get("division_id", default_division_id)
            q_category = q.get("category_id", default_category_id)
            filters = {}
            if q_division is not None:
                filters["division_id"] = q_division
            if q_category is not None:
                filters["category_id"] = q_category

            item = EvaluationItem(
                run_id=run.id,
                question_ref=str(q["id"]) if q.get("id") is not None else None,
                question=question,
                ground_truth=ground_truth,
                category=q.get("category"),
                division_id=q_division,
                category_id=q_category,
            )
            try:
                resp = loop.run_until_complete(
                    rag_chain.generate_response_async(
                        query=question, conversation_history=[], filters=filters or None
                    )
                )
                answer = resp.get("answer") or resp.get("text") or ""
                if isinstance(answer, list):
                    answer = " ".join(answer)
                # FIX: use retrieved chunk TEXT (page_content), not document titles
                source_docs = resp.get("source_documents", []) or []
                contexts = [getattr(d, "page_content", "") for d in source_docs if getattr(d, "page_content", "")]
                if not contexts:
                    contexts = ["No context retrieved"]
                item.answer = answer
                item.contexts = contexts
                item.sources = resp.get("sources_metadata", [])
                item.total_time_ms = resp.get("total_time_ms", 0)
            except Exception as e:
                logger.error(f"RAG response error (q{i}): {e}", exc_info=True)
                item.answer = ""
                item.contexts = ["Error"]
                item.error = str(e)

            db.add(item)
            items.append(item)
            eval_data["question"].append(question)
            eval_data["answer"].append(item.answer or "")
            eval_data["contexts"].append(item.contexts)
            eval_data["ground_truth"].append(ground_truth)

            run.processed_samples = i
            db.commit()
            logger.info(f"[eval {run_id}] collected {i}/{len(questions)}")

        loop.close()
        has_gt = any(eval_data["ground_truth"])

        # --- 2. BLEU-4 + ROUGE-L (per item, deterministic) ---
        if has_gt and ("bleu" in metrics or "rouge" in metrics):
            _ensure_nltk()
            scorer = _rouge_scorer() if "rouge" in metrics else None
            for item, ref in zip(items, eval_data["ground_truth"]):
                gen = item.answer or ""
                if not ref or not gen:
                    continue
                if "bleu" in metrics:
                    item.bleu = _bleu4(gen, ref)
                if scorer is not None:
                    try:
                        s = scorer.score(ref, gen)
                        item.rougeL = float(s["rougeL"].fmeasure)
                    except Exception as e:
                        logger.warning(f"ROUGE error: {e}")
            db.commit()

        # --- 3. BERTScore (batch) ---
        bert_agg = {}
        if has_gt and "bertscore" in metrics:
            try:
                bert_agg = _bertscore(eval_data["answer"], eval_data["ground_truth"])
                per = bert_agg.get("per_sample_f1", {})
                for idx, item in enumerate(items):
                    if idx in per:
                        item.bertscore_f1 = per[idx]
                db.commit()
            except Exception as e:
                logger.error(f"BERTScore failed: {e}", exc_info=True)
                bert_agg = {"error": str(e)}

        # --- 4. RAGAS (batch) ---
        ragas_res = {}
        if "ragas" in metrics:
            try:
                ragas_res = _run_ragas(eval_data, use_ground_truth=use_gt and has_gt)
                for key, attr in (
                    ("faithfulness", "ragas_faithfulness"),
                    ("answer_relevancy", "ragas_answer_relevancy"),
                    ("context_precision", "ragas_context_precision"),
                ):
                    col = ragas_res.get(key)
                    if col:
                        for item, val in zip(items, col):
                            setattr(item, attr, val)
                db.commit()
            except Exception as e:
                logger.error(f"RAGAS failed: {e}", exc_info=True)
                ragas_res = {"error": str(e)}

        # --- 5. Aggregate ---
        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        if "bleu" in metrics:
            run.bleu = _mean([it.bleu for it in items])
        if "rouge" in metrics:
            run.rougeL = _mean([it.rougeL for it in items])
        if bert_agg and not bert_agg.get("error") and not bert_agg.get("skipped"):
            run.bertscore_f1 = bert_agg.get("f1")
            run.bertscore_precision = bert_agg.get("precision")
            run.bertscore_recall = bert_agg.get("recall")
        if "ragas" in metrics and not ragas_res.get("error"):
            run.ragas_faithfulness = _mean([it.ragas_faithfulness for it in items])
            run.ragas_answer_relevancy = _mean([it.ragas_answer_relevancy for it in items])
            run.ragas_context_precision = _mean([it.ragas_context_precision for it in items])
        run.avg_total_ms = _mean([it.total_time_ms for it in items])

        # Record any partial errors in the run row (non-fatal)
        partial_errors = []
        if bert_agg.get("error"):
            partial_errors.append(f"BERTScore: {bert_agg['error']}")
        if ragas_res.get("error"):
            partial_errors.append(f"RAGAS: {ragas_res['error']}")
        if partial_errors:
            run.error = " | ".join(partial_errors)

        run.status = "completed"
        run.completed_at = datetime.utcnow()
        db.commit()
        logger.info(f"[eval {run_id}] completed")

    except Exception as e:
        logger.error(f"Evaluation run {run_id} failed: {e}", exc_info=True)
        try:
            run = db.query(EvaluationRun).filter(EvaluationRun.id == run_id).first()
            if run:
                run.status = "failed"
                run.error = str(e)
                run.completed_at = datetime.utcnow()
                db.commit()
        except Exception:
            pass
    finally:
        db.close()
