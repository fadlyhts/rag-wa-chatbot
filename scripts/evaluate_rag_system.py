"""
RAG System Evaluation Script
============================
Evaluates your RAG system using multiple metrics:
- RAGAS (Faithfulness, Answer Relevancy, Context Precision)
- BERTScore
- BLEU-4
- ROUGE-L

Usage:
    python scripts/evaluate_rag_system.py
    python scripts/evaluate_rag_system.py --dataset evaluation_data.json
    python scripts/evaluate_rag_system.py --quick  # Quick test with sample data
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check and import dependencies
def check_dependencies():
    """Check if evaluation dependencies are installed"""
    missing = []
    
    try:
        import nltk
    except ImportError:
        missing.append("nltk")
    
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        missing.append("rouge-score")
    
    try:
        from bert_score import score as bert_score
    except ImportError:
        missing.append("bert-score")
    
    try:
        # Patch missing langchain_community.chat_models.vertexai in newer langchain versions
        import sys
        import types
        if 'langchain_community' in sys.modules:
            pass
        else:
            import langchain_community
        
        if not hasattr(langchain_community, 'chat_models'):
            langchain_community.chat_models = types.ModuleType('langchain_community.chat_models')
            sys.modules['langchain_community.chat_models'] = langchain_community.chat_models
            
        vertexai = types.ModuleType('langchain_community.chat_models.vertexai')
        vertexai.ChatVertexAI = type('ChatVertexAI', (object,), {})
        sys.modules['langchain_community.chat_models.vertexai'] = vertexai
        
        from ragas import evaluate as ragas_evaluate
    except ImportError as e:
        print(f"Ragas import error: {e}")
        missing.append("ragas")
    
    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install -r requirements-evaluation.txt")
        print("\nOr manually:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 60)
        return False
    
    return True


class RAGEvaluator:
    """
    Comprehensive RAG System Evaluator
    
    Metrics:
    - BLEU-4: N-gram precision
    - ROUGE-L: Longest common subsequence
    - BERTScore: Semantic similarity
    - RAGAS: Faithfulness, Answer Relevancy, Context Precision
    """
    
    def __init__(self, use_ragas: bool = True):
        self.use_ragas = use_ragas
        self.results = {}
        
        # Initialize scorers
        from rouge_score import rouge_scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Download NLTK data
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    
    def collect_rag_responses(self, questions: List[Dict]) -> Dict[str, List]:
        """
        Run RAG system on evaluation questions
        
        Args:
            questions: List of dicts with 'question' and optionally 'ground_truth'
            
        Returns:
            Dict with questions, answers, contexts, ground_truths
        """
        from app.rag.chain import rag_chain
        
        eval_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
            "retrieval_time_ms": [],
            "generation_time_ms": [],
            "total_time_ms": [],
            "sources": []
        }
        
        print(f"\n{'='*60}")
        print(f"Collecting RAG responses for {len(questions)} questions...")
        print(f"{'='*60}\n")
        
        for i, q in enumerate(questions, 1):
            question = q.get('question') or q.get('query')
            print(f"[{i}/{len(questions)}] {question[:60]}...")
            
            try:
                # Get RAG response
                response = rag_chain.generate_response(
                    query=question,
                    conversation_history=[]
                )
                
                # Extract answer
                answer = response.get('text', '')
                if isinstance(answer, list):
                    answer = ' '.join(answer)
                
                # Extract contexts from sources
                contexts = []
                sources = response.get('sources', [])
                for source in sources:
                    title = source.get('title', '')
                    contexts.append(title)
                
                # Store results
                eval_data["question"].append(question)
                eval_data["answer"].append(answer)
                eval_data["contexts"].append(contexts if contexts else ["No context retrieved"])
                eval_data["ground_truth"].append(q.get('ground_truth', ''))
                eval_data["retrieval_time_ms"].append(response.get('retrieval_time_ms', 0))
                eval_data["generation_time_ms"].append(response.get('generation_time_ms', 0))
                eval_data["total_time_ms"].append(response.get('total_time_ms', 0))
                eval_data["sources"].append(sources)
                
                print(f"    Answer: {answer[:80]}...")
                print(f"    Time: {response.get('total_time_ms', 0)}ms")
                
            except Exception as e:
                print(f"    ERROR: {str(e)}")
                eval_data["question"].append(question)
                eval_data["answer"].append(f"Error: {str(e)}")
                eval_data["contexts"].append(["Error"])
                eval_data["ground_truth"].append(q.get('ground_truth', ''))
                eval_data["retrieval_time_ms"].append(0)
                eval_data["generation_time_ms"].append(0)
                eval_data["total_time_ms"].append(0)
                eval_data["sources"].append([])
        
        return eval_data
    
    def evaluate_bleu4(self, generated: str, reference: str) -> float:
        """Calculate BLEU-4 score"""
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        
        if not reference or not generated:
            return 0.0
        
        try:
            ref_tokens = word_tokenize(reference.lower())
            gen_tokens = word_tokenize(generated.lower())
            
            if len(gen_tokens) < 4:
                # Use lower n-grams for short sentences
                weights = (0.5, 0.5, 0, 0) if len(gen_tokens) >= 2 else (1, 0, 0, 0)
            else:
                weights = (0.25, 0.25, 0.25, 0.25)
            
            return sentence_bleu([ref_tokens], gen_tokens, weights=weights)
        except Exception as e:
            print(f"BLEU error: {e}")
            return 0.0
    
    def evaluate_rouge(self, generated: str, reference: str) -> Dict:
        """Calculate ROUGE scores"""
        if not reference or not generated:
            return {'rouge1_f1': 0, 'rouge2_f1': 0, 'rougeL_f1': 0}
        
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure,
                'rougeL_precision': scores['rougeL'].precision,
                'rougeL_recall': scores['rougeL'].recall
            }
        except Exception as e:
            print(f"ROUGE error: {e}")
            return {'rouge1_f1': 0, 'rouge2_f1': 0, 'rougeL_f1': 0}
    
    def evaluate_bertscore(self, generated_list: List[str], 
                           reference_list: List[str]) -> Dict:
        """Calculate BERTScore"""
        from bert_score import score as bert_score_fn
        
        # Filter out empty pairs
        valid_pairs = [
            (g, r) for g, r in zip(generated_list, reference_list)
            if g and r
        ]
        
        if not valid_pairs:
            return {'precision': 0, 'recall': 0, 'f1': 0, 'skipped': True}
        
        gen_filtered, ref_filtered = zip(*valid_pairs)
        
        try:
            print("  Calculating BERTScore (this may take a moment)...")
            P, R, F1 = bert_score_fn(
                list(gen_filtered), 
                list(ref_filtered), 
                lang="id",  # Indonesian - change to "en" for English
                model_type="bert-base-multilingual-cased",
                verbose=False
            )
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item(),
                'per_sample_f1': F1.tolist()
            }
        except Exception as e:
            print(f"BERTScore error: {e}")
            return {'precision': 0, 'recall': 0, 'f1': 0, 'error': str(e)}
    
    def evaluate_with_ragas(self, eval_data: Dict) -> Dict:
        """Evaluate using RAGAS metrics with Google Gemini as LLM evaluator"""
        if not self.use_ragas:
            return {'skipped': True, 'reason': 'RAGAS disabled'}
        
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision
            )
            from datasets import Dataset
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            
            print("  Running RAGAS evaluation (using Gemini)...")
            
            # Get Google API key from environment or app config
            google_api_key = os.environ.get("GOOGLE_API_KEY", "")
            if not google_api_key:
                try:
                    from app.config import settings
                    google_api_key = settings.GOOGLE_API_KEY
                except Exception:
                    pass
            
            if not google_api_key:
                return {'error': 'GOOGLE_API_KEY not found. Set it in .env or environment.'}
            
            # Use Gemini as LLM evaluator instead of OpenAI
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=google_api_key,
                temperature=0.0
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=google_api_key
            )
            
            # Prepare dataset
            dataset = Dataset.from_dict({
                "question": eval_data["question"],
                "answer": eval_data["answer"],
                "contexts": eval_data["contexts"],
                "ground_truth": eval_data["ground_truth"]
            })
            
            # Select metrics based on available data
            metrics = [faithfulness, answer_relevancy, context_precision]
            
            results = ragas_evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings
            )
            
            return {
                "faithfulness": results.get('faithfulness'),
                "answer_relevancy": results.get('answer_relevancy'),
                "context_precision": results.get('context_precision'),
            }
            
        except ImportError as e:
            missing_pkg = str(e)
            print(f"RAGAS import error: {missing_pkg}")
            print("  Install with: pip install langchain-google-genai")
            return {'error': f'Missing package: {missing_pkg}. Install: pip install langchain-google-genai'}
        except Exception as e:
            print(f"RAGAS error: {e}")
            return {'error': str(e)}
    
    def evaluate_latency(self, eval_data: Dict) -> Dict:
        """Evaluate system latency"""
        retrieval_times = [t for t in eval_data["retrieval_time_ms"] if t > 0]
        generation_times = [t for t in eval_data["generation_time_ms"] if t > 0]
        total_times = [t for t in eval_data["total_time_ms"] if t > 0]
        
        def calc_stats(times):
            if not times:
                return {'mean': 0, 'min': 0, 'max': 0}
            return {
                'mean': sum(times) / len(times),
                'min': min(times),
                'max': max(times)
            }
        
        return {
            'retrieval_ms': calc_stats(retrieval_times),
            'generation_ms': calc_stats(generation_times),
            'total_ms': calc_stats(total_times)
        }
    
    def run_evaluation(self, eval_data: Dict, include_ragas: bool = False) -> Dict:
        """
        Run all evaluations on collected data
        
        Args:
            eval_data: Dict with questions, answers, contexts, ground_truths
            include_ragas: Whether to include RAGAS (requires OpenAI API)
            
        Returns:
            Dict with all evaluation results
        """
        print(f"\n{'='*60}")
        print("Running Evaluations...")
        print(f"{'='*60}\n")
        
        n = len(eval_data["question"])
        has_ground_truth = any(eval_data["ground_truth"])
        
        # BLEU-4 and ROUGE-L (require ground truth)
        bleu_scores = []
        rouge_scores = []
        
        if has_ground_truth:
            print("Calculating BLEU-4 and ROUGE-L...")
            for gen, ref in zip(eval_data["answer"], eval_data["ground_truth"]):
                if ref:
                    bleu_scores.append(self.evaluate_bleu4(gen, ref))
                    rouge_scores.append(self.evaluate_rouge(gen, ref))
        else:
            print("Skipping BLEU-4 and ROUGE-L (no ground truth provided)")
        
        # BERTScore (requires ground truth)
        if has_ground_truth:
            bert_results = self.evaluate_bertscore(
                eval_data["answer"], 
                eval_data["ground_truth"]
            )
        else:
            bert_results = {'skipped': True, 'reason': 'No ground truth'}
        
        # RAGAS (optional)
        if include_ragas and self.use_ragas:
            ragas_results = self.evaluate_with_ragas(eval_data)
        else:
            ragas_results = {'skipped': True, 'reason': 'RAGAS not requested'}
        
        # Latency
        latency_results = self.evaluate_latency(eval_data)
        
        # Compile results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'num_samples': n,
            'has_ground_truth': has_ground_truth,
            'bleu4': {
                'mean': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0,
                'min': min(bleu_scores) if bleu_scores else 0,
                'max': max(bleu_scores) if bleu_scores else 0,
                'per_sample': bleu_scores
            } if bleu_scores else {'skipped': True},
            'rouge': {
                'rouge1_f1_mean': sum(r['rouge1_f1'] for r in rouge_scores) / len(rouge_scores) if rouge_scores else 0,
                'rouge2_f1_mean': sum(r['rouge2_f1'] for r in rouge_scores) / len(rouge_scores) if rouge_scores else 0,
                'rougeL_f1_mean': sum(r['rougeL_f1'] for r in rouge_scores) / len(rouge_scores) if rouge_scores else 0,
            } if rouge_scores else {'skipped': True},
            'bertscore': bert_results,
            'ragas': ragas_results,
            'latency': latency_results,
            'raw_data': {
                'questions': eval_data['question'],
                'answers': eval_data['answer'],
                'ground_truths': eval_data['ground_truth'] if has_ground_truth else None
            }
        }
        
        return self.results
    
    def print_summary(self):
        """Print evaluation summary"""
        r = self.results
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        
        print(f"\nTimestamp: {r['timestamp']}")
        print(f"Samples: {r['num_samples']}")
        print(f"Ground Truth: {'Yes' if r['has_ground_truth'] else 'No'}")
        
        # BLEU-4
        print(f"\n--- BLEU-4 ---")
        if r['bleu4'].get('skipped'):
            print("  Skipped (no ground truth)")
        else:
            print(f"  Mean:  {r['bleu4']['mean']:.4f}")
            print(f"  Range: {r['bleu4']['min']:.4f} - {r['bleu4']['max']:.4f}")
        
        # ROUGE
        print(f"\n--- ROUGE ---")
        if r['rouge'].get('skipped'):
            print("  Skipped (no ground truth)")
        else:
            print(f"  ROUGE-1 F1: {r['rouge']['rouge1_f1_mean']:.4f}")
            print(f"  ROUGE-2 F1: {r['rouge']['rouge2_f1_mean']:.4f}")
            print(f"  ROUGE-L F1: {r['rouge']['rougeL_f1_mean']:.4f}")
        
        # BERTScore
        print(f"\n--- BERTScore ---")
        if r['bertscore'].get('skipped') or r['bertscore'].get('error'):
            print(f"  Skipped: {r['bertscore'].get('reason', r['bertscore'].get('error', 'Unknown'))}")
        else:
            print(f"  Precision: {r['bertscore']['precision']:.4f}")
            print(f"  Recall:    {r['bertscore']['recall']:.4f}")
            print(f"  F1:        {r['bertscore']['f1']:.4f}")
        
        # RAGAS
        print(f"\n--- RAGAS ---")
        if r['ragas'].get('skipped') or r['ragas'].get('error'):
            print(f"  Skipped: {r['ragas'].get('reason', r['ragas'].get('error', 'Unknown'))}")
        else:
            if r['ragas'].get('faithfulness') is not None:
                print(f"  Faithfulness:      {r['ragas']['faithfulness']:.4f}")
            if r['ragas'].get('answer_relevancy') is not None:
                print(f"  Answer Relevancy:  {r['ragas']['answer_relevancy']:.4f}")
            if r['ragas'].get('context_precision') is not None:
                print(f"  Context Precision: {r['ragas']['context_precision']:.4f}")
        
        # Latency
        print(f"\n--- Latency ---")
        lat = r['latency']
        print(f"  Retrieval:  {lat['retrieval_ms']['mean']:.0f}ms (avg)")
        print(f"  Generation: {lat['generation_ms']['mean']:.0f}ms (avg)")
        print(f"  Total:      {lat['total_ms']['mean']:.0f}ms (avg)")
        
        print(f"\n{'='*60}\n")
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        # Convert non-serializable items
        results_copy = json.loads(json.dumps(self.results, default=str))
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {filepath}")


def get_sample_questions() -> List[Dict]:
    """Get sample questions for quick testing"""
    return [
        {
            "question": "Apa jam operasional kantor?",
            "ground_truth": "Kantor beroperasi Senin-Jumat, pukul 08:00-17:00 WIB"
        },
        {
            "question": "Bagaimana cara mengajukan cuti?",
            "ground_truth": "Pengajuan cuti dilakukan melalui sistem HRIS"
        },
        {
            "question": "Apa saja layanan yang tersedia?",
            "ground_truth": ""  # No ground truth - will skip BLEU/ROUGE/BERTScore for this
        }
    ]


def load_evaluation_dataset(filepath: str) -> List[Dict]:
    """Load evaluation dataset from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'questions' in data:
        return data['questions']
    else:
        raise ValueError("Invalid dataset format. Expected list of questions or {'questions': [...]}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG System')
    parser.add_argument('--dataset', type=str, help='Path to evaluation dataset JSON')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file path')
    parser.add_argument('--quick', action='store_true', help='Quick test with sample data')
    parser.add_argument('--ragas', action='store_true', help='Include RAGAS evaluation (requires OpenAI API)')
    parser.add_argument('--no-collect', action='store_true', help='Skip RAG collection, use provided answers')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("\n" + "="*60)
    print("RAG SYSTEM EVALUATION")
    print("="*60)
    
    # Load or create questions
    if args.quick:
        print("\nUsing sample questions for quick test...")
        questions = get_sample_questions()
    elif args.dataset:
        print(f"\nLoading dataset from: {args.dataset}")
        questions = load_evaluation_dataset(args.dataset)
    else:
        print("\nNo dataset provided. Use --quick for sample or --dataset <path>")
        print("\nExample usage:")
        print("  python scripts/evaluate_rag_system.py --quick")
        print("  python scripts/evaluate_rag_system.py --dataset my_questions.json")
        print("  python scripts/evaluate_rag_system.py --dataset my_questions.json --ragas")
        sys.exit(0)
    
    print(f"Questions loaded: {len(questions)}")
    
    # Initialize evaluator
    evaluator = RAGEvaluator(use_ragas=args.ragas)
    
    # Collect RAG responses
    eval_data = evaluator.collect_rag_responses(questions)
    
    # Run evaluation
    results = evaluator.run_evaluation(eval_data, include_ragas=args.ragas)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(args.output)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
