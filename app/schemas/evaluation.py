"""Evaluation schemas."""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class EvaluationRunCreate(BaseModel):
    """Request to start a new evaluation run."""
    dataset_name: str = "evaluation_dataset.json"
    run_name: Optional[str] = None
    metrics: List[str] = ["bertscore", "bleu", "rouge", "ragas"]
    ragas_use_ground_truth: bool = True
    limit: Optional[int] = None  # run only first N questions (for quick testing)
    # Text normalization for BLEU/ROUGE only: "none" | "basic" | "strong"
    lexical_normalization: str = "basic"
    # Retrieval scope — mirrors production per-user access control.
    # Per-question division_id/category_id in the dataset override these run-level defaults.
    division_id: Optional[int] = None
    category_id: Optional[int] = None


class EvaluationItemResponse(BaseModel):
    """Per-question result."""
    id: int
    question_ref: Optional[str] = None
    question: str
    ground_truth: Optional[str] = None
    answer: Optional[str] = None
    contexts: Optional[List[str]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    category: Optional[str] = None
    division_id: Optional[int] = None
    category_id: Optional[int] = None
    bertscore_f1: Optional[float] = None
    bleu: Optional[float] = None
    rougeL: Optional[float] = None
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    total_time_ms: Optional[int] = None
    error: Optional[str] = None

    class Config:
        from_attributes = True


class EvaluationRunSummary(BaseModel):
    """Aggregate view of a run (list rows)."""
    id: int
    run_name: Optional[str] = None
    dataset_name: Optional[str] = None
    status: str
    num_samples: int
    processed_samples: int
    division_id: Optional[int] = None
    category_id: Optional[int] = None
    bertscore_f1: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bleu: Optional[float] = None
    rougeL: Optional[float] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    ragas_faithfulness: Optional[float] = None
    ragas_answer_relevancy: Optional[float] = None
    ragas_context_precision: Optional[float] = None
    avg_total_ms: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EvaluationRunDetail(EvaluationRunSummary):
    """Full run with per-question items."""
    items: List[EvaluationItemResponse] = []


class DatasetInfo(BaseModel):
    name: str
    num_questions: int
    has_ground_truth: bool
