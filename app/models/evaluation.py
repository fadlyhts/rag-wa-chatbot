"""Evaluation models — persistent record of automatic-metric evaluation runs.

Each EvaluationRun represents one execution of the test dataset against the RAG
pipeline. Each EvaluationItem stores a single question's result (answer, retrieved
contexts, and per-metric scores) so every run is fully auditable and reproducible.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey
from sqlalchemy.dialects.mysql import JSON, LONGTEXT
from sqlalchemy.orm import relationship
from datetime import datetime

from app.database.base import Base


class EvaluationRun(Base):
    """One evaluation run over a test dataset (aggregate scores + metadata)."""

    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(255), nullable=True)
    dataset_name = Column(String(255), nullable=True)

    # pending -> running -> completed / failed
    status = Column(String(20), default="pending", index=True)

    # Snapshot of the config used (metrics enabled, models, ragas mode, ...)
    config = Column(JSON, nullable=True)

    # Run-level retrieval scope (mirrors production per-user access control)
    division_id = Column(Integer, nullable=True)
    category_id = Column(Integer, nullable=True)

    num_samples = Column(Integer, default=0, nullable=False)
    processed_samples = Column(Integer, default=0, nullable=False)  # for progress bar

    # Aggregate scores (nullable — a metric may be skipped)
    bertscore_f1 = Column(Float, nullable=True)
    bertscore_precision = Column(Float, nullable=True)
    bertscore_recall = Column(Float, nullable=True)
    bleu = Column(Float, nullable=True)
    rougeL = Column(Float, nullable=True)
    rouge1 = Column(Float, nullable=True)
    rouge2 = Column(Float, nullable=True)
    ragas_faithfulness = Column(Float, nullable=True)
    ragas_answer_relevancy = Column(Float, nullable=True)
    ragas_context_precision = Column(Float, nullable=True)
    avg_total_ms = Column(Float, nullable=True)

    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)

    items = relationship(
        "EvaluationItem",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="EvaluationItem.id",
    )

    def __repr__(self):
        return f"<EvaluationRun(id={self.id}, status={self.status}, n={self.num_samples})>"


class EvaluationItem(Base):
    """Per-question result within an evaluation run."""

    __tablename__ = "evaluation_items"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(
        Integer, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Stable id from the dataset (e.g. "TEH-01") — for reference / export cross-matching
    question_ref = Column(String(64), nullable=True, index=True)

    question = Column(Text, nullable=False)
    ground_truth = Column(LONGTEXT, nullable=True)
    answer = Column(LONGTEXT, nullable=True)

    # Retrieved chunk texts (list[str]) — used by RAGAS Faithfulness / Context Precision
    contexts = Column(JSON, nullable=True)
    # Source metadata (title, page, score, ...)
    sources = Column(JSON, nullable=True)
    category = Column(String(100), nullable=True)

    # Effective retrieval filter applied for this question (per-question override or run default)
    division_id = Column(Integer, nullable=True)
    category_id = Column(Integer, nullable=True)

    # Per-item scores (nullable)
    bertscore_f1 = Column(Float, nullable=True)
    bleu = Column(Float, nullable=True)
    rougeL = Column(Float, nullable=True)
    ragas_faithfulness = Column(Float, nullable=True)
    ragas_answer_relevancy = Column(Float, nullable=True)
    ragas_context_precision = Column(Float, nullable=True)

    total_time_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)

    run = relationship("EvaluationRun", back_populates="items")

    def __repr__(self):
        return f"<EvaluationItem(id={self.id}, run_id={self.run_id})>"
