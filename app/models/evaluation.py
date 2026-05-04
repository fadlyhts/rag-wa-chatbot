"""Evaluation models for storing evaluation results"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Index, ForeignKey
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.base import Base


class EvaluationRun(Base):
    """Model for evaluation runs"""
    
    __tablename__ = "evaluation_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String(200), nullable=False, index=True)
    dataset_name = Column(String(200), nullable=True)
    dataset_size = Column(Integer, nullable=False)
    
    # Configuration
    ai_provider = Column(String(50), nullable=True)
    evaluation_model = Column(String(100), nullable=True)
    enabled_metrics = Column(JSON, nullable=True)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    evaluation_time_seconds = Column(Float, nullable=True)
    
    # Status
    status = Column(String(20), default="running", nullable=False, index=True)  # running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Aggregated results
    faithfulness_mean = Column(Float, nullable=True)
    answer_relevancy_mean = Column(Float, nullable=True)
    context_precision_mean = Column(Float, nullable=True)
    context_recall_mean = Column(Float, nullable=True)
    
    # Full results (JSON)
    aggregated_metrics = Column(JSON, nullable=True)
    results_file_path = Column(String(500), nullable=True)
    
    # Relationships
    sample_results = relationship("EvaluationSampleResult", back_populates="run", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_run_status', 'status', 'started_at'),
    )
    
    def __repr__(self):
        return f"<EvaluationRun(id={self.id}, run_name={self.run_name}, status={self.status})>"


class EvaluationSampleResult(Base):
    """Model for individual sample evaluation results"""
    
    __tablename__ = "evaluation_sample_results"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Sample data
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    contexts = Column(JSON, nullable=True)  # List of context strings
    ground_truth = Column(Text, nullable=True)
    
    # Scores
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    
    # Metadata
    sample_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    run = relationship("EvaluationRun", back_populates="sample_results")
    
    __table_args__ = (
        Index('idx_run_scores', 'run_id', 'faithfulness'),
    )
    
    def __repr__(self):
        return f"<EvaluationSampleResult(id={self.id}, run_id={self.run_id})>"


class UserFeedback(Base):
    """Model for user feedback on RAG responses"""
    
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Message reference
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(Integer, ForeignKey("messages.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Feedback
    rating = Column(Integer, nullable=True)  # 1-5 stars
    feedback_type = Column(String(50), nullable=True, index=True)  # helpful, not_helpful, incorrect, etc.
    feedback_text = Column(Text, nullable=True)
    
    # Correction (if user provides correct answer)
    corrected_answer = Column(Text, nullable=True)
    
    # Original Q&A for context
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    conversation = relationship("Conversation")
    message = relationship("Message")
    
    __table_args__ = (
        Index('idx_feedback_rating', 'rating', 'created_at'),
        Index('idx_feedback_type', 'feedback_type', 'created_at'),
    )
    
    def __repr__(self):
        return f"<UserFeedback(id={self.id}, rating={self.rating}, type={self.feedback_type})>"
