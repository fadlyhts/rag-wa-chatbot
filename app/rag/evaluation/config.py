"""RAGAS Evaluation Configuration"""

from dataclasses import dataclass
from typing import List, Optional
from app.rag.config import rag_config


@dataclass
class EvaluationConfig:
    """Configuration for RAG evaluation"""
    
    # AI Provider for RAGAS evaluation (inherits from RAG config)
    ai_provider: str = rag_config.ai_provider
    
    # LLM settings for RAGAS evaluators
    openai_api_key: Optional[str] = rag_config.openai_api_key
    google_api_key: Optional[str] = rag_config.google_api_key
    
    # Model for evaluation (can be different from generation model)
    evaluation_model: str = rag_config.gemini_model if rag_config.ai_provider == "gemini" else rag_config.llm_model
    
    # Metrics to evaluate
    enabled_metrics: List[str] = None
    
    # Test dataset paths
    test_sets_dir: str = "evaluation_data/test_sets"
    results_dir: str = "evaluation_data/results"
    
    # Evaluation settings
    batch_size: int = 10
    enable_async: bool = True
    save_results: bool = True
    
    # Production sampling
    sample_rate: float = 0.05  # Sample 5% of production queries
    enable_production_eval: bool = False
    
    # Thresholds for alerts
    min_faithfulness: float = 0.7
    min_answer_relevancy: float = 0.7
    min_context_precision: float = 0.6
    min_context_recall: float = 0.6
    
    def __post_init__(self):
        """Initialize default metrics if not specified"""
        if self.enabled_metrics is None:
            self.enabled_metrics = [
                'faithfulness',
                'answer_relevancy',
                'context_precision',
                'context_recall'
            ]


# Global evaluation config instance
evaluation_config = EvaluationConfig()
