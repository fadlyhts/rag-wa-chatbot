"""RAG Evaluation Service using RAGAS"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import asyncio

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# LangChain integrations for RAGAS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.rag.evaluation.config import evaluation_config
from app.rag.evaluation.dataset import EvaluationDataset
from app.rag.chain import rag_chain
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for RAG system using RAGAS metrics"""
    
    def __init__(self):
        self.config = evaluation_config
        self._setup_llm()
        self._setup_embeddings()
        self._setup_metrics()
    
    def _setup_llm(self):
        """Setup LLM for RAGAS evaluation"""
        try:
            if self.config.ai_provider == "gemini":
                # LangChain Google GenAI expects model name without "models/" prefix
                model_name = self.config.evaluation_model
                if model_name.startswith("models/"):
                    model_name = model_name.replace("models/", "")
                
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.config.google_api_key,
                    temperature=0.0,  # Deterministic for evaluation
                    convert_system_message_to_human=True  # Required for Gemini compatibility
                )
                logger.info(f"Initialized Gemini LLM for evaluation: {model_name}")
            else:
                self.llm = ChatOpenAI(
                    model=self.config.evaluation_model,
                    api_key=self.config.openai_api_key,
                    temperature=0.0
                )
                logger.info(f"Initialized OpenAI LLM for evaluation: {self.config.evaluation_model}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _setup_embeddings(self):
        """Setup embeddings for RAGAS evaluation"""
        try:
            if self.config.ai_provider == "gemini":
                # Use the same embedding model as RAG config (with models/ prefix)
                embedding_model = rag_config.gemini_embedding_model
                if not embedding_model.startswith("models/"):
                    embedding_model = f"models/{embedding_model}"
                
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model=embedding_model,
                    google_api_key=self.config.google_api_key
                )
                logger.info(f"Initialized Gemini embeddings for evaluation: {embedding_model}")
            else:
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=self.config.openai_api_key
                )
                logger.info("Initialized OpenAI embeddings for evaluation")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise
    
    def _setup_metrics(self):
        """Setup RAGAS metrics based on configuration"""
        self.metrics = []
        
        if 'faithfulness' in self.config.enabled_metrics:
            self.metrics.append(faithfulness)
        if 'answer_relevancy' in self.config.enabled_metrics:
            self.metrics.append(answer_relevancy)
        if 'context_precision' in self.config.enabled_metrics:
            self.metrics.append(context_precision)
        if 'context_recall' in self.config.enabled_metrics:
            self.metrics.append(context_recall)
        
        logger.info(f"Enabled metrics: {[m.name for m in self.metrics]}")
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single Q&A pair
        
        Args:
            question: User query
            answer: Generated answer
            contexts: Retrieved context chunks
            ground_truth: Expected answer (optional, needed for context_recall)
            
        Returns:
            Dict with metric scores
        """
        try:
            # Create single-item dataset
            data = {
                'question': [question],
                'answer': [answer],
                'contexts': [contexts],
                'ground_truth': [ground_truth or '']
            }
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            logger.info(f"Evaluating single response: {question[:50]}...")
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Extract scores
            scores = result.to_pandas().iloc[0].to_dict()
            logger.info(f"Evaluation complete: {scores}")
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in single evaluation: {e}", exc_info=True)
            return {'error': str(e)}
    
    async def evaluate_single_async(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async wrapper for single evaluation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.evaluate_single,
            question,
            answer,
            contexts,
            ground_truth
        )
    
    def evaluate_dataset(
        self,
        dataset: EvaluationDataset
    ) -> Dict[str, Any]:
        """
        Evaluate entire dataset
        
        Args:
            dataset: EvaluationDataset instance
            
        Returns:
            Dict with aggregated metrics and per-sample scores
        """
        try:
            if len(dataset) == 0:
                logger.warning("Empty dataset provided")
                return {'error': 'Empty dataset'}
            
            # Convert to RAGAS format
            ragas_dataset = dataset.to_ragas_dataset()
            
            # Run evaluation
            logger.info(f"Evaluating dataset with {len(dataset)} samples...")
            start_time = datetime.now()
            
            result = evaluate(
                dataset=ragas_dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            eval_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to DataFrame and dict
            df = result.to_pandas()
            
            # Aggregate metrics (mean)
            aggregated = {}
            for metric in self.metrics:
                metric_name = metric.name
                if metric_name in df.columns:
                    aggregated[f'{metric_name}_mean'] = float(df[metric_name].mean())
                    aggregated[f'{metric_name}_std'] = float(df[metric_name].std())
                    aggregated[f'{metric_name}_min'] = float(df[metric_name].min())
                    aggregated[f'{metric_name}_max'] = float(df[metric_name].max())
            
            # Per-sample scores
            per_sample_scores = df.to_dict('records')
            
            result_dict = {
                'aggregated_metrics': aggregated,
                'per_sample_scores': per_sample_scores,
                'dataset_size': len(dataset),
                'evaluation_time_seconds': eval_time,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Dataset evaluation complete in {eval_time:.2f}s")
            logger.info(f"Aggregated metrics: {aggregated}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in dataset evaluation: {e}", exc_info=True)
            return {'error': str(e)}
    
    def evaluate_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load dataset from file and evaluate
        
        Args:
            file_path: Path to JSON or CSV file
            
        Returns:
            Evaluation results
        """
        dataset = EvaluationDataset()
        
        if file_path.endswith('.json'):
            dataset.load_from_json(file_path)
        elif file_path.endswith('.csv'):
            dataset.load_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        return self.evaluate_dataset(dataset)
    
    def evaluate_rag_query(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate RAG response and evaluate it
        
        Args:
            question: User query
            ground_truth: Expected answer (optional)
            user_id: User ID for tracking
            
        Returns:
            Dict with RAG response and evaluation scores
        """
        try:
            # Generate RAG response
            logger.info(f"Generating RAG response for evaluation: {question[:50]}...")
            rag_response = rag_chain.generate_response(
                query=question,
                user_id=user_id
            )
            
            # Extract data for evaluation
            answer = rag_response.get('text')
            if isinstance(answer, list):
                answer = ' '.join(answer)
            
            # Extract contexts from sources
            contexts = []
            for source in rag_response.get('sources', []):
                # Get content from source (may need to fetch from DB)
                if 'content' in source:
                    contexts.append(source['content'])
                elif 'title' in source:
                    contexts.append(source['title'])
            
            if not contexts:
                logger.warning("No contexts found in RAG response")
                contexts = ["No context retrieved"]
            
            # Evaluate
            scores = self.evaluate_single(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            )
            
            return {
                'rag_response': rag_response,
                'evaluation_scores': scores,
                'question': question,
                'answer': answer,
                'contexts': contexts,
                'ground_truth': ground_truth
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query evaluation: {e}", exc_info=True)
            return {'error': str(e)}


# Global evaluator instance
_evaluator: Optional[RAGEvaluator] = None


def get_evaluator() -> RAGEvaluator:
    """Get or create global evaluator instance"""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


# Convenience function
def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a RAG response
    
    Args:
        question: User query
        answer: Generated answer
        contexts: Retrieved contexts
        ground_truth: Expected answer (optional)
        
    Returns:
        Dict with evaluation scores
    """
    evaluator = get_evaluator()
    return evaluator.evaluate_single(question, answer, contexts, ground_truth)
