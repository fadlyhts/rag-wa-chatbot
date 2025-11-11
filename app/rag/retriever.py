"""Semantic search retriever"""

from typing import List, Dict, Any, Optional
import logging
from app.rag.embeddings import embeddings_service
from app.rag.vector_store import vector_store
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class Retriever:
    """Retriever for semantic search"""
    
    def __init__(self):
        self.embeddings = embeddings_service
        self.vector_store = vector_store
        self.top_k = rag_config.top_k
        self.min_score = rag_config.min_score
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: User query text
            top_k: Number of documents to retrieve (default: from config)
            min_score: Minimum relevance score (default: from config)
            filters: Optional metadata filters
            
        Returns:
            List of relevant documents with scores
        """
        try:
            # Generate query embedding
            logger.info(f"Generating embedding for query: {query[:50]}...")
            query_vector = self.embeddings.generate_embedding(query)
            
            # Search vector store
            top_k = top_k or self.top_k
            min_score = min_score or self.min_score
            
            logger.info(f"Searching for top-{top_k} documents (min_score: {min_score})")
            results = self.vector_store.search(
                query_vector=query_vector,
                limit=top_k,
                score_threshold=min_score,
                filter_conditions=filters
            )
            
            # Re-rank by score (already sorted by Qdrant, but explicit)
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Retrieved {len(results)} documents")
            for i, result in enumerate(results, 1):
                logger.debug(f"  {i}. Score: {result['score']:.3f} - {result['payload'].get('title', 'Untitled')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def retrieve_async(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query (async)
        
        Args:
            query: User query text
            top_k: Number of documents to retrieve
            min_score: Minimum relevance score
            filters: Optional metadata filters
            
        Returns:
            List of relevant documents with scores
        """
        try:
            # Generate query embedding asynchronously
            logger.info(f"Generating embedding (async) for query: {query[:50]}...")
            query_vector = await self.embeddings.generate_embedding_async(query)
            
            # Search vector store (Qdrant client doesn't have async API, but operation is fast)
            top_k = top_k or self.top_k
            min_score = min_score or self.min_score
            
            logger.info(f"Searching for top-{top_k} documents (min_score: {min_score})")
            results = self.vector_store.search(
                query_vector=query_vector,
                limit=top_k,
                score_threshold=min_score,
                filter_conditions=filters
            )
            
            # Re-rank by score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Retrieved {len(results)} documents (async)")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents (async): {e}")
            return []
    
    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            results: List of search results
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result['payload']
            title = payload.get('title', 'Document')
            content = payload.get('content', '')
            score = result['score']
            
            context_parts.append(
                f"[Source {i}] {title} (Relevance: {score:.2f})\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def get_sources_metadata(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source metadata from results
        
        Args:
            results: List of search results
            
        Returns:
            List of source metadata
        """
        sources = []
        for result in results:
            payload = result['payload']
            sources.append({
                'id': result['id'],
                'title': payload.get('title', 'Untitled'),
                'content_type': payload.get('content_type', 'unknown'),
                'score': result['score'],
                'source_url': payload.get('source_url')
            })
        
        return sources


# Global retriever instance
retriever = Retriever()
