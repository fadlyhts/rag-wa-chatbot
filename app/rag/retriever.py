"""Semantic search retriever"""

from typing import List, Dict, Any, Optional
import logging
from app.rag.vector_store import vector_store
from app.rag.config import rag_config
from app.rag.factory import get_embeddings_service

logger = logging.getLogger(__name__)

# LangChain Document
from langchain_core.documents import Document


class Retriever:
    """Retriever for semantic search"""
    
    def __init__(self):
        self.vector_store = vector_store
        self.top_k = rag_config.top_k
        self.min_score = rag_config.min_score
    
    @property
    def embeddings(self):
        """Get embeddings service lazily"""
        return get_embeddings_service()
    
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


def qdrant_result_to_document(result: Dict[str, Any]) -> Document:
    """
    Convert a single Qdrant search result to a LangChain Document.
    """
    payload: Dict[str, Any] = result.get("payload", {})

    file_name: str = payload.get("file_name") or payload.get("title", "unknown_file")
    page_number: Optional[int] = payload.get("page_number") or payload.get("chunk_index")
    source_url: Optional[str] = payload.get("url") or payload.get("source_url")

    metadata: Dict[str, Any] = {
        "file_name":    file_name,
        "page_number":  page_number,
        "url":          source_url,
        "document_id":  payload.get("document_id"),
        "title":        payload.get("title", "Untitled"),
        "content_type": payload.get("content_type", "unknown"),
        "chunk_index":  payload.get("chunk_index"),
        "total_chunks": payload.get("total_chunks"),
        "doc_metadata": payload.get("doc_metadata", {}),
        "score":        result.get("score", 0.0),
        "qdrant_id":    result.get("id"),
    }

    return Document(
        page_content=payload.get("content", ""),
        metadata=metadata,
    )


class LCELRetriever:
    """
    Thin wrapper around the existing Qdrant VectorStore
    to be used as a Runnable in the LCEL chain.
    """

    def __init__(
        self,
        top_k: int = rag_config.top_k,
        min_score: float = rag_config.min_score,
    ):
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Retrieve and convert to LangChain Documents."""
        embeddings_svc = get_embeddings_service()

        query_vector = embeddings_svc.generate_embedding(query)
        
        sparse_query_vector = None
        if getattr(rag_config, 'RAG_HYBRID_SEARCH', False):
            try:
                from app.rag.embeddings_sparse import SparseEmbeddings
                sparse_embedder = SparseEmbeddings(model_name=rag_config.RAG_SPARSE_MODEL_NAME)
                sparse_query_vector = sparse_embedder.generate_sparse_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate sparse query embedding: {e}")
                
        raw_results = vector_store.search(
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.min_score,
            filter_conditions=filters,
            sparse_query_vector=sparse_query_vector
        )

        docs = [qdrant_result_to_document(r) for r in raw_results]
        logger.info(f"[LCELRetriever] Retrieved {len(docs)} documents")
        return docs

    async def aretrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        embeddings_svc = get_embeddings_service()
        query_vector = await embeddings_svc.generate_embedding_async(query)
        
        sparse_query_vector = None
        if getattr(rag_config, 'RAG_HYBRID_SEARCH', False):
            try:
                from app.rag.embeddings_sparse import SparseEmbeddings
                sparse_embedder = SparseEmbeddings(model_name=rag_config.RAG_SPARSE_MODEL_NAME)
                # fastembed doesn't have native async yet, so we call it sync (it's fast enough)
                sparse_query_vector = sparse_embedder.generate_sparse_embedding(query)
            except Exception as e:
                logger.error(f"Failed to generate sparse query embedding async: {e}")
                
        raw_results = vector_store.search(
            query_vector=query_vector,
            limit=self.top_k,
            score_threshold=self.min_score,
            filter_conditions=filters,
            sparse_query_vector=sparse_query_vector
        )
        docs = [qdrant_result_to_document(r) for r in raw_results]
        logger.info(f"[LCELRetriever] Retrieved {len(docs)} documents (async)")
        return docs
