"""Qdrant vector database client"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
import logging
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store using Qdrant"""
    
    def __init__(self):
        self.client = None
        self._initialized = False
        # Lazy initialization - connect on first use
        try:
            self.client = self._init_client()
            self._ensure_collection()
            self._initialized = True
        except Exception as e:
            logger.warning(f"Qdrant not available at startup: {e}")
            logger.warning("Vector store will be initialized on first use")
    
    @property
    def collection_name(self) -> str:
        """Get collection name dynamically based on current config"""
        return rag_config.qdrant_collection
    
    @property
    def vector_size(self) -> int:
        """Get vector size dynamically based on current config"""
        return rag_config.vector_size
    
    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client"""
        try:
            if rag_config.qdrant_api_key:
                client = QdrantClient(
                    url=rag_config.qdrant_url,
                    api_key=rag_config.qdrant_api_key
                )
            else:
                client = QdrantClient(url=rag_config.qdrant_url)
            
            logger.info(f"Connected to Qdrant at {rag_config.qdrant_url}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure client is initialized before use"""
        if not self._initialized and self.client is None:
            try:
                self.client = self._init_client()
                self._ensure_collection()
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize Qdrant: {e}")
                raise Exception("Qdrant vector store is not available")
    
    def _ensure_collection(self):
        """Ensure collection exists, create if not"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection created: {self.collection_name}")
            else:
                logger.info(f"Collection exists: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy"""
        try:
            if self.client is None:
                self._ensure_initialized()
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def insert_documents(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ) -> bool:
        """
        Insert documents into vector store
        
        Args:
            ids: List of document IDs
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
            
        Returns:
            True if successful
        """
        try:
            self._ensure_initialized()
            
            points = [
                PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload
                )
                for doc_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Inserted {len(points)} documents into {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = None,
        score_threshold: float = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results (default: from config)
            score_threshold: Minimum similarity score (default: from config)
            filter_conditions: Optional metadata filters
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            self._ensure_initialized()
            
            limit = limit or rag_config.top_k
            score_threshold = score_threshold or rag_config.min_score
            
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            # Format results
            formatted_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
            
            logger.info(f"Found {len(formatted_results)} documents (threshold: {score_threshold})")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from vector store
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self._ensure_initialized()
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
            
            logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            self._ensure_initialized()
            
            collection = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection.vectors_count,
                "points_count": collection.points_count,
                "status": collection.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def delete_collection(self) -> bool:
        """Delete the entire collection (use with caution!)"""
        try:
            self._ensure_initialized()
            
            self.client.delete_collection(self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise


# Global vector store instance
vector_store = VectorStore()
