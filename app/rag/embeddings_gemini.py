"""Google Gemini embeddings service"""

from typing import List, Optional
import google.generativeai as genai
import redis
import json
import hashlib
import logging
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class GeminiEmbeddingsService:
    """Service for generating embeddings using Google Gemini"""
    
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=rag_config.google_api_key)
        self.model_name = rag_config.gemini_embedding_model
        
        # Redis cache for embeddings
        self.cache_enabled = rag_config.enable_cache
        if self.cache_enabled:
            try:
                self.redis_client = redis.from_url(
                    rag_config.redis_url,
                    decode_responses=False  # Store bytes for embeddings
                )
                logger.info("Redis cache enabled for embeddings")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis cache: {e}")
                self.cache_enabled = False
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"emb:gemini:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        if not self.cache_enabled:
            return None
        
        try:
            key = self._get_cache_key(text)
            cached = self.redis_client.get(key)
            if cached:
                logger.debug("Cache hit for embedding")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to cache"""
        if not self.cache_enabled:
            return
        
        try:
            key = self._get_cache_key(text)
            self.redis_client.setex(
                key,
                rag_config.cache_ttl,
                json.dumps(embedding)
            )
            logger.debug("Cached embedding")
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Gemini
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # Check cache first
        cached = self._get_from_cache(text)
        if cached:
            return cached
        
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embedding = result['embedding']
            
            # Cache the result
            self._save_to_cache(text, embedding)
            
            logger.debug(f"Generated Gemini embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            raise
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Generate embedding asynchronously (Gemini SDK is sync, so we wrap it)
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        # For now, Gemini SDK doesn't have native async support
        # We'll use the sync version
        return self.generate_embedding(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = []
            
            # Gemini supports batch embedding
            for text in texts:
                # Check cache for each text
                cached = self._get_from_cache(text)
                if cached:
                    embeddings.append(cached)
                else:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type="retrieval_document"
                    )
                    embedding = result['embedding']
                    embeddings.append(embedding)
                    self._save_to_cache(text, embedding)
            
            logger.info(f"Generated {len(embeddings)} Gemini embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch Gemini embeddings: {e}")
            raise
    
    async def generate_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (async wrapper)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.generate_embeddings_batch(texts)


# Global Gemini embeddings service instance
gemini_embeddings_service = GeminiEmbeddingsService()
