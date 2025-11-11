"""OpenAI embeddings service"""

from typing import List, Optional
from openai import OpenAI, AsyncOpenAI
import redis
import json
import hashlib
import logging
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Service for generating embeddings using OpenAI"""
    
    def __init__(self):
        self.client = OpenAI(api_key=rag_config.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=rag_config.openai_api_key)
        self.model = rag_config.embedding_model
        
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
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        if not self.cache_enabled:
            return None
        
        try:
            key = self._get_cache_key(text)
            cached = self.redis_client.get(key)
            if cached:
                logger.debug(f"Cache hit for embedding")
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
            logger.debug(f"Cached embedding")
        except Exception as e:
            logger.warning(f"Cache save error: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
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
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            self._save_to_cache(text, embedding)
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """
        Generate embedding asynchronously
        
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
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Cache the result
            self._save_to_cache(text, embedding)
            
            logger.debug(f"Generated embedding (async) for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding (async): {e}")
            raise
    
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
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            
            # Cache individual embeddings
            for text, embedding in zip(texts, embeddings):
                self._save_to_cache(text, embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    async def generate_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (async)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            
            # Cache individual embeddings
            for text, embedding in zip(texts, embeddings):
                self._save_to_cache(text, embedding)
            
            logger.info(f"Generated {len(embeddings)} embeddings in batch (async)")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings (async): {e}")
            raise


# Global embeddings service instance
embeddings_service = EmbeddingsService()
