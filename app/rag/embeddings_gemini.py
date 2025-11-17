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
        
        # Ensure embedding model has "models/" prefix
        model_name = rag_config.gemini_embedding_model
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        self.model_name = model_name
        
        logger.info(f"Initializing Gemini embeddings with model: {self.model_name}")
        
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
            texts_to_generate = []
            indices_to_generate = []
            
            # First, check cache for all texts
            for i, text in enumerate(texts):
                cached = self._get_from_cache(text)
                if cached:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_generate.append(text)
                    indices_to_generate.append(i)
            
            # If we have texts to generate, use TRUE batch API (one call)
            if texts_to_generate:
                logger.info(f"Generating embeddings for {len(texts_to_generate)} texts (cached: {len(texts) - len(texts_to_generate)})")
                
                # Gemini batch API - process in chunks of 100 max
                BATCH_SIZE = 100
                for batch_start in range(0, len(texts_to_generate), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(texts_to_generate))
                    batch_texts = texts_to_generate[batch_start:batch_end]
                    
                    # Single API call for the batch!
                    result = genai.embed_content(
                        model=self.model_name,
                        content=batch_texts,  # Pass all texts at once
                        task_type="retrieval_document"
                    )
                    
                    # Extract embeddings and cache them
                    batch_embeddings = result['embedding'] if isinstance(result['embedding'][0], list) else [result['embedding']]
                    
                    for i, embedding in enumerate(batch_embeddings):
                        original_index = indices_to_generate[batch_start + i]
                        embeddings[original_index] = embedding
                        self._save_to_cache(batch_texts[i], embedding)
                    
                    logger.info(f"Batch {batch_start//BATCH_SIZE + 1}: Generated {len(batch_embeddings)} embeddings")
            
            logger.info(f"Total: Generated {len(embeddings)} embeddings (new: {len(texts_to_generate)}, cached: {len(texts) - len(texts_to_generate)})")
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
