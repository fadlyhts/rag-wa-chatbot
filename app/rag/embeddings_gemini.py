"""Google Gemini embeddings service (using google-genai SDK)"""

from typing import List, Optional
from google import genai
from google.genai import types
import redis
import json
import hashlib
import logging
import time
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class GeminiEmbeddingsService:
    """Service for generating embeddings using Google Gemini (new google-genai SDK)"""
    
    def __init__(self):
        # Initialize the new genai client
        self.client = genai.Client(api_key=rag_config.google_api_key)
        
        # Model name (without "models/" prefix for new SDK)
        model_name = rag_config.gemini_embedding_model
        if model_name.startswith("models/"):
            model_name = model_name.replace("models/", "")
        self.model_name = model_name
        
        logger.info(f"Initializing Gemini embeddings with model: {self.model_name}")
        
        # Redis cache for embeddings
        self.cache_enabled = rag_config.enable_cache
        if self.cache_enabled:
            try:
                self.redis_client = redis.from_url(
                    rag_config.redis_url,
                    decode_responses=False
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
        cached = self._get_from_cache(text)
        if cached:
            return cached
        
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            embedding = response.embeddings[0].values
            
            self._save_to_cache(text, embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating Gemini embedding: {e}")
            raise
    
    async def generate_embedding_async(self, text: str) -> List[float]:
        """Generate embedding asynchronously"""
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
            
            # Check cache for all texts
            for i, text in enumerate(texts):
                cached = self._get_from_cache(text)
                if cached:
                    embeddings.append(cached)
                else:
                    embeddings.append(None)
                    texts_to_generate.append(text)
                    indices_to_generate.append(i)
            
            if texts_to_generate:
                logger.info(f"Generating embeddings for {len(texts_to_generate)} texts (cached: {len(texts) - len(texts_to_generate)})")
                
                # Process in smaller batches with retry for rate limits
                BATCH_SIZE = 20
                for batch_start in range(0, len(texts_to_generate), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(texts_to_generate))
                    batch_texts = texts_to_generate[batch_start:batch_end]
                    
                    # Generate embeddings one by one within each batch
                    # Gemini embed_content treats list input as a single multi-part document,
                    # so we must call it per-text to get individual embeddings
                    batch_embeddings = []
                    for text in batch_texts:
                        # Retry with exponential backoff for rate limits
                        max_retries = 5
                        for attempt in range(max_retries):
                            try:
                                response = self.client.models.embed_content(
                                    model=self.model_name,
                                    contents=text,
                                )
                                batch_embeddings.append(response.embeddings[0].values)
                                break
                            except Exception as retry_err:
                                if "429" in str(retry_err) or "Resource exhausted" in str(retry_err):
                                    wait_time = (2 ** attempt) * 2
                                    logger.warning(f"Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                                    time.sleep(wait_time)
                                    if attempt == max_retries - 1:
                                        logger.error(f"Failed to embed text after {max_retries} retries")
                                        batch_embeddings.append(None)
                                else:
                                    raise
                    
                    for i, embedding in enumerate(batch_embeddings):
                        original_index = indices_to_generate[batch_start + i]
                        embeddings[original_index] = embedding
                        if embedding is not None:
                            self._save_to_cache(batch_texts[i], embedding)
                    
                    logger.info(f"Batch {batch_start//BATCH_SIZE + 1}: Generated {len(batch_embeddings)} embeddings")
                    
                    # Small delay between batches to avoid rate limits
                    if batch_end < len(texts_to_generate):
                        time.sleep(1)
            
            logger.info(f"Total: Generated {len(embeddings)} embeddings (new: {len(texts_to_generate)}, cached: {len(texts) - len(texts_to_generate)})")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch Gemini embeddings: {e}")
            raise
    
    async def generate_embeddings_batch_async(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch (async wrapper)"""
        return self.generate_embeddings_batch(texts)


# Global Gemini embeddings service instance
gemini_embeddings_service = GeminiEmbeddingsService()
