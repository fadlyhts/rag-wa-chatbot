"""RAG system configuration"""

from app.config import settings
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    
    # AI Provider Selection
    ai_provider: str = settings.AI_PROVIDER  # "openai" or "gemini"
    
    # OpenAI Settings
    openai_api_key: str = settings.OPENAI_API_KEY
    llm_model: str = settings.OPENAI_MODEL
    embedding_model: str = settings.OPENAI_EMBEDDING_MODEL
    max_tokens: int = settings.OPENAI_MAX_TOKENS
    temperature: float = settings.OPENAI_TEMPERATURE
    
    # Gemini Settings
    google_api_key: str = settings.GOOGLE_API_KEY
    gemini_model: str = settings.GEMINI_MODEL
    gemini_embedding_model: str = settings.GEMINI_EMBEDDING_MODEL
    gemini_max_tokens: int = settings.GEMINI_MAX_TOKENS
    gemini_temperature: float = settings.GEMINI_TEMPERATURE
    
    # Qdrant Settings
    qdrant_url: str = settings.QDRANT_URL
    qdrant_api_key: str = settings.QDRANT_API_KEY
    qdrant_collection: str = settings.QDRANT_COLLECTION
    # Vector size depends on embedding model:
    # - OpenAI text-embedding-3-small: 1536
    # - Gemini text-embedding-004: 768
    vector_size: int = 768 if settings.AI_PROVIDER == "gemini" else 1536
    
    # RAG Settings
    chunk_size: int = settings.RAG_CHUNK_SIZE
    chunk_overlap: int = settings.RAG_CHUNK_OVERLAP
    top_k: int = settings.RAG_TOP_K
    min_score: float = settings.RAG_MIN_SCORE
    enable_cache: bool = settings.RAG_ENABLE_CACHE
    
    # Redis Cache
    redis_url: str = settings.REDIS_URL
    cache_ttl: int = 3600  # 1 hour


# Global RAG config instance
rag_config = RAGConfig()
