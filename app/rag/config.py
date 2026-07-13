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
    
    # Vertex AI Settings
    use_vertex_ai: bool = settings.USE_VERTEX_AI
    vertex_project_id: str = settings.VERTEX_PROJECT_ID
    vertex_location: str = settings.VERTEX_LOCATION
    
    # Qdrant Settings
    qdrant_url: str = settings.QDRANT_URL
    qdrant_api_key: str = settings.QDRANT_API_KEY
    
    # Dynamic collection name based on AI provider to avoid dimension conflicts
    # This allows switching between providers without deleting data
    @property
    def qdrant_collection(self) -> str:
        """Get collection name based on AI provider"""
        base_name = settings.QDRANT_COLLECTION
        # Append provider suffix to collection name
        return f"{base_name}_{self.ai_provider}"
    
    # Vector size depends on embedding model:
    # - OpenAI text-embedding-3-small: 1536
    # - Gemini text-embedding-004: 768
    vector_size: int = settings.VECTOR_SIZE
    
    # RAG Settings
    chunk_size: int = settings.RAG_CHUNK_SIZE
    chunk_overlap: int = settings.RAG_CHUNK_OVERLAP
    top_k: int = settings.RAG_TOP_K
    min_score: float = settings.RAG_MIN_SCORE
    enable_cache: bool = settings.RAG_ENABLE_CACHE
    
    # Hybrid Search Settings
    RAG_HYBRID_SEARCH: bool = settings.RAG_HYBRID_SEARCH
    RAG_SPARSE_MODEL_NAME: str = settings.RAG_SPARSE_MODEL_NAME
    
    # Docling Parser
    use_docling: bool = settings.USE_DOCLING
    
    # Redis Cache
    redis_url: str = settings.REDIS_URL
    cache_ttl: int = 3600  # 1 hour


# Global RAG config instance
rag_config = RAGConfig()
