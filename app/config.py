"""Application configuration"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
import os
from pathlib import Path

# Get the backend directory (parent of app directory)
BACKEND_DIR = Path(__file__).parent.parent
ENV_FILE = BACKEND_DIR / ".env"


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "WhatsApp RAG Chatbot"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str = "mysql+pymysql://root:password123@localhost:3306/whatsapp_chatbot"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "documents"
    
    # AI Model Configuration
    AI_PROVIDER: str = "gemini"  # "openai" or "gemini"
    
    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 500
    OPENAI_TEMPERATURE: float = 0.7
    
    # Gemini
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/text-embedding-004"
    GEMINI_MAX_TOKENS: int = 500
    GEMINI_TEMPERATURE: float = 0.7
    
    # RAG Settings
    RAG_CHUNK_SIZE: int = 512
    RAG_CHUNK_OVERLAP: int = 50
    RAG_TOP_K: int = 5
    RAG_MIN_SCORE: float = 0.7
    RAG_ENABLE_CACHE: bool = True
    
    # WAHA
    WAHA_API_URL: str = "http://localhost:3000"
    WAHA_API_KEY: str = ""
    WAHA_DASHBOARD_USERNAME: str = ""
    WAHA_DASHBOARD_PASSWORD: str = ""
    
    # Webhook
    WEBHOOK_SECRET: str = "change-me-in-production"
    
    # Rate Limiting
    RATE_LIMIT_MESSAGES_PER_MINUTE: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    model_config = ConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
