"""Factory for AI service providers"""

import logging
from typing import Any
from app.rag.config import rag_config

logger = logging.getLogger(__name__)


class AIServiceFactory:
    """Factory to get the correct AI service based on configuration"""
    
    _embeddings_service = None
    _generator_service = None
    _provider_logged = False
    
    @classmethod
    def get_embeddings_service(cls) -> Any:
        """Get embeddings service based on AI_PROVIDER"""
        if cls._embeddings_service is None:
            provider = rag_config.ai_provider.lower()
            
            if not cls._provider_logged:
                logger.info(f"🚀 AI Provider: {provider.upper()}")
                cls._provider_logged = True
            
            if provider == "gemini":
                logger.info("Loading Gemini embeddings service...")
                from app.rag.embeddings_gemini import gemini_embeddings_service
                cls._embeddings_service = gemini_embeddings_service
                logger.info("✅ Gemini embeddings service loaded")
            else:
                logger.info("Loading OpenAI embeddings service...")
                from app.rag.embeddings import embeddings_service
                cls._embeddings_service = embeddings_service
                logger.info("✅ OpenAI embeddings service loaded")
        
        return cls._embeddings_service
    
    @classmethod
    def get_generator_service(cls) -> Any:
        """Get generator service based on AI_PROVIDER"""
        if cls._generator_service is None:
            provider = rag_config.ai_provider.lower()
            
            if not cls._provider_logged:
                logger.info(f"🚀 AI Provider: {provider.upper()}")
                cls._provider_logged = True
            
            if provider == "gemini":
                logger.info("Loading Gemini generator service...")
                from app.rag.generator_gemini import gemini_generator
                cls._generator_service = gemini_generator
                logger.info("✅ Gemini generator service loaded")
            else:
                logger.info("Loading OpenAI generator service...")
                from app.rag.generator import generator
                cls._generator_service = generator
                logger.info("✅ OpenAI generator service loaded")
        
        return cls._generator_service


# Convenience functions
def get_embeddings_service():
    """Get the configured embeddings service"""
    return AIServiceFactory.get_embeddings_service()


def get_generator_service():
    """Get the configured generator service"""
    return AIServiceFactory.get_generator_service()
