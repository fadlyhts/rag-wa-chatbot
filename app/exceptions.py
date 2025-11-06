"""Custom exception classes"""


class ChatbotException(Exception):
    """Base exception for chatbot"""
    pass


class WebhookException(ChatbotException):
    """Webhook-related errors"""
    pass


class ValidationException(ChatbotException):
    """Validation errors"""
    pass


class RateLimitException(ChatbotException):
    """Rate limit exceeded"""
    pass


class RAGException(ChatbotException):
    """RAG pipeline errors"""
    pass


class DatabaseException(ChatbotException):
    """Database operation errors"""
    pass


class ExternalAPIException(ChatbotException):
    """External API errors (WAHA, OpenAI)"""
    pass


class JobProcessingException(ChatbotException):
    """Job processing errors"""
    pass
