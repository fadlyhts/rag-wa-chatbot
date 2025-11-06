"""Generic response schemas"""

from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: str
    request_id: Optional[str] = None
    timestamp: datetime = datetime.utcnow()
    retry_after: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str
    timestamp: datetime = datetime.utcnow()
    dependencies: dict


class StatsResponse(BaseModel):
    """Statistics response schema"""
    messages_per_hour: float
    avg_response_time_ms: float
    active_conversations: int
    queue_depth: int
    llm_tokens_used: int
    error_rate_percent: float
    timestamp: datetime = datetime.utcnow()
