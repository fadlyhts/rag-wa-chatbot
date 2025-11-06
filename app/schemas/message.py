"""Message schemas"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class MessageBase(BaseModel):
    """Base message schema"""
    content: str
    content_type: str = "text"


class MessageCreate(MessageBase):
    """Message creation schema"""
    user_id: int
    conversation_id: int
    role: str


class MessageResponse(MessageBase):
    """Message response schema"""
    id: int
    conversation_id: int
    user_id: int
    role: str
    created_at: datetime
    processed_at: Optional[datetime] = None
    response_time_ms: Optional[int] = None
    llm_tokens: Optional[int] = None
    rag_context: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    """Message list response schema"""
    total: int
    limit: int
    offset: int
    data: list[MessageResponse]
