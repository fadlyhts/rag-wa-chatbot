"""Webhook schemas"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any


class WebhookData(BaseModel):
    """WAHA webhook data schema"""
    messageId: Optional[str] = None
    timestamp: Optional[int] = None
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None
    text: Optional[str] = None
    status: Optional[str] = None


class WebhookPayload(BaseModel):
    """WAHA webhook payload schema"""
    event: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    
    @validator('event')
    def validate_event(cls, v):
        allowed_events = ['message.incoming', 'message.status', 'ready', 'session.status']
        if v not in allowed_events:
            raise ValueError(f'Event must be one of {allowed_events}')
        return v


class WebhookResponse(BaseModel):
    """Webhook response schema"""
    status: str
    request_id: Optional[str] = None
    job_id: Optional[str] = None
    message_id: Optional[str] = None
