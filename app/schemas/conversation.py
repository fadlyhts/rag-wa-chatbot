"""Conversation schemas"""

from pydantic import BaseModel
from typing import Optional, List


class ConversationListItem(BaseModel):
    """Conversation list item (summary)"""
    id: str
    phone: str
    name: Optional[str] = None
    last_message: Optional[str] = None
    last_message_time: Optional[str] = None
    message_count: int
    status: str  # "active" or "ended"
    created_at: str
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class ConversationsListResponse(BaseModel):
    """Paginated conversations list response"""
    success: bool = True
    conversations: List[ConversationListItem]
    total: int
    page: int
    limit: int


class MessageItem(BaseModel):
    """Message item for conversation detail"""
    id: str
    role: str
    content: str
    content_type: str
    media_url: Optional[str] = None
    created_at: str


class ConversationDetail(BaseModel):
    """Conversation detail with messages"""
    id: str
    phone: str
    name: Optional[str] = None
    status: str
    message_count: int
    messages: List[MessageItem]


class ConversationDetailResponse(BaseModel):
    """Conversation detail response wrapper"""
    success: bool = True
    conversation: ConversationDetail
