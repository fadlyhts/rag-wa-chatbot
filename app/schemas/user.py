"""User schemas"""

from pydantic import BaseModel
from typing import Optional, List


class UserListItem(BaseModel):
    """User list item (summary)"""
    id: int
    phone: str
    name: Optional[str] = None
    messages_count: int
    last_active: str
    status: str  # "active" or "blocked"
    division_id: Optional[int] = None

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    """Create user request body"""
    phone_number: str
    whatsapp_name: Optional[str] = None
    division_id: Optional[int] = None


class UserListResponse(BaseModel):
    """Paginated user list response"""
    success: bool = True
    users: List[UserListItem]
    total: int
    page: int
    limit: int


class UserConversationItem(BaseModel):
    """User conversation item for detail view"""
    id: str
    last_message: str
    last_message_time: str
    message_count: int


class UserDetail(BaseModel):
    """User detail with conversations"""
    id: int
    phone: str
    name: Optional[str] = None
    profile_pic_url: Optional[str] = None
    language: str
    created_at: str
    last_active: str
    status: str
    notes: Optional[str] = None
    division_id: Optional[int] = None
    conversations: List[UserConversationItem]
    total_tokens_used: int = 0
    avg_response_time_ms: int = 0


class UserDetailResponse(BaseModel):
    """User detail response wrapper"""
    success: bool = True
    user: UserDetail


class BlockUserResponse(BaseModel):
    """Block/unblock user response"""
    success: bool = True
    message: str


class UpdateNotesRequest(BaseModel):
    """Update notes request body"""
    notes: str
