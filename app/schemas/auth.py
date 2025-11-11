"""Authentication schemas"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class LoginRequest(BaseModel):
    """Login request schema"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    token_type: str = "bearer"
    role: str
    user_info: dict


class AdminResponse(BaseModel):
    """Admin user response schema"""
    id: int
    username: str
    email: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class AdminCreate(BaseModel):
    """Create admin schema"""
    username: str
    email: Optional[str] = None
    password: str
    role: str = "admin"


class AdminUpdate(BaseModel):
    """Update admin schema"""
    email: Optional[str] = None
    password: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
