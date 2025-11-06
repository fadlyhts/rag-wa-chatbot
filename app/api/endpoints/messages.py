"""Messages endpoint"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.message import Message
from app.schemas.message import MessageResponse, MessageListResponse
from typing import Optional
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/messages", response_model=MessageListResponse)
async def get_messages(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    conversation_id: Optional[int] = Query(None, description="Filter by conversation ID"),
    role: Optional[str] = Query(None, description="Filter by role (user/assistant)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of messages"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Get messages with filtering and pagination
    """
    query = db.query(Message)
    
    # Apply filters
    if user_id:
        query = query.filter(Message.user_id == user_id)
    if conversation_id:
        query = query.filter(Message.conversation_id == conversation_id)
    if role:
        query = query.filter(Message.role == role)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    messages = query.order_by(Message.created_at.desc()).offset(offset).limit(limit).all()
    
    return MessageListResponse(
        total=total,
        limit=limit,
        offset=offset,
        data=[MessageResponse.from_orm(msg) for msg in messages]
    )
