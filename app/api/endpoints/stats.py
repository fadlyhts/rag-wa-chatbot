"""Statistics endpoint"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database.session import get_db
from app.models.message import Message
from app.models.conversation import Conversation
from app.schemas.response import StatsResponse
from datetime import datetime, timedelta
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/stats", response_model=StatsResponse)
async def get_stats(db: Session = Depends(get_db)):
    """
    Get system statistics
    - Messages per hour
    - Average response time
    - Active conversations
    - Queue depth (placeholder)
    - LLM tokens used
    - Error rate (placeholder)
    """
    
    # Calculate messages per hour (last hour)
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    messages_last_hour = db.query(func.count(Message.id)).filter(
        Message.created_at >= one_hour_ago
    ).scalar() or 0
    
    # Calculate average response time
    avg_response_time = db.query(func.avg(Message.response_time_ms)).filter(
        Message.response_time_ms.isnot(None),
        Message.created_at >= one_hour_ago
    ).scalar() or 0
    
    # Count active conversations
    active_conversations = db.query(func.count(Conversation.id)).filter(
        Conversation.is_active == True
    ).scalar() or 0
    
    # Sum LLM tokens (last 24 hours)
    twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
    llm_tokens_used = db.query(func.sum(Message.llm_tokens)).filter(
        Message.llm_tokens.isnot(None),
        Message.created_at >= twenty_four_hours_ago
    ).scalar() or 0
    
    # TODO: Get actual queue depth from RQ
    queue_depth = 0
    
    # TODO: Calculate actual error rate
    error_rate_percent = 0.0
    
    return StatsResponse(
        messages_per_hour=float(messages_last_hour),
        avg_response_time_ms=float(avg_response_time),
        active_conversations=active_conversations,
        queue_depth=queue_depth,
        llm_tokens_used=int(llm_tokens_used),
        error_rate_percent=error_rate_percent
    )
