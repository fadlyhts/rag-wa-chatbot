"""Dashboard API endpoints for analytics and statistics"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

from app.database.session import get_db
from app.security.auth import get_current_active_admin
from app.models.admin import Admin
from app.models.document import Document
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.rag.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/dashboard/stats")
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get comprehensive dashboard statistics
    
    Returns:
    - Total documents (by status)
    - Total conversations
    - Message statistics
    - Token usage
    - System health
    """
    try:
        # Document statistics
        total_documents = db.query(func.count(Document.id)).filter(Document.is_active == True).scalar() or 0
        
        docs_by_status = db.query(
            Document.embedding_status,
            func.count(Document.id)
        ).filter(
            Document.is_active == True
        ).group_by(Document.embedding_status).all()
        
        status_counts = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0
        }
        for status, count in docs_by_status:
            status_counts[status] = count
        
        # Conversation statistics
        total_conversations = db.query(func.count(Conversation.id)).scalar() or 0
        active_conversations = db.query(func.count(Conversation.id)).filter(
            Conversation.is_active == True
        ).scalar() or 0
        
        # Message statistics
        total_messages = db.query(func.count(Message.id)).scalar() or 0
        
        # Messages today
        today = datetime.now().date()
        messages_today = db.query(func.count(Message.id)).filter(
            func.date(Message.created_at) == today
        ).scalar() or 0
        
        # Messages this week
        week_ago = datetime.now() - timedelta(days=7)
        messages_week = db.query(func.count(Message.id)).filter(
            Message.created_at >= week_ago
        ).scalar() or 0
        
        # Messages this month
        month_ago = datetime.now() - timedelta(days=30)
        messages_month = db.query(func.count(Message.id)).filter(
            Message.created_at >= month_ago
        ).scalar() or 0
        
        # Token usage
        total_tokens = db.query(func.sum(Message.llm_tokens)).filter(
            Message.llm_tokens.isnot(None)
        ).scalar() or 0
        
        # Average response time
        avg_response_time = db.query(func.avg(Message.response_time_ms)).filter(
            Message.response_time_ms.isnot(None),
            Message.role == 'assistant'
        ).scalar() or 0
        
        # User statistics
        total_users = db.query(func.count(User.id)).scalar() or 0
        active_users_week = db.query(func.count(User.id)).filter(
            User.last_active >= week_ago
        ).scalar() or 0
        
        # Vector database statistics
        try:
            collection_info = vector_store.get_collection_info()
            vector_stats = {
                "collection": collection_info["name"],
                "vectors_count": collection_info["vectors_count"],
                "status": collection_info["status"]
            }
        except Exception as e:
            logger.error(f"Error getting vector stats: {e}")
            vector_stats = {
                "collection": "unknown",
                "vectors_count": 0,
                "status": "error"
            }
        
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "documents": {
                "total": total_documents,
                "by_status": status_counts
            },
            "conversations": {
                "total": total_conversations,
                "active": active_conversations
            },
            "messages": {
                "total": total_messages,
                "today": messages_today,
                "this_week": messages_week,
                "this_month": messages_month
            },
            "tokens": {
                "total_used": int(total_tokens),
                "avg_per_message": int(total_tokens / total_messages) if total_messages > 0 else 0
            },
            "performance": {
                "avg_response_time_ms": round(float(avg_response_time), 2)
            },
            "users": {
                "total": total_users,
                "active_this_week": active_users_week
            },
            "vector_database": vector_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.get("/dashboard/activity")
async def get_recent_activity(
    limit: int = 20,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get recent activity
    
    Returns:
    - Recent document uploads
    - Recent messages
    - Recent conversations
    """
    try:
        # Recent documents
        recent_documents = db.query(Document).filter(
            Document.is_active == True
        ).order_by(desc(Document.upload_date)).limit(limit).all()
        
        doc_activity = [
            {
                "id": doc.id,
                "title": doc.title,
                "status": doc.embedding_status,
                "upload_date": doc.upload_date.isoformat(),
                "type": "document_upload"
            }
            for doc in recent_documents
        ]
        
        # Recent messages
        recent_messages = db.query(Message).order_by(
            desc(Message.created_at)
        ).limit(limit).all()
        
        message_activity = [
            {
                "id": msg.id,
                "role": msg.role,
                "content_preview": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                "created_at": msg.created_at.isoformat(),
                "user_id": msg.user_id,
                "type": "message"
            }
            for msg in recent_messages
        ]
        
        # Recent conversations
        recent_conversations = db.query(Conversation).order_by(
            desc(Conversation.started_at)
        ).limit(limit).all()
        
        conversation_activity = [
            {
                "id": conv.id,
                "user_id": conv.user_id,
                "message_count": conv.message_count,
                "started_at": conv.started_at.isoformat(),
                "is_active": conv.is_active,
                "type": "conversation"
            }
            for conv in recent_conversations
        ]
        
        return {
            "success": True,
            "recent_documents": doc_activity,
            "recent_messages": message_activity[:10],  # Limit to 10 for UI
            "recent_conversations": conversation_activity[:10]
        }
        
    except Exception as e:
        logger.error(f"Error getting activity: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get activity: {str(e)}")


@router.get("/dashboard/charts/messages")
async def get_message_trends(
    period: str = "week",  # week, month, year
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get message trends for charts
    
    Returns time-series data for message count
    """
    try:
        # Calculate date range based on period
        now = datetime.now()
        
        if period == "week":
            start_date = now - timedelta(days=7)
            group_format = "%Y-%m-%d"
        elif period == "month":
            start_date = now - timedelta(days=30)
            group_format = "%Y-%m-%d"
        elif period == "year":
            start_date = now - timedelta(days=365)
            group_format = "%Y-%m"
        else:
            raise HTTPException(status_code=400, detail="Invalid period. Use: week, month, or year")
        
        # Query messages grouped by date
        messages_by_date = db.query(
            func.date(Message.created_at).label('date'),
            func.count(Message.id).label('count')
        ).filter(
            Message.created_at >= start_date
        ).group_by(
            func.date(Message.created_at)
        ).all()
        
        # Format results
        chart_data = [
            {
                "date": date.isoformat() if date else None,
                "count": count
            }
            for date, count in messages_by_date
        ]
        
        return {
            "success": True,
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": now.isoformat(),
            "data": chart_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting message trends: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/dashboard/system-health")
async def get_system_health(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get system health status
    
    Checks:
    - Database connectivity
    - Qdrant connectivity
    - Redis connectivity (if enabled)
    """
    try:
        health_status = {
            "overall": "healthy",
            "components": {}
        }
        
        # Database check
        try:
            db.execute("SELECT 1")
            health_status["components"]["database"] = {
                "status": "healthy",
                "message": "Database connected"
            }
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "message": str(e)
            }
            health_status["overall"] = "degraded"
        
        # Qdrant check
        try:
            is_healthy = vector_store.health_check()
            if is_healthy:
                health_status["components"]["qdrant"] = {
                    "status": "healthy",
                    "message": "Qdrant connected"
                }
            else:
                health_status["components"]["qdrant"] = {
                    "status": "unhealthy",
                    "message": "Qdrant health check failed"
                }
                health_status["overall"] = "degraded"
        except Exception as e:
            health_status["components"]["qdrant"] = {
                "status": "unhealthy",
                "message": str(e)
            }
            health_status["overall"] = "degraded"
        
        # Redis check (optional)
        try:
            from app.rag.embeddings import embeddings_service
            if embeddings_service.cache_enabled:
                embeddings_service.redis_client.ping()
                health_status["components"]["redis"] = {
                    "status": "healthy",
                    "message": "Redis connected"
                }
            else:
                health_status["components"]["redis"] = {
                    "status": "disabled",
                    "message": "Redis caching disabled"
                }
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "message": str(e)
            }
            # Redis is optional, don't degrade overall status
        
        health_status["timestamp"] = datetime.utcnow().isoformat()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error checking system health: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
