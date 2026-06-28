"""Conversations management API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.database.session import get_db
from app.security.auth import get_current_active_admin
from app.models.admin import Admin
from app.schemas.conversation import (
    ConversationsListResponse,
    ConversationDetailResponse,
)
from app.services.conversation_service import conversation_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/conversations/export")
async def export_conversations(
    format: str = Query("csv", regex="^(csv|json)$"),
    search: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Export conversations data as CSV or JSON file
    """
    try:
        data = conversation_service.export_conversations(
            db=db,
            format=format,
            search=search,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )

        if format == "json":
            return Response(
                content=data,
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=conversations_export.json"},
            )
        else:
            return Response(
                content=data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=conversations_export.csv"},
            )

    except Exception as e:
        logger.error(f"Error exporting conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export conversations: {str(e)}")


@router.get("/conversations", response_model=ConversationsListResponse)
async def list_conversations(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    List conversations with pagination, search, date range, and status filter

    - **page**: Page number (1-indexed)
    - **limit**: Items per page (1-100)
    - **search**: Search in user phone_number and whatsapp_name
    - **status**: Filter by status (active/ended/all)
    - **start_date**: Filter conversations started on or after this date (ISO format)
    - **end_date**: Filter conversations started on or before this date (ISO format)
    """
    try:
        result = conversation_service.list_conversations(
            db=db,
            page=page,
            limit=limit,
            search=search,
            status=status,
            start_date=start_date,
            end_date=end_date,
        )
        return result
    except Exception as e:
        logger.error(f"Error listing conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation_detail(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Get conversation detail with messages
    """
    try:
        result = conversation_service.get_conversation_detail(db, conversation_id)

        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get conversation: {str(e)}")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Delete conversation and all associated messages
    """
    try:
        result = conversation_service.delete_conversation(db, conversation_id)

        if not result:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {conversation_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete conversation: {str(e)}")
