"""Conversation management service"""

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, or_, desc
from datetime import datetime
import csv
import io
import json
import logging

from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.conversation import (
    ConversationsListResponse,
    ConversationListItem,
    ConversationDetailResponse,
    ConversationDetail,
    MessageItem,
)

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for conversation management operations"""

    def list_conversations(
        self,
        db: Session,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> ConversationsListResponse:
        """
        List conversations with pagination, search, date range, and status filter

        Args:
            db: Database session
            page: Page number (1-indexed)
            limit: Items per page
            search: Search query for user phone_number or whatsapp_name
            status: Filter by status ("active" or "ended")
            start_date: Filter conversations started on or after this date (ISO format)
            end_date: Filter conversations started on or before this date (ISO format)

        Returns:
            Paginated conversations list
        """
        # Subquery for last message per conversation
        last_message_subq = (
            db.query(
                Message.conversation_id,
                func.max(Message.id).label("last_message_id"),
            )
            .group_by(Message.conversation_id)
            .subquery()
        )

        # Get last message content
        last_msg_alias = (
            db.query(
                Message.conversation_id,
                Message.content.label("last_content"),
                Message.created_at.label("last_created_at"),
            )
            .join(last_message_subq, Message.id == last_message_subq.c.last_message_id)
            .subquery()
        )

        query = db.query(
            Conversation,
            User.phone_number,
            User.whatsapp_name,
            last_msg_alias.c.last_content,
            last_msg_alias.c.last_created_at,
        ).join(User, Conversation.user_id == User.id).outerjoin(
            last_msg_alias, Conversation.id == last_msg_alias.c.conversation_id
        )

        # Apply search filter on user phone/name
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    User.phone_number.ilike(search_pattern),
                    User.whatsapp_name.ilike(search_pattern),
                )
            )

        # Apply status filter
        if status and status != "all":
            if status == "active":
                query = query.filter(Conversation.is_active == True)
            elif status == "ended":
                query = query.filter(Conversation.is_active == False)

        # Apply date range filter
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                query = query.filter(Conversation.started_at >= start_dt)
            except ValueError:
                pass

        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
                query = query.filter(Conversation.started_at <= end_dt)
            except ValueError:
                pass

        # Count total
        total = query.count()

        # Calculate pagination
        offset = (page - 1) * limit

        # Get page of results
        results = query.order_by(desc(Conversation.started_at)).offset(offset).limit(limit).all()

        # Convert to response items
        items = []
        for conv, phone_number, whatsapp_name, last_content, last_created_at in results:
            item = ConversationListItem(
                id=str(conv.id),
                phone=phone_number,
                name=whatsapp_name,
                last_message=last_content,
                last_message_time=last_created_at.isoformat() if last_created_at else None,
                message_count=conv.message_count,
                status="active" if conv.is_active else "ended",
                created_at=conv.started_at.isoformat() if conv.started_at else "",
                updated_at=conv.ended_at.isoformat() if conv.ended_at else None,
            )
            items.append(item)

        return ConversationsListResponse(
            conversations=items,
            total=total,
            page=page,
            limit=limit,
        )

    def get_conversation_detail(
        self, db: Session, conversation_id: int
    ) -> Optional[ConversationDetailResponse]:
        """
        Get conversation detail with messages ordered ASC

        Args:
            db: Database session
            conversation_id: Conversation ID

        Returns:
            Conversation detail response or None
        """
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )

        if not conv:
            return None

        # Get user info
        user = db.query(User).filter(User.id == conv.user_id).first()

        # Get messages ordered by created_at ASC
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .all()
        )

        message_items = [
            MessageItem(
                id=str(msg.id),
                role=msg.role,
                content=msg.content,
                content_type=msg.content_type or "text",
                media_url=msg.media_url,
                created_at=msg.created_at.isoformat() if msg.created_at else "",
            )
            for msg in messages
        ]

        detail = ConversationDetail(
            id=str(conv.id),
            phone=user.phone_number if user else "",
            name=user.whatsapp_name if user else None,
            status="active" if conv.is_active else "ended",
            message_count=conv.message_count,
            messages=message_items,
        )

        return ConversationDetailResponse(conversation=detail)

    def delete_conversation(self, db: Session, conversation_id: int) -> Optional[Dict[str, Any]]:
        """
        Delete conversation and cascade delete messages

        Args:
            db: Database session
            conversation_id: Conversation ID

        Returns:
            Success dict or None if not found
        """
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id)
            .first()
        )

        if not conv:
            return None

        db.delete(conv)
        db.commit()

        logger.info(f"Deleted conversation {conversation_id}")
        return {"success": True, "message": "Conversation deleted successfully"}

    def export_conversations(
        self,
        db: Session,
        format: str = "csv",
        search: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """
        Export conversations as CSV or JSON

        Args:
            db: Database session
            format: Export format ("csv" or "json")
            search: Search filter
            status: Status filter
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Exported data as string
        """
        query = db.query(
            Conversation,
            User.phone_number,
            User.whatsapp_name,
        ).join(User, Conversation.user_id == User.id)

        # Apply filters
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    User.phone_number.ilike(search_pattern),
                    User.whatsapp_name.ilike(search_pattern),
                )
            )

        if status and status != "all":
            if status == "active":
                query = query.filter(Conversation.is_active == True)
            elif status == "ended":
                query = query.filter(Conversation.is_active == False)

        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date)
                query = query.filter(Conversation.started_at >= start_dt)
            except ValueError:
                pass

        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date)
                query = query.filter(Conversation.started_at <= end_dt)
            except ValueError:
                pass

        results = query.order_by(desc(Conversation.started_at)).all()

        if format == "json":
            data = []
            for conv, phone_number, whatsapp_name in results:
                data.append({
                    "id": conv.id,
                    "phone_number": phone_number,
                    "whatsapp_name": whatsapp_name,
                    "started_at": conv.started_at.isoformat() if conv.started_at else None,
                    "ended_at": conv.ended_at.isoformat() if conv.ended_at else None,
                    "message_count": conv.message_count,
                    "is_active": conv.is_active,
                })
            return json.dumps(data, indent=2)
        else:
            # CSV format
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "id", "phone_number", "whatsapp_name",
                "started_at", "ended_at", "message_count", "is_active"
            ])
            for conv, phone_number, whatsapp_name in results:
                writer.writerow([
                    conv.id,
                    phone_number,
                    whatsapp_name or "",
                    conv.started_at.isoformat() if conv.started_at else "",
                    conv.ended_at.isoformat() if conv.ended_at else "",
                    conv.message_count,
                    conv.is_active,
                ])
            return output.getvalue()


# Global conversation service instance
conversation_service = ConversationService()
