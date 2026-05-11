"""User management service"""

from typing import Optional, List, Dict, Any
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
from app.schemas.user import (
    UserListResponse,
    UserListItem,
    UserDetailResponse,
    UserDetail,
    UserConversationItem,
    BlockUserResponse,
)

logger = logging.getLogger(__name__)


class UserService:
    """Service for user management operations"""

    def list_users(
        self,
        db: Session,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = None,
    ) -> UserListResponse:
        """
        List users with pagination, search, and status filter

        Args:
            db: Database session
            page: Page number (1-indexed)
            limit: Items per page
            search: Search query for phone_number or whatsapp_name
            status: Filter by status ("active" or "blocked")

        Returns:
            Paginated user list
        """
        # Subquery for messages count per user
        messages_count_subq = (
            db.query(
                Message.user_id,
                func.count(Message.id).label("messages_count"),
            )
            .group_by(Message.user_id)
            .subquery()
        )

        query = db.query(
            User,
            func.coalesce(messages_count_subq.c.messages_count, 0).label("messages_count"),
        ).outerjoin(messages_count_subq, User.id == messages_count_subq.c.user_id)

        # Apply search filter
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
            if status == "blocked":
                query = query.filter(User.is_blocked == True)
            elif status == "active":
                query = query.filter(User.is_blocked == False)

        # Count total
        total = query.count()

        # Calculate pagination
        pages = (total + limit - 1) // limit
        offset = (page - 1) * limit

        # Get page of results
        results = query.order_by(desc(User.last_active)).offset(offset).limit(limit).all()

        # Convert to response items
        items = []
        for user, messages_count in results:
            item = UserListItem(
                id=user.id,
                phone=user.phone_number,
                name=user.whatsapp_name,
                messages_count=messages_count,
                last_active=user.last_active.isoformat() if user.last_active else "",
                status="blocked" if user.is_blocked else "active",
            )
            items.append(item)

        return UserListResponse(
            users=items,
            total=total,
            page=page,
            limit=limit,
        )

    def get_user_detail(self, db: Session, user_id: int) -> Optional[UserDetailResponse]:
        """
        Get user detail with conversations

        Args:
            db: Database session
            user_id: User ID

        Returns:
            User detail response or None
        """
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return None

        # Get user conversations with last message info
        conversations = (
            db.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(desc(Conversation.started_at))
            .all()
        )

        # Build conversation items with last message
        conv_items = []
        for conv in conversations:
            # Get last message for this conversation
            last_msg = (
                db.query(Message)
                .filter(Message.conversation_id == conv.id)
                .order_by(desc(Message.created_at))
                .first()
            )

            conv_items.append(
                UserConversationItem(
                    id=str(conv.id),
                    last_message=last_msg.content if last_msg else "",
                    last_message_time=last_msg.created_at.isoformat() if last_msg else conv.started_at.isoformat(),
                    message_count=conv.message_count,
                )
            )

        # Calculate total tokens and avg response time
        total_tokens = (
            db.query(func.coalesce(func.sum(Message.llm_tokens), 0))
            .filter(Message.user_id == user_id)
            .scalar()
        )

        avg_response_time = (
            db.query(func.coalesce(func.avg(Message.response_time_ms), 0))
            .filter(Message.user_id == user_id, Message.role == "assistant")
            .scalar()
        )

        detail = UserDetail(
            id=user.id,
            phone=user.phone_number,
            name=user.whatsapp_name,
            profile_pic_url=user.profile_pic_url,
            language=user.language or "en",
            created_at=user.created_at.isoformat() if user.created_at else "",
            last_active=user.last_active.isoformat() if user.last_active else "",
            status="blocked" if user.is_blocked else "active",
            notes=user.notes,
            conversations=conv_items,
            total_tokens_used=int(total_tokens or 0),
            avg_response_time_ms=int(avg_response_time or 0),
        )

        return UserDetailResponse(user=detail)

    def block_user(self, db: Session, user_id: int) -> Optional[BlockUserResponse]:
        """
        Block a user

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Block response or None if user not found
        """
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return None

        user.is_blocked = True
        db.commit()

        logger.info(f"Blocked user {user_id}")
        return BlockUserResponse(message="User blocked successfully")

    def unblock_user(self, db: Session, user_id: int) -> Optional[BlockUserResponse]:
        """
        Unblock a user

        Args:
            db: Database session
            user_id: User ID

        Returns:
            Unblock response or None if user not found
        """
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return None

        user.is_blocked = False
        db.commit()

        logger.info(f"Unblocked user {user_id}")
        return BlockUserResponse(message="User unblocked successfully")

    def update_notes(self, db: Session, user_id: int, notes: str) -> Optional[Dict[str, Any]]:
        """
        Update user notes

        Args:
            db: Database session
            user_id: User ID
            notes: Notes text

        Returns:
            Success dict or None if user not found
        """
        user = db.query(User).filter(User.id == user_id).first()

        if not user:
            return None

        user.notes = notes
        db.commit()

        logger.info(f"Updated notes for user {user_id}")
        return {"success": True, "message": "Notes updated successfully"}

    def export_users(
        self,
        db: Session,
        format: str = "csv",
        search: Optional[str] = None,
        status: Optional[str] = None,
    ) -> str:
        """
        Export users as CSV or JSON

        Args:
            db: Database session
            format: Export format ("csv" or "json")
            search: Search filter
            status: Status filter

        Returns:
            Exported data as string
        """
        # Subquery for messages count
        messages_count_subq = (
            db.query(
                Message.user_id,
                func.count(Message.id).label("messages_count"),
            )
            .group_by(Message.user_id)
            .subquery()
        )

        query = db.query(
            User,
            func.coalesce(messages_count_subq.c.messages_count, 0).label("messages_count"),
        ).outerjoin(messages_count_subq, User.id == messages_count_subq.c.user_id)

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
            if status == "blocked":
                query = query.filter(User.is_blocked == True)
            elif status == "active":
                query = query.filter(User.is_blocked == False)

        results = query.order_by(desc(User.last_active)).all()

        if format == "json":
            data = []
            for user, messages_count in results:
                data.append({
                    "phone_number": user.phone_number,
                    "whatsapp_name": user.whatsapp_name,
                    "language": user.language,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "last_active": user.last_active.isoformat() if user.last_active else None,
                    "is_blocked": user.is_blocked,
                    "messages_count": messages_count,
                })
            return json.dumps(data, indent=2)
        else:
            # CSV format
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "phone_number", "whatsapp_name", "language",
                "created_at", "last_active", "is_blocked", "messages_count"
            ])
            for user, messages_count in results:
                writer.writerow([
                    user.phone_number,
                    user.whatsapp_name or "",
                    user.language or "",
                    user.created_at.isoformat() if user.created_at else "",
                    user.last_active.isoformat() if user.last_active else "",
                    user.is_blocked,
                    messages_count,
                ])
            return output.getvalue()


# Global user service instance
user_service = UserService()
