"""Message service"""

from sqlalchemy.orm import Session
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_or_create_user(phone_number: str, db: Session) -> User:
    """Get existing user or create new one"""
    user = db.query(User).filter(User.phone_number == phone_number).first()
    
    if not user:
        user = User(
            phone_number=phone_number,
            language="en",
            created_at=datetime.utcnow()
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new user: {user.id} ({phone_number})")
    else:
        # Update last active
        user.last_active = datetime.utcnow()
        db.commit()
    
    return user


def get_or_create_conversation(user_id: int, db: Session) -> Conversation:
    """Get active conversation or create new one"""
    # Look for active conversation
    conversation = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.is_active == True
    ).order_by(Conversation.started_at.desc()).first()
    
    # Create new conversation if none exists or old one is stale
    if not conversation or is_conversation_stale(conversation):
        if conversation:
            conversation.is_active = False
            conversation.ended_at = datetime.utcnow()
        
        conversation = Conversation(
            user_id=user_id,
            started_at=datetime.utcnow(),
            is_active=True
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        logger.info(f"Created new conversation: {conversation.id} for user {user_id}")
    
    return conversation


def is_conversation_stale(conversation: Conversation, hours: int = 24) -> bool:
    """Check if conversation is older than specified hours"""
    from datetime import timedelta
    age = datetime.utcnow() - conversation.started_at
    return age > timedelta(hours=hours)


def save_user_message(
    conversation_id: int,
    user_id: int,
    content: str,
    content_type: str = "text",
    db: Session = None
) -> Message:
    """Save user message to database"""
    message = Message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="user",
        content=content,
        content_type=content_type,
        created_at=datetime.utcnow()
    )
    db.add(message)
    
    # Update conversation message count
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation:
        conversation.message_count += 1
    
    db.commit()
    db.refresh(message)
    logger.info(f"Saved user message: {message.id}")
    return message


def save_assistant_message(
    conversation_id: int,
    user_id: int,
    content: str,
    rag_context: dict = None,
    llm_tokens: int = None,
    response_time_ms: int = None,
    db: Session = None
) -> Message:
    """Save assistant message to database"""
    message = Message(
        conversation_id=conversation_id,
        user_id=user_id,
        role="assistant",
        content=content,
        rag_context=rag_context,
        llm_tokens=llm_tokens,
        response_time_ms=response_time_ms,
        created_at=datetime.utcnow(),
        processed_at=datetime.utcnow()
    )
    db.add(message)
    
    # Update conversation message count
    conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conversation:
        conversation.message_count += 1
    
    db.commit()
    db.refresh(message)
    logger.info(f"Saved assistant message: {message.id}")
    return message


def get_conversation_history(
    conversation_id: int,
    limit: int = 10,
    db: Session = None
) -> list[Message]:
    """Get recent conversation history"""
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at.desc()).limit(limit).all()
    
    # Return in chronological order
    return list(reversed(messages))
