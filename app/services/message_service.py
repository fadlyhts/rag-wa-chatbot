"""Message service"""

from sqlalchemy.orm import Session
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from datetime import datetime
import logging
import time
from typing import Dict, Any

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


async def generate_ai_response(
    user_message: str,
    conversation_id: int,
    user_id: int,
    db: Session
) -> Dict[str, Any]:
    """
    Generate AI response using RAG pipeline
    
    Args:
        user_message: User's message text
        conversation_id: Conversation ID
        user_id: User ID
        db: Database session
        
    Returns:
        Dict with response text and metadata
    """
    try:
        # Import RAG here to avoid circular imports
        from app.rag import generate_rag_response_async
        
        # Get conversation history
        history = get_conversation_history(conversation_id, limit=5, db=db)
        
        # Format history for RAG
        formatted_history = [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ]
        
        # Call RAG pipeline
        logger.info(f"Generating RAG response for user {user_id}")
        start_time = time.time()
        
        response = await generate_rag_response_async(
            query=user_message,
            conversation_history=formatted_history,
            user_id=user_id
        )
        
        response_time = int((time.time() - start_time) * 1000)
        logger.info(f"RAG response generated in {response_time}ms for user {user_id}")
        
        # Save assistant response with metadata
        save_assistant_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=response['text'],
            rag_context={
                'retrieved_docs': response['sources'],
                'relevance_scores': response['scores'],
                'docs_retrieved': response['docs_retrieved']
            },
            llm_tokens=response['tokens'],
            response_time_ms=response.get('total_time_ms', response_time),
            db=db
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}", exc_info=True)
        
        # Fallback to simple response
        fallback_text = (
            "I apologize, but I'm having trouble processing your request right now. "
            "Please try again in a moment. 😊"
        )
        
        # Still save the fallback response
        save_assistant_message(
            conversation_id=conversation_id,
            user_id=user_id,
            content=fallback_text,
            rag_context={'error': str(e)},
            db=db
        )
        
        return {
            'text': fallback_text,
            'error': str(e)
        }
