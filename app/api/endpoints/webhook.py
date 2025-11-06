"""Webhook endpoint"""

from fastapi import APIRouter, HTTPException, Depends, Header, BackgroundTasks
from sqlalchemy.orm import Session
from app.schemas.webhook import WebhookPayload, WebhookResponse
from app.database.session import get_db
from app.services.message_service import get_or_create_user, get_or_create_conversation, save_user_message
from app.services.waha_client import WAHAClient
from app.security.rate_limiter import RateLimiter
from app.config import settings
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)
rate_limiter = RateLimiter()


@router.post("/webhook")
async def webhook(
    request: dict,
    background_tasks: BackgroundTasks,
    x_signature: str = Header(None, alias="X-Signature"),
    db: Session = Depends(get_db)
):
    """
    Receive webhooks from WAHA
    - Validates incoming events
    - Rate limits to 10 msg/min per user
    - Responds within <100ms
    - Sends auto-reply in background
    """
    request_id = f"webhook_{int(time.time() * 1000)}"
    
    try:
        # Log raw payload for debugging
        logger.info(f"[{request_id}] Raw webhook payload: {request}")
        
        event = request.get("event")
        data = request.get("payload") or request.get("data") or {}
        
        logger.info(f"[{request_id}] Webhook event: {event}")
        
        # Handle based on event type
        if event == "message" or event == "message.any":
            return await handle_incoming_message_raw(data, request_id, background_tasks, db)
        elif event == "message.status":
            return {"status": "acknowledged", "request_id": request_id}
        elif event == "session.status":
            return {"status": "acknowledged", "request_id": request_id}
        else:
            logger.info(f"[{request_id}] Ignored event: {event}")
            return {"status": "ignored", "request_id": request_id}
    
    except Exception as e:
        logger.error(f"[{request_id}] Webhook error: {str(e)}", exc_info=True)
        return {"status": "error", "detail": str(e), "request_id": request_id}


async def handle_incoming_message_raw(data: dict, request_id: str, background_tasks: BackgroundTasks, db: Session):
    """Handle incoming message with raw dict data"""
    # Extract phone number - WAHA format varies
    phone = data.get("from") or data.get("chatId", "").split("@")[0]
    message_text = data.get("body") or data.get("text") or ""
    message_id = data.get("id") or data.get("messageId", "")
    
    logger.info(f"[{request_id}] Processing message from {phone}: {message_text[:50]}")
    
    # Rate limit check
    if not rate_limiter.allow_request(phone, limit=settings.RATE_LIMIT_MESSAGES_PER_MINUTE):
        logger.warning(f"[{request_id}] Rate limit exceeded for {phone}")
        return {"status": "rate_limited", "request_id": request_id}
    
    try:
        # Get or create user and conversation
        user = get_or_create_user(phone, db)
        conversation = get_or_create_conversation(user.id, db)
        
        # Save user message
        save_user_message(
            conversation_id=conversation.id,
            user_id=user.id,
            content=message_text,
            db=db
        )
        
        logger.info(f"[{request_id}] Message saved from user {user.id} ({phone})")
        
        # Send auto-reply in background
        background_tasks.add_task(
            send_auto_reply,
            phone=phone,
            user_message=message_text,
            request_id=request_id
        )
        
        return {"status": "queued", "request_id": request_id, "message_id": message_id}
        
    except Exception as e:
        logger.error(f"[{request_id}] Error handling message: {str(e)}", exc_info=True)
        return {"status": "error", "detail": str(e), "request_id": request_id}


async def handle_incoming_message(payload: WebhookPayload, request_id: str, background_tasks: BackgroundTasks, db: Session):
    """Handle incoming message event"""
    phone = payload.data.get("from")
    message_text = payload.data.get("text", "")
    message_id = payload.data.get("messageId", "")
    
    # Rate limit check
    if not rate_limiter.allow_request(phone, limit=settings.RATE_LIMIT_MESSAGES_PER_MINUTE):
        logger.warning(f"[{request_id}] Rate limit exceeded for {phone}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {settings.RATE_LIMIT_MESSAGES_PER_MINUTE} messages/minute"
        )
    
    try:
        # Get or create user and conversation
        user = get_or_create_user(phone, db)
        conversation = get_or_create_conversation(user.id, db)
        
        # Save user message
        save_user_message(
            conversation_id=conversation.id,
            user_id=user.id,
            content=message_text,
            db=db
        )
        
        logger.info(f"[{request_id}] Message from user {user.id} ({phone}): {message_text[:50]}")
        
        # Send auto-reply in background
        background_tasks.add_task(
            send_auto_reply,
            phone=phone,
            user_message=message_text,
            request_id=request_id
        )
        
        # Placeholder job_id
        job_id = f"job_{int(time.time() * 1000)}"
        
        return WebhookResponse(
            status="queued",
            request_id=request_id,
            job_id=job_id,
            message_id=message_id
        )
        
    except Exception as e:
        logger.error(f"[{request_id}] Error handling message: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def handle_message_status(payload: WebhookPayload, request_id: str):
    """Handle message status update"""
    message_id = payload.data.get("messageId")
    status = payload.data.get("status")
    
    logger.info(f"[{request_id}] Message {message_id} status: {status}")
    
    return WebhookResponse(
        status="acknowledged",
        request_id=request_id,
        message_id=message_id
    )


def send_auto_reply(phone: str, user_message: str, request_id: str):
    """
    Send auto-reply to user
    Later this will be replaced with RAG-powered responses
    """
    try:
        logger.info(f"[{request_id}] Starting auto-reply to {phone}")
        
        # Initialize WAHA client
        waha = WAHAClient(session="default")
        logger.info(f"[{request_id}] WAHA client initialized")
        
        # Generate simple response (will be replaced with RAG)
        reply_text = generate_simple_response(user_message)
        logger.info(f"[{request_id}] Generated reply: {reply_text[:50]}...")
        
        # Send message
        logger.info(f"[{request_id}] Sending message to {phone}")
        result = waha.send_message(to=phone, text=reply_text)
        
        logger.info(f"[{request_id}] Auto-reply sent successfully to {phone}. Result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"[{request_id}] Error sending auto-reply: {str(e)}", exc_info=True)


def generate_simple_response(user_message: str) -> str:
    """
    Generate simple response based on user message
    TODO: Replace with RAG pipeline
    """
    message_lower = user_message.lower()
    
    # Simple keyword-based responses
    if any(word in message_lower for word in ["hello", "hi", "hey", "halo"]):
        return "Hello! 👋 Thank you for contacting us. I'm your AI assistant. How can I help you today?"
    
    elif any(word in message_lower for word in ["help", "bantuan"]):
        return "I'm here to help! You can ask me about:\n• Business hours\n• Products and services\n• Order status\n• General inquiries\n\nWhat would you like to know?"
    
    elif any(word in message_lower for word in ["hours", "jam", "buka"]):
        return "Our business hours are:\n📅 Monday - Friday: 9:00 AM - 6:00 PM\n📅 Saturday: 9:00 AM - 3:00 PM\n📅 Sunday: Closed\n\nHow else can I assist you?"
    
    elif any(word in message_lower for word in ["thank", "thanks", "terima kasih"]):
        return "You're welcome! 😊 Is there anything else I can help you with?"
    
    else:
        return f"Thank you for your message! I've received: \"{user_message[:50]}...\"\n\nI'm currently in demo mode. Soon I'll be powered by AI to give you intelligent responses! 🤖\n\nHow can I assist you further?"
