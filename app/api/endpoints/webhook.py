"""Webhook endpoint"""

from fastapi import APIRouter, HTTPException, Depends, Header, BackgroundTasks
from sqlalchemy.orm import Session
from app.schemas.webhook import WebhookPayload, WebhookResponse
from app.database.session import get_db
from app.services.message_service import (
    get_or_create_user,
    get_or_create_conversation,
    save_user_message,
    generate_ai_response
)
from app.services.waha_client import WAHAClient
from app.security.rate_limiter import RateLimiter
from app.config import settings
import logging
import time
import redis

router = APIRouter()
logger = logging.getLogger(__name__)
rate_limiter = RateLimiter()

# Redis client for message deduplication
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


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
        if event == "message":
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
    # Remove @c.us suffix if present to get clean phone number
    phone_raw = data.get("from") or data.get("chatId", "")
    phone = phone_raw.split("@")[0] if "@" in phone_raw else phone_raw
    message_text = data.get("body") or data.get("text") or ""
    message_id = data.get("id") or data.get("messageId", "")
    from_me = data.get("fromMe", False)
    
    # Check for document/media attachments
    has_media = data.get("hasMedia", False)
    media_url = None
    media_type = None
    
    if has_media:
        # WAHA includes media information in the payload
        media_url = data.get("mediaUrl") or data.get("media", {}).get("url")
        media_type = data.get("mimetype") or data.get("media", {}).get("mimetype")
        
        logger.info(f"[{request_id}] Message has media: type={media_type}, url={media_url}")
    
    # FILTER 1: Ignore messages sent by the bot itself
    if from_me:
        logger.info(f"[{request_id}] Ignoring message from bot itself (fromMe=True)")
        return {"status": "ignored_own_message", "request_id": request_id}
    
    # FILTER 2: Ignore group messages (chatId contains @g.us)
    if "@g.us" in phone_raw:
        logger.info(f"[{request_id}] Ignoring group message from {phone_raw}")
        return {"status": "ignored_group", "request_id": request_id}
    # FILTER 3: Ignore status/broadcast messages (numbers starting with 120363)
    if phone.startswith("120363") or phone.startswith("status"):
        logger.info(f"[{request_id}] Ignoring status/broadcast message from {phone}")
        return {"status": "ignored_status", "request_id": request_id}
    
    # FILTER 5: Ignore non-actionable or empty messages (system notifications)
    message_type = data.get("type", "")
    if not message_text.strip() and not has_media:
        logger.info(f"[{request_id}] Ignoring empty/system message (type: {message_type})")
        return {"status": "ignored_empty", "request_id": request_id}
    
    # FILTER 4: Only process valid phone numbers (6-15 digits)
    if not phone.isdigit() or len(phone) < 6 or len(phone) > 15:
        logger.info(f"[{request_id}] Ignoring invalid phone number: {phone}")
        return {"status": "ignored_invalid_phone", "request_id": request_id}
    
    # Deduplication: Check if we already processed this message
    # WAHA sends both 'message' and 'message.any' events for the same message
    dedup_key = f"msg:{message_id}"
    try:
        is_new = redis_client.setnx(dedup_key, "1")
        if not is_new:
            logger.info(f"[{request_id}] Duplicate message {message_id}, skipping")
            return {"status": "duplicate", "request_id": request_id, "message_id": message_id}
        # Mark as processed for 60 seconds
        redis_client.expire(dedup_key, 60)
    except Exception as e:
        logger.warning(f"[{request_id}] Redis deduplication failed: {e}, continuing anyway")
    
    logger.info(f"[{request_id}] Processing message from {phone}: {message_text[:50]}")
    
    # Extract WhatsApp display name from _data.notifyName
    whatsapp_name = None
    _data = data.get("_data", {})
    if isinstance(_data, dict):
        whatsapp_name = _data.get("notifyName")
    
    # Resolve LID to real phone number if the sender uses @lid format
    real_phone = phone
    if "@lid" in phone_raw:
        try:
            from app.services.waha_client import WAHAClient
            waha = WAHAClient(session="default")
            resolved = waha.resolve_lid(phone)
            if resolved:
                logger.info(f"[{request_id}] Resolved LID {phone} -> {resolved}")
                real_phone = resolved
            else:
                logger.info(f"[{request_id}] Could not resolve LID {phone}, using as-is")
        except Exception as e:
            logger.warning(f"[{request_id}] LID resolution failed: {e}, using LID as phone")
    
    # Rate limit check
    if not rate_limiter.allow_request(real_phone, limit=settings.RATE_LIMIT_MESSAGES_PER_MINUTE):
        logger.warning(f"[{request_id}] Rate limit exceeded for {real_phone}")
        return {"status": "rate_limited", "request_id": request_id}
    
    try:
        # Get or create user and conversation
        user = get_or_create_user(real_phone, db, whatsapp_name=whatsapp_name)
        
        # Check if user is blocked
        if user.is_blocked:
            logger.info(f"[{request_id}] User {real_phone} is blocked, ignoring message")
            return {"status": "blocked", "request_id": request_id}
        
        conversation = get_or_create_conversation(user.id, db)
        
        # Save user message
        save_user_message(
            conversation_id=conversation.id,
            user_id=user.id,
            content=message_text,
            db=db
        )
        
        logger.info(f"[{request_id}] Message saved from user {user.id} ({real_phone})")
        
        # Send auto-reply in background
        background_tasks.add_task(
            send_auto_reply,
            phone=real_phone,
            phone_raw=phone_raw,  # sertakan suffix @lid / @c.us
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


def markdown_to_whatsapp(text: str) -> str:
    """Convert standard Markdown to WhatsApp formatting"""
    import re
    # Convert bold **text** to *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    # Convert headers # Header to *Header*
    text = re.sub(r'^(?:#+)\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    # Convert lists * or - to • 
    text = re.sub(r'^(\s*)[*\-]\s+', r'\1• ', text, flags=re.MULTILINE)
    return text


def format_sources_for_whatsapp(
    sources_metadata: list,
    min_score: float = 0.0,
    max_sources: int = 3,
) -> str:
    """
    Format source documents into WhatsApp-friendly text.
    Groups multiple pages from the same file into a single line.
    """
    relevant = [
        s for s in sources_metadata
        if s.get('score', 0) >= min_score
    ]
    if not relevant:
        return ""

    # Group by file_name
    grouped_sources = {}
    for s in relevant:
        file_name = s.get('file_name') or s.get('title') or 'Document'
        page = s.get('page_number')
        pages = s.get('page_numbers')
        url = s.get('url')
        
        if file_name not in grouped_sources:
            grouped_sources[file_name] = {'pages': set(), 'url': url}
            
        if pages and isinstance(pages, list):
            grouped_sources[file_name]['pages'].update(pages)
        elif page:
            grouped_sources[file_name]['pages'].add(page)

    lines = ["\n\n📚 *Sumber:*"]
    for i, (file_name, data) in enumerate(grouped_sources.items(), 1):
        if i > max_sources:
            break
            
        url = data['url']
        page_set = sorted(list(data['pages']))
        
        if url:
            lines.append(f"{i}. [{file_name}]({url})")
        elif page_set:
            page_str = ", ".join(str(p) for p in page_set)
            lines.append(f"{i}. {file_name} (hal. {page_str})")
        else:
            lines.append(f"{i}. {file_name}")

    return "\n".join(lines)


async def send_auto_reply(phone: str, user_message: str, request_id: str, phone_raw: str = None):
    """
    Send AI-powered auto-reply using RAG system.
    phone_raw: Full number with suffix (@c.us / @lid) from WAHA.
               Required because newer WhatsApp accounts use @lid instead of @c.us.
    """
    from app.database.session import get_db
    
    try:
        logger.info(f"[{request_id}] Starting AI reply to {phone}")
        
        # Initialize WAHA client
        waha = WAHAClient(session="default")
        
        # Send typing indicator
        if phone_raw:
            waha.send_typing(to=phone, chat_id=phone_raw)
        else:
            waha.send_typing(to=phone)
        logger.info(f"[{request_id}] Typing indicator sent")
        
        # Get database session
        db = next(get_db())
        
        try:
            # Get user and conversation
            logger.info(f"[{request_id}] Getting user from database")
            user = get_or_create_user(phone, db)
            logger.info(f"[{request_id}] User ID: {user.id}")
            
            conversation = get_or_create_conversation(user.id, db)
            logger.info(f"[{request_id}] Conversation ID: {conversation.id}")
            
            # Generate AI response using RAG
            logger.info(f"[{request_id}] Generating AI response with RAG")
            response = await generate_ai_response(
                user_message=user_message,
                conversation_id=conversation.id,
                user_id=user.id,
                db=db
            )
            
            reply_text = response['text']
            sources_metadata = response.get('sources_metadata', [])
            logger.info(f"[{request_id}] AI response generated: {reply_text[:100]}...")
            
            # Log retrieved source documents
            if sources_metadata:
                logger.info(f"[{request_id}] Sources ({len(sources_metadata)} docs):")
                for i, src in enumerate(sources_metadata, 1):
                    logger.info(
                        f"  [{i}] {src.get('file_name', '-')} "
                        f"p.{src.get('page_number', '-')} "
                        f"score={src.get('score', 0):.3f}"
                    )
            else:
                logger.warning(f"[{request_id}] No source documents retrieved")
            
            # Append source references to reply
            # Use regex to find all citations like [1], [1, 2]
            import re
            citations = re.findall(r'\[([\d,\s]+)\]', reply_text)
            cited_indices = set()
            for citation in citations:
                for num in re.split(r'[,\s]+', citation):
                    if num.strip().isdigit():
                        cited_indices.add(int(num.strip()))
            
            # If the LLM cited specific indices, filter sources_metadata
            if cited_indices:
                cited_metadata = []
                for idx in cited_indices:
                    if 1 <= idx <= len(sources_metadata):
                        cited_metadata.append(sources_metadata[idx - 1])
                # We can skip the min_score threshold here since the LLM explicitly used it
                sources_text = format_sources_for_whatsapp(cited_metadata, min_score=0.0)
            else:
                # If LLM didn't cite anything (e.g. said "I don't know"), don't append sources
                sources_text = ""
                
            # Clean up the citations from reply text
            clean_reply_text = re.sub(r'\s*\[[\d,\s]+\]', '', reply_text).strip()
            
            # Format markdown to WhatsApp
            clean_reply_text = markdown_to_whatsapp(clean_reply_text)
            
            if sources_text:
                reply_text = clean_reply_text + sources_text
                logger.info(f"[{request_id}] Sources appended to reply. Cited indices: {cited_indices}")
            else:
                reply_text = clean_reply_text
                logger.info(f"[{request_id}] No high-relevance sources to append")
            
            # Use phone_raw (@lid/@c.us) as chat_id for newer WhatsApp accounts
            chat_id = phone_raw if phone_raw else None
            logger.info(f"[{request_id}] Sending message to {phone} (chat_id={chat_id})")
            result = waha.send_message(to=phone, text=reply_text, chat_id=chat_id)
            logger.info(f"[{request_id}] WAHA send result: {result}")
            
            logger.info(
                f"[{request_id}] AI reply sent successfully. "
                f"Tokens: {response.get('tokens', 0)}, "
                f"Docs: {response.get('docs_retrieved', 0)}, "
                f"Time: {response.get('total_time_ms', 0)}ms"
            )
            return result
            
        finally:
            db.close()
        
    except Exception as e:
        logger.error(f"[{request_id}] Error sending AI reply: {str(e)}", exc_info=True)
        
        # Send error message to user
        try:
            waha = WAHAClient(session="default")
            error_msg = "Maaf, saya mengalami kesulitan memproses permintaan Anda. Silakan coba lagi dalam beberapa saat. 😊"
            waha.send_message(to=phone, text=error_msg, chat_id=phone_raw if phone_raw else None)
            logger.info(f"[{request_id}] Error message sent to user")
        except Exception as send_error:
            logger.error(f"[{request_id}] Failed to send error message: {str(send_error)}")


# Keyword-based fallback (kept for emergency use only)
def generate_simple_response(user_message: str) -> str:
    """
    Simple keyword-based response (fallback only)
    Main responses now use RAG pipeline
    """
    message_lower = user_message.lower()
    
    if any(word in message_lower for word in ["hello", "hi", "hey", "halo"]):
        return "Hello! 👋 I'm your AI assistant. How can I help you today?"
    
    return "Thank you for your message! Our AI system is processing your request. Please wait a moment. 😊"
