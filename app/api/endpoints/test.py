"""Test endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.waha_client import WAHAClient
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class SendMessageRequest(BaseModel):
    """Send message request"""
    to: str
    text: str
    session: str = "default"


@router.post("/test/send-message")
async def test_send_message(request: SendMessageRequest):
    """
    Test sending a WhatsApp message via WAHA
    
    Example:
    ```json
    {
        "to": "6285156121852",
        "text": "Hello from backend!",
        "session": "default"
    }
    ```
    """
    try:
        waha = WAHAClient(session=request.session)
        result = waha.send_message(to=request.to, text=request.text)
        
        return {
            "status": "sent",
            "result": result,
            "session_used": request.session
        }
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test/sessions")
async def test_get_sessions():
    """Get all WAHA sessions"""
    try:
        waha = WAHAClient()
        sessions = waha.get_sessions()
        
        return {
            "total": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
