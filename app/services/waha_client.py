"""WAHA API client"""

import httpx
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class WAHAClient:
    """WAHA (WhatsApp HTTP API) client"""
    
    def __init__(self, session: str = "default"):
        self.base_url = settings.WAHA_API_URL.rstrip('/').rstrip('/api')
        self.api_key = settings.WAHA_API_KEY
        self.session = session
        self.timeout = 30.0
    
    def send_message(self, to: str, text: str, chat_id: str = None) -> dict:
        """
        Send text message via WAHA
        
        Args:
            to: Phone number (e.g., "6281234567890")
            text: Message text
            chat_id: Optional WhatsApp chat ID (defaults to {to}@c.us)
            
        Returns:
            WAHA API response
        """
        if chat_id is None:
            chat_id = f"{to}@c.us"
        
        payload = {
            "session": self.session,
            "chatId": chat_id,
            "text": text
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/sendText",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                result = response.json()
                logger.info(f"Message sent to {to} via session {self.session}")
                return result
                
        except httpx.HTTPError as e:
            logger.error(f"WAHA API error: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def get_sessions(self) -> list:
        """Get all WAHA sessions"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/api/sessions",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"WAHA get sessions error: {str(e)}")
            return []
    
    def get_session_status(self, session: str = None) -> dict:
        """Get WAHA session status"""
        if session is None:
            session = self.session
            
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(
                    f"{self.base_url}/api/sessions/{session}",
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"WAHA session status error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def send_typing(self, to: str, chat_id: str = None) -> dict:
        """
        Send typing indicator (sedang mengetik...)
        WAHA API: POST /api/{session}/presence
        
        Args:
            to: Phone number (e.g., "6281234567890")
            chat_id: Optional WhatsApp chat ID (defaults to {to}@c.us)
        """
        if chat_id is None:
            chat_id = f"{to}@c.us"
        
        payload = {
            "chatId": chat_id,
            "presence": "typing"
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/{self.session}/presence",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                logger.info(f"Typing indicator sent to {to}")
                return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Failed to send typing: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.warning(f"Response: {e.response.text}")
            return {"status": "error"}
