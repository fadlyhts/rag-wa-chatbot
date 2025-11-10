"""Keep-alive job to prevent WAHA session timeout"""

import logging
from app.services.waha_client import WAHAClient
from app.config import settings

logger = logging.getLogger(__name__)


def keep_waha_session_alive():
    """
    Ping WAHA session to keep it alive
    Run this every 5 minutes via scheduler
    """
    try:
        waha = WAHAClient(session="default")
        status = waha.get_session_status()
        
        if status.get("status") == "WORKING":
            logger.info("WAHA session is alive and working")
        elif status.get("status") == "STOPPED":
            logger.error("WAHA session is STOPPED! Needs restart")
        else:
            logger.warning(f"WAHA session status: {status.get('status')}")
            
        return status
        
    except Exception as e:
        logger.error(f"Keep-alive check failed: {str(e)}")
        return {"status": "error", "error": str(e)}
