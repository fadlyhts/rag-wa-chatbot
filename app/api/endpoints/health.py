"""Health check endpoint"""

from fastapi import APIRouter, Response, status
from sqlalchemy import text
from app.database.session import SessionLocal
from app.schemas.response import HealthResponse
from app.config import settings
from redis import Redis
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check(response: Response):
    """
    Health check endpoint
    Checks connectivity to:
    - MySQL database
    - Redis
    - Qdrant (optional)
    - WAHA (optional)
    """
    health_status = {
        "status": "healthy",
        "dependencies": {}
    }
    
    # Check database
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        health_status["dependencies"]["database"] = "connected"
    except Exception as e:
        health_status["dependencies"]["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
        logger.error(f"Database health check failed: {str(e)}")
    
    # Check Redis
    try:
        redis_client = Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        redis_client.ping()
        health_status["dependencies"]["redis"] = "connected"
    except Exception as e:
        health_status["dependencies"]["redis"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
        logger.error(f"Redis health check failed: {str(e)}")
    
    # Check Qdrant (optional - don't fail if not available)
    try:
        # Use httpx directly for external HTTPS Qdrant (EasyPanel)
        import httpx
        
        headers = {}
        if settings.QDRANT_API_KEY:
            headers["api-key"] = settings.QDRANT_API_KEY
        
        # Test with REST API directly
        qdrant_url = settings.QDRANT_URL.rstrip('/')
        response = httpx.get(
            f"{qdrant_url}/collections",
            headers=headers,
            timeout=10.0
        )
        response.raise_for_status()
        
        # If successful, Qdrant is connected
        collections_data = response.json()
        health_status["dependencies"]["qdrant"] = "connected"
        logger.info(f"Qdrant connected: {len(collections_data.get('result', {}).get('collections', []))} collections")
        
    except Exception as e:
        health_status["dependencies"]["qdrant"] = f"not available: {str(e)}"
        logger.warning(f"Qdrant health check failed: {str(e)}")
    
    # Check WAHA (optional - don't fail if not available)
    try:
        # Use httpx directly to test WAHA connection
        import httpx
        
        headers = {}
        if settings.WAHA_API_KEY:
            headers["X-Api-Key"] = settings.WAHA_API_KEY
        
        # Remove trailing /api to avoid double /api
        waha_base_url = settings.WAHA_API_URL.rstrip('/').rstrip('/api')
        
        # Test with sessions endpoint
        response = httpx.get(
            f"{waha_base_url}/api/sessions",
            headers=headers,
            timeout=10.0
        )
        response.raise_for_status()
        
        sessions_data = response.json()
        health_status["dependencies"]["waha"] = "connected"
        logger.info(f"WAHA connected: {len(sessions_data) if isinstance(sessions_data, list) else 'OK'}")
        
    except Exception as e:
        health_status["dependencies"]["waha"] = f"not available: {str(e)}"
        logger.warning(f"WAHA health check failed: {str(e)}")
    
    # Set HTTP status code
    if health_status["status"] == "unhealthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return HealthResponse(**health_status)
