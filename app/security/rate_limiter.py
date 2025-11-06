"""Rate limiting implementation"""

from redis import Redis
from app.config import settings
import time
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter using Redis"""
    
    def __init__(self):
        self.redis = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5
        )
    
    def allow_request(self, identifier: str, limit: int = 10, window: int = 60) -> bool:
        """
        Check if request is allowed under rate limit
        
        Args:
            identifier: Unique identifier (e.g., phone number)
            limit: Maximum requests per window
            window: Time window in seconds
            
        Returns:
            True if request allowed, False otherwise
        """
        key = f"rate_limit:{identifier}"
        
        try:
            # Get current count
            current = self.redis.get(key)
            
            if current is None:
                # First request in window
                self.redis.setex(key, window, 1)
                return True
            
            count = int(current)
            
            if count < limit:
                # Under limit, increment
                self.redis.incr(key)
                return True
            
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for {identifier}: {count}/{limit}")
            return False
            
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            # Fail open - allow request if Redis unavailable
            return True
    
    def get_remaining(self, identifier: str, limit: int = 10) -> int:
        """Get remaining requests in current window"""
        key = f"rate_limit:{identifier}"
        
        try:
            current = self.redis.get(key)
            if current is None:
                return limit
            
            return max(0, limit - int(current))
        except Exception:
            return limit
