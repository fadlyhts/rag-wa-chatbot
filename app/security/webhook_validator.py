"""Webhook signature validation"""

import hmac
import hashlib
import logging

logger = logging.getLogger(__name__)


def validate_webhook_signature(payload: dict, signature: str, secret: str) -> bool:
    """
    Validate WAHA webhook signature
    
    Args:
        payload: Webhook payload
        signature: Signature from header
        secret: Webhook secret key
        
    Returns:
        True if signature is valid
    """
    if not signature or not secret:
        logger.warning("Webhook signature validation disabled (no signature or secret)")
        return True  # Skip validation if not configured
    
    try:
        # Convert payload to string
        import json
        payload_str = json.dumps(payload, sort_keys=True)
        
        # Calculate expected signature
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        is_valid = hmac.compare_digest(signature, expected_signature)
        
        if not is_valid:
            logger.warning("Invalid webhook signature")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Webhook validation error: {str(e)}")
        return False
