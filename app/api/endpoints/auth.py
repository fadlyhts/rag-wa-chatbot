"""Authentication API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from app.database.session import get_db
from app.schemas.auth import LoginRequest, TokenResponse, AdminResponse
from app.security.auth import (
    authenticate_admin,
    create_access_token,
    get_current_active_admin
)
from app.models.admin import Admin

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    Admin login endpoint
    
    Authenticates admin user and returns JWT token
    """
    try:
        # Authenticate user
        admin = authenticate_admin(db, login_data.username, login_data.password)
        
        if not admin:
            logger.warning(f"Failed login attempt for username: {login_data.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        admin.last_login = datetime.utcnow()
        db.commit()
        
        # Create access token
        access_token = create_access_token(
            data={"sub": admin.username, "role": admin.role}
        )
        
        logger.info(f"Admin logged in: {admin.username} (role: {admin.role})")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            role=admin.role,
            user_info={
                "id": admin.id,
                "username": admin.username,
                "email": admin.email,
                "role": admin.role
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/auth/me", response_model=AdminResponse)
async def get_current_user(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get current authenticated admin user
    
    Requires valid JWT token
    """
    return AdminResponse.from_orm(current_admin)


@router.post("/auth/logout")
async def logout(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Logout endpoint
    
    Currently just validates token. Client should discard token.
    In production, consider token blacklisting with Redis.
    """
    logger.info(f"Admin logged out: {current_admin.username}")
    
    return JSONResponse(
        content={
            "success": True,
            "message": "Logged out successfully"
        }
    )


@router.post("/auth/refresh")
async def refresh_token(
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Refresh access token
    
    Returns a new JWT token for the current user
    """
    try:
        # Create new access token
        access_token = create_access_token(
            data={"sub": current_admin.username, "role": current_admin.role}
        )
        
        logger.info(f"Token refreshed for admin: {current_admin.username}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            role=current_admin.role,
            user_info={
                "id": current_admin.id,
                "username": current_admin.username,
                "email": current_admin.email,
                "role": current_admin.role
            }
        )
        
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )
