"""Users management API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.database.session import get_db
from app.security.auth import get_current_active_admin
from app.models.admin import Admin
from app.schemas.user import (
    UserListResponse,
    UserDetailResponse,
    BlockUserResponse,
    UpdateNotesRequest,
    UserCreate,
)
from app.services.user_service import user_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/users/export")
async def export_users(
    format: str = Query("csv", regex="^(csv|json)$"),
    search: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Export users data as CSV or JSON file
    """
    try:
        data = user_service.export_users(
            db=db,
            format=format,
            search=search,
            status=status,
        )

        if format == "json":
            return Response(
                content=data,
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=users_export.json"},
            )
        else:
            return Response(
                content=data,
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=users_export.csv"},
            )

    except Exception as e:
        logger.error(f"Error exporting users: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export users: {str(e)}")


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    List users with pagination, search, and status filter

    - **page**: Page number (1-indexed)
    - **limit**: Items per page (1-100)
    - **search**: Search in phone_number and whatsapp_name
    - **status**: Filter by status (active/blocked/all)
    """
    try:
        result = user_service.list_users(
            db=db,
            page=page,
            limit=limit,
            search=search,
            status=status,
        )
        return result
    except Exception as e:
        logger.error(f"Error listing users: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")

@router.post("/users")
async def create_user(
    request: UserCreate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Create a new WhatsApp user
    """
    try:
        return user_service.create_user(db, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Get user detail with conversations
    """
    try:
        result = user_service.get_user_detail(db, user_id)

        if not result:
            raise HTTPException(status_code=404, detail="User not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")


@router.put("/users/{user_id}/block", response_model=BlockUserResponse)
async def block_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Block a user
    """
    try:
        result = user_service.block_user(db, user_id)

        if not result:
            raise HTTPException(status_code=404, detail="User not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error blocking user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to block user: {str(e)}")


@router.put("/users/{user_id}/unblock", response_model=BlockUserResponse)
async def unblock_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Unblock a user
    """
    try:
        result = user_service.unblock_user(db, user_id)

        if not result:
            raise HTTPException(status_code=404, detail="User not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unblocking user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to unblock user: {str(e)}")


@router.put("/users/{user_id}/notes")
async def update_user_notes(
    user_id: int,
    request: UpdateNotesRequest,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Update user notes
    """
    try:
        result = user_service.update_notes(db, user_id, request.notes)

        if not result:
            raise HTTPException(status_code=404, detail="User not found")

        return result

    except HTTPException:
        raise
        raise HTTPException(status_code=500, detail=f"Failed to update notes: {str(e)}")

from pydantic import BaseModel

class UpdateDivisionRequest(BaseModel):
    division_id: Optional[int] = None

@router.put("/users/{user_id}/division")
async def update_user_division(
    user_id: int,
    request: UpdateDivisionRequest,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin),
):
    """
    Update user division
    """
    try:
        from app.models.user import User
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
            
        user.division_id = request.division_id
        db.commit()
        db.refresh(user)
        
        return {"status": "success", "message": "User division updated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating division for user {user_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update division: {str(e)}")
