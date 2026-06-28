"""Admin management API endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import logging
from datetime import datetime

from app.database.session import get_db
from app.models.admin import Admin, AdminRole
from app.schemas.auth import AdminResponse, AdminCreate, AdminUpdate
from app.security.auth import get_current_active_admin, require_role, get_password_hash

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/admins", response_model=List[AdminResponse])
async def list_admins(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(require_role(AdminRole.SUPER_ADMIN))
):
    """
    List all admins
    
    Requires SUPER_ADMIN role
    """
    try:
        admins = db.query(Admin).all()
        return admins
    except Exception as e:
        logger.error(f"Error listing admins: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list admins")


@router.post("/admins", response_model=AdminResponse, status_code=status.HTTP_201_CREATED)
async def create_admin(
    admin_in: AdminCreate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(require_role(AdminRole.SUPER_ADMIN))
):
    """
    Create new admin
    
    Requires SUPER_ADMIN role
    """
    try:
        # Check if username exists
        user = db.query(Admin).filter(Admin.username == admin_in.username).first()
        if user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
            
        new_admin = Admin(
            username=admin_in.username,
            email=admin_in.email,
            password_hash=get_password_hash(admin_in.password),
            role=admin_in.role,
            division_id=admin_in.division_id,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)
        
        logger.info(f"Created new admin: {new_admin.username}")
        return new_admin
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating admin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create admin")


@router.put("/admins/{admin_id}", response_model=AdminResponse)
async def update_admin(
    admin_id: int,
    admin_in: AdminUpdate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(require_role(AdminRole.SUPER_ADMIN))
):
    """
    Update admin details
    
    Requires SUPER_ADMIN role
    """
    try:
        admin = db.query(Admin).filter(Admin.id == admin_id).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin not found")
            
        if admin_in.email is not None:
            admin.email = admin_in.email
            
        if admin_in.password:
            admin.password_hash = get_password_hash(admin_in.password)
            
        if admin_in.role is not None:
            admin.role = admin_in.role
            
        if admin_in.division_id is not None:
            admin.division_id = admin_in.division_id
            
        if admin_in.is_active is not None:
            # Don't let user deactivate themselves
            if admin_id == current_admin.id and not admin_in.is_active:
                raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
            admin.is_active = admin_in.is_active
            
        db.commit()
        db.refresh(admin)
        
        logger.info(f"Updated admin: {admin.username}")
        return admin
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating admin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update admin")


@router.delete("/admins/{admin_id}")
async def delete_admin(
    admin_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(require_role(AdminRole.SUPER_ADMIN))
):
    """
    Delete admin
    
    Requires SUPER_ADMIN role
    """
    try:
        admin = db.query(Admin).filter(Admin.id == admin_id).first()
        if not admin:
            raise HTTPException(status_code=404, detail="Admin not found")
            
        # Don't let user delete themselves
        if admin_id == current_admin.id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")
            
        db.delete(admin)
        db.commit()
        
        logger.info(f"Deleted admin ID: {admin_id}")
        return {"status": "success", "message": "Admin deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting admin: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete admin")
