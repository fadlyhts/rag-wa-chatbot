"""Settings management API endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from app.database.session import get_db
from app.security.auth import get_current_active_admin, require_role
from app.models.admin import Admin
from app.models.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter()


class SettingUpdate(BaseModel):
    """Setting update schema"""
    setting_value: Dict[str, Any]


@router.get("/settings")
async def get_all_settings(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get all application settings
    
    Returns all settings as key-value pairs
    """
    try:
        settings = db.query(Settings).all()
        
        settings_dict = {}
        for setting in settings:
            settings_dict[setting.setting_key] = setting.setting_value
        
        return {
            "success": True,
            "settings": settings_dict
        }
        
    except Exception as e:
        logger.error(f"Error getting settings: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


@router.get("/settings/{key}")
async def get_setting(
    key: str,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get a specific setting by key
    """
    try:
        setting = db.query(Settings).filter(Settings.setting_key == key).first()
        
        if not setting:
            raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
        
        return {
            "success": True,
            "key": setting.setting_key,
            "value": setting.setting_value,
            "updated_at": setting.updated_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting setting {key}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get setting: {str(e)}")


@router.put("/settings/{key}")
async def update_setting(
    key: str,
    update_data: SettingUpdate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(require_role("super_admin", "admin"))
):
    """
    Update a specific setting
    
    Requires admin or super_admin role
    """
    try:
        setting = db.query(Settings).filter(Settings.setting_key == key).first()
        
        if not setting:
            # Create new setting if it doesn't exist
            setting = Settings(
                setting_key=key,
                setting_value=update_data.setting_value
            )
            db.add(setting)
        else:
            # Update existing setting
            setting.setting_value = update_data.setting_value
        
        db.commit()
        db.refresh(setting)
        
        logger.info(f"Setting '{key}' updated by {current_admin.username}")
        
        return {
            "success": True,
            "key": setting.setting_key,
            "value": setting.setting_value,
            "message": f"Setting '{key}' updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating setting {key}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update setting: {str(e)}")


@router.put("/settings")
async def update_multiple_settings(
    settings_data: Dict[str, Dict[str, Any]],
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(require_role("super_admin", "admin"))
):
    """
    Update multiple settings at once
    
    Request body format:
    {
        "rag_config": {"model": "gpt-4", "temperature": 0.7, ...},
        "rate_limiting": {"messages_per_minute": 10, ...}
    }
    """
    try:
        updated_count = 0
        
        for key, value in settings_data.items():
            setting = db.query(Settings).filter(Settings.setting_key == key).first()
            
            if not setting:
                setting = Settings(
                    setting_key=key,
                    setting_value=value
                )
                db.add(setting)
            else:
                setting.setting_value = value
            
            updated_count += 1
        
        db.commit()
        
        logger.info(f"Updated {updated_count} settings by {current_admin.username}")
        
        return {
            "success": True,
            "updated_count": updated_count,
            "message": f"Updated {updated_count} settings successfully"
        }
        
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


@router.delete("/settings/{key}")
async def delete_setting(
    key: str,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(require_role("super_admin"))
):
    """
    Delete a setting
    
    Requires super_admin role
    """
    try:
        setting = db.query(Settings).filter(Settings.setting_key == key).first()
        
        if not setting:
            raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
        
        db.delete(setting)
        db.commit()
        
        logger.warning(f"Setting '{key}' deleted by {current_admin.username}")
        
        return {
            "success": True,
            "message": f"Setting '{key}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting setting {key}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete setting: {str(e)}")


@router.get("/settings/rag/config")
async def get_rag_config(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get RAG configuration
    
    Quick access to RAG-specific settings
    """
    try:
        setting = db.query(Settings).filter(Settings.setting_key == "rag_config").first()
        
        if not setting:
            # Return default RAG config
            return {
                "success": True,
                "config": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "top_k": 5,
                    "min_score": 0.7
                },
                "note": "Using default configuration"
            }
        
        return {
            "success": True,
            "config": setting.setting_value
        }
        
    except Exception as e:
        logger.error(f"Error getting RAG config: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get RAG config: {str(e)}")
