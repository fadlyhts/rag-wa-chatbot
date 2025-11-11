"""Application settings model"""

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.dialects.mysql import JSON
from datetime import datetime
from app.database.base import Base


class Settings(Base):
    """Application settings stored in database"""
    
    __tablename__ = "settings"
    
    id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String(100), unique=True, nullable=False, index=True)
    setting_value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Settings(key={self.setting_key})>"
