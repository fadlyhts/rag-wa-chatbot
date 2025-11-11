"""Admin user model for authentication"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from datetime import datetime
from app.database.base import Base
import enum


class AdminRole(str, enum.Enum):
    """Admin role enumeration"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    VIEWER = "viewer"


class Admin(Base):
    """Admin user model for panel authentication"""
    
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), nullable=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(AdminRole), default=AdminRole.ADMIN, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    def __repr__(self):
        return f"<Admin(id={self.id}, username={self.username}, role={self.role})>"
    
    @property
    def is_super_admin(self) -> bool:
        """Check if user is super admin"""
        return self.role == AdminRole.SUPER_ADMIN
