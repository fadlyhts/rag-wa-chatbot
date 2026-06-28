"""User model"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.base import Base


class User(Base):
    """User model for WhatsApp users"""
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(20), unique=True, nullable=False, index=True)
    whatsapp_name = Column(String(255), nullable=True)
    profile_pic_url = Column(String(500), nullable=True)
    language = Column(String(10), default="en")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_blocked = Column(Boolean, default=False)
    division_id = Column(Integer, ForeignKey("divisions.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Relationships
    division = relationship("Division", back_populates="users")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="user", cascade="all, delete-orphan")
    analytics = relationship("Analytics", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, phone={self.phone_number})>"
