"""Conversation model"""

from sqlalchemy import Column, Integer, ForeignKey, DateTime, Boolean, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.base import Base


class Conversation(Base):
    """Conversation model for tracking chat sessions"""
    
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    ended_at = Column(DateTime, nullable=True)
    message_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True, index=True)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_active', 'user_id', 'is_active'),
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id}, active={self.is_active})>"
