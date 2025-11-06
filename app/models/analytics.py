"""Analytics model"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Index
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.base import Base


class Analytics(Base):
    """Analytics model for tracking events"""
    
    __tablename__ = "analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)  # message_received, rag_query, etc.
    event_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="analytics")
    
    __table_args__ = (
        Index('idx_event_created', 'event_type', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Analytics(id={self.id}, event_type={self.event_type})>"
