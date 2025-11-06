"""Document model"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index
from sqlalchemy.dialects.mysql import JSON
from datetime import datetime
from app.database.base import Base


class Document(Base):
    """Document model for knowledge base"""
    
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=True)  # policy, faq, product, etc.
    source_url = Column(Text, nullable=True)
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True, index=True)
    embedding_status = Column(String(20), default="pending", index=True)  # pending, processing, completed, failed
    
    __table_args__ = (
        Index('idx_active_status', 'is_active', 'embedding_status'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title}, status={self.embedding_status})>"
