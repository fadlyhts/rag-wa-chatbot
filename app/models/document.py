"""Document model"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Index, ForeignKey
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import relationship
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
    
    # New columns for admin panel
    category_id = Column(Integer, ForeignKey("document_categories.id", ondelete="SET NULL"), nullable=True, index=True)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=True)  # Size in bytes
    file_type = Column(String(50), nullable=True)  # pdf, docx, txt, md
    chunks_count = Column(Integer, default=0, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    failed_reason = Column(Text, nullable=True)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    category = relationship("DocumentCategory", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_active_status', 'is_active', 'embedding_status'),
        Index('idx_category_status', 'category_id', 'embedding_status'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title}, status={self.embedding_status})>"
