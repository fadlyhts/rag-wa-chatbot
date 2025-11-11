"""Document category model"""

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database.base import Base


class DocumentCategory(Base):
    """Document category for organizing knowledge base"""
    
    __tablename__ = "document_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    documents = relationship("Document", back_populates="category")
    
    def __repr__(self):
        return f"<DocumentCategory(id={self.id}, name={self.name})>"
