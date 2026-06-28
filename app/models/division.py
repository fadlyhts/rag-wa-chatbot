"""Division model"""

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from app.database.base import Base

class Division(Base):
    """Division model for users and documents"""
    
    __tablename__ = "divisions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    
    # Relationships
    users = relationship("User", back_populates="division")
    documents = relationship("Document", back_populates="division")
    
    def __repr__(self):
        return f"<Division(id={self.id}, name={self.name})>"
