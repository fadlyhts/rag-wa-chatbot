"""Document schemas"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime


class DocumentBase(BaseModel):
    """Base document schema"""
    title: Optional[str] = None
    content_type: Optional[str] = None
    category_id: Optional[int] = None


class DocumentCreate(DocumentBase):
    """Create document schema"""
    content: str


class DocumentUpdate(BaseModel):
    """Update document schema"""
    title: Optional[str] = None
    content_type: Optional[str] = None
    category_id: Optional[int] = None
    doc_metadata: Optional[Dict[str, Any]] = None


class DocumentChunkResponse(BaseModel):
    """Document chunk response"""
    id: int
    chunk_index: int
    chunk_text: str
    chunk_size: Optional[int] = None
    qdrant_point_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentCategoryResponse(BaseModel):
    """Document category response"""
    id: int
    name: str
    description: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentListItem(BaseModel):
    """Document list item (summary)"""
    id: int
    title: Optional[str] = None
    content_type: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    embedding_status: str
    chunks_count: int
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    upload_date: datetime
    processed_at: Optional[datetime] = None
    is_active: bool
    
    class Config:
        from_attributes = True


class DocumentDetailResponse(BaseModel):
    """Document detail response"""
    id: int
    title: Optional[str] = None
    content: str
    content_type: Optional[str] = None
    source_url: Optional[str] = None
    doc_metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_type: Optional[str] = None
    embedding_status: str
    chunks_count: int
    category_id: Optional[int] = None
    category: Optional[DocumentCategoryResponse] = None
    created_at: datetime
    updated_at: datetime
    upload_date: datetime
    processed_at: Optional[datetime] = None
    failed_reason: Optional[str] = None
    is_active: bool
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Paginated document list response"""
    items: List[DocumentListItem]
    total: int
    page: int
    limit: int
    pages: int


class DocumentPreviewResponse(BaseModel):
    """Document content preview"""
    id: int
    title: Optional[str] = None
    preview: str
    full_length: int
    preview_length: int


class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    success: bool
    document_id: int
    title: str
    status: str
    message: str


class DocumentBulkDeleteRequest(BaseModel):
    """Bulk delete request"""
    document_ids: List[int]


class DocumentBulkDeleteResponse(BaseModel):
    """Bulk delete response"""
    success: bool
    deleted_count: int
    failed_ids: List[int]
    message: str


class DocumentCategoryCreate(BaseModel):
    """Create category schema"""
    name: str
    description: Optional[str] = None


class DocumentUsageStats(BaseModel):
    """Document usage statistics"""
    document_id: int
    title: Optional[str] = None
    retrieval_count: int
    last_retrieved_at: Optional[datetime] = None
    avg_relevance_score: Optional[float] = None
