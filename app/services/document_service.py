"""Document management service"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, or_, desc
from datetime import datetime
import logging

from app.models.document import Document
from app.models.document_category import DocumentCategory
from app.models.document_chunk import DocumentChunk
from app.models.analytics import Analytics
from app.schemas.document import (
    DocumentListResponse,
    DocumentListItem,
    DocumentDetailResponse,
    DocumentPreviewResponse,
    DocumentUsageStats
)
from app.rag.vector_store import vector_store

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document management operations"""
    
    def list_documents(
        self,
        db: Session,
        page: int = 1,
        limit: int = 20,
        search: Optional[str] = None,
        status: Optional[str] = None,
        category_id: Optional[int] = None
    ) -> DocumentListResponse:
        """
        List documents with pagination and filters
        
        Args:
            db: Database session
            page: Page number (1-indexed)
            limit: Items per page
            search: Search query for title/content
            status: Filter by embedding_status
            category_id: Filter by category
            
        Returns:
            Paginated document list
        """
        query = db.query(Document).options(joinedload(Document.category))
        
        # Apply filters
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    Document.title.like(search_pattern),
                    Document.content.like(search_pattern)
                )
            )
        
        if status:
            query = query.filter(Document.embedding_status == status)
        
        if category_id:
            query = query.filter(Document.category_id == category_id)
        
        # Count total
        total = query.count()
        
        # Calculate pagination
        pages = (total + limit - 1) // limit
        offset = (page - 1) * limit
        
        # Get page of results
        documents = query.order_by(desc(Document.upload_date)).offset(offset).limit(limit).all()
        
        # Convert to response items
        items = []
        for doc in documents:
            item = DocumentListItem(
                id=doc.id,
                title=doc.title,
                content_type=doc.content_type,
                file_type=doc.file_type,
                file_size=doc.file_size,
                embedding_status=doc.embedding_status,
                chunks_count=doc.chunks_count,
                category_id=doc.category_id,
                category_name=doc.category.name if doc.category else None,
                upload_date=doc.upload_date,
                processed_at=doc.processed_at,
                is_active=doc.is_active
            )
            items.append(item)
        
        return DocumentListResponse(
            items=items,
            total=total,
            page=page,
            limit=limit,
            pages=pages
        )
    
    def get_document(self, db: Session, document_id: int) -> Optional[DocumentDetailResponse]:
        """
        Get document details
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            Document details or None
        """
        doc = db.query(Document).options(joinedload(Document.category)).filter(
            Document.id == document_id
        ).first()
        
        if not doc:
            return None
        
        return DocumentDetailResponse.from_orm(doc)
    
    def get_document_preview(
        self,
        db: Session,
        document_id: int,
        preview_length: int = 1000
    ) -> Optional[DocumentPreviewResponse]:
        """
        Get document content preview
        
        Args:
            db: Database session
            document_id: Document ID
            preview_length: Number of characters to preview
            
        Returns:
            Document preview or None
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            return None
        
        content = doc.content or ""
        preview = content[:preview_length]
        
        return DocumentPreviewResponse(
            id=doc.id,
            title=doc.title,
            preview=preview,
            full_length=len(content),
            preview_length=len(preview)
        )
    
    def get_document_chunks(
        self,
        db: Session,
        document_id: int
    ) -> List[DocumentChunk]:
        """
        Get all chunks for a document
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()
        
        return chunks
    
    def update_document(
        self,
        db: Session,
        document_id: int,
        title: Optional[str] = None,
        content_type: Optional[str] = None,
        category_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """
        Update document metadata
        
        Args:
            db: Database session
            document_id: Document ID
            title: New title
            content_type: New content type
            category_id: New category
            metadata: Additional metadata
            
        Returns:
            Updated document or None
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            return None
        
        if title is not None:
            doc.title = title
        
        if content_type is not None:
            doc.content_type = content_type
        
        if category_id is not None:
            doc.category_id = category_id
        
        if metadata is not None:
            doc.doc_metadata = metadata
        
        doc.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(doc)
        
        logger.info(f"Updated document {document_id}")
        return doc
    
    def delete_document(self, db: Session, document_id: int) -> bool:
        """
        Soft delete document and remove from Qdrant
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            True if successful
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            return False
        
        # Get all chunk point IDs
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).all()
        
        point_ids = [chunk.qdrant_point_id for chunk in chunks if chunk.qdrant_point_id]
        
        # Delete from Qdrant
        if point_ids:
            try:
                vector_store.delete_documents(point_ids)
                logger.info(f"Deleted {len(point_ids)} points from Qdrant for document {document_id}")
            except Exception as e:
                logger.error(f"Failed to delete from Qdrant: {e}")
        
        # Soft delete document
        doc.is_active = False
        doc.updated_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Soft deleted document {document_id}")
        return True
    
    def bulk_delete_documents(self, db: Session, document_ids: List[int]) -> Dict[str, Any]:
        """
        Delete multiple documents
        
        Args:
            db: Database session
            document_ids: List of document IDs
            
        Returns:
            Dict with deletion results
        """
        deleted_count = 0
        failed_ids = []
        
        for doc_id in document_ids:
            try:
                if self.delete_document(db, doc_id):
                    deleted_count += 1
                else:
                    failed_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to delete document {doc_id}: {e}")
                failed_ids.append(doc_id)
        
        return {
            "deleted_count": deleted_count,
            "failed_ids": failed_ids
        }
    
    def get_usage_stats(self, db: Session, document_id: int) -> DocumentUsageStats:
        """
        Get usage statistics for a document
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            Usage statistics
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            return DocumentUsageStats(
                document_id=document_id,
                title=None,
                retrieval_count=0,
                last_retrieved_at=None,
                avg_relevance_score=None
            )
        
        # Count retrieval events in analytics
        # This is a placeholder - in production, track document retrievals in analytics
        retrieval_count = 0
        last_retrieved_at = None
        
        return DocumentUsageStats(
            document_id=document_id,
            title=doc.title,
            retrieval_count=retrieval_count,
            last_retrieved_at=last_retrieved_at,
            avg_relevance_score=None
        )
    
    def get_categories(self, db: Session) -> List[DocumentCategory]:
        """
        Get all document categories
        
        Args:
            db: Database session
            
        Returns:
            List of categories
        """
        return db.query(DocumentCategory).order_by(DocumentCategory.name).all()
    
    def create_category(
        self,
        db: Session,
        name: str,
        description: Optional[str] = None
    ) -> DocumentCategory:
        """
        Create a new document category
        
        Args:
            db: Database session
            name: Category name
            description: Category description
            
        Returns:
            Created category
        """
        category = DocumentCategory(
            name=name,
            description=description
        )
        
        db.add(category)
        db.commit()
        db.refresh(category)
        
        logger.info(f"Created category: {name}")
        return category


# Global document service instance
document_service = DocumentService()
