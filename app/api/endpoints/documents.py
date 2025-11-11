"""Document management API endpoints"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from pathlib import Path
import shutil
import logging

from app.database.session import get_db
from app.security.auth import get_current_active_admin
from app.models.admin import Admin
from app.schemas.document import (
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentPreviewResponse,
    DocumentUploadResponse,
    DocumentBulkDeleteRequest,
    DocumentBulkDeleteResponse,
    DocumentCategoryResponse,
    DocumentCategoryCreate,
    DocumentChunkResponse,
    DocumentUpdate,
    DocumentUsageStats
)
from app.services.document_service import document_service
from app.services.file_processor import file_processor

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[int] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    List documents with pagination and filters
    
    - **page**: Page number (1-indexed)
    - **limit**: Items per page (1-100)
    - **search**: Search in title and content
    - **status**: Filter by embedding status (pending/processing/completed/failed)
    - **category**: Filter by category ID
    """
    try:
        result = document_service.list_documents(
            db=db,
            page=page,
            limit=limit,
            search=search,
            status=status,
            category_id=category
        )
        return result
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    content_type: str = "document",
    category_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Upload and process a document
    
    Supports: PDF, DOCX, TXT, MD
    Max size: 10MB
    
    Processing happens in background:
    1. File is saved
    2. Document record created with status='pending'
    3. Background task processes file
    4. Status updates to 'completed' or 'failed'
    """
    try:
        # Validate file type
        allowed_types = ['.pdf', '.docx', '.txt', '.md']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_types)}"
            )
        
        # Validate file size (10MB max)
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 10MB limit"
            )
        
        # Save file
        timestamp = int(Path(file.filename).stem.encode().hex(), 16) % 1000000
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file: {safe_filename}")
        
        # Create document record with pending status
        document = file_processor.create_document_record(
            db=db,
            file_path=str(file_path),
            title=title or file.filename,
            content_type=content_type,
            category_id=category_id,
            file_size=file_size,
            file_type=file_ext.replace('.', '')
        )
        
        # Queue background processing
        background_tasks.add_task(file_processor.process_document_async, document.id)
        
        logger.info(f"Queued document {document.id} for processing")
        
        return DocumentUploadResponse(
            success=True,
            document_id=document.id,
            title=document.title,
            status="pending",
            message="Document uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/documents/bulk-upload")
async def bulk_upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    content_type: str = "document",
    category_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Upload multiple documents at once
    
    Each file is processed independently in background
    """
    try:
        results = []
        
        for file in files:
            try:
                # Validate and save each file
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in ['.pdf', '.docx', '.txt', '.md']:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": f"Unsupported file type: {file_ext}"
                    })
                    continue
                
                # Save file
                timestamp = int(Path(file.filename).stem.encode().hex(), 16) % 1000000
                safe_filename = f"{timestamp}_{file.filename}"
                file_path = UPLOAD_DIR / safe_filename
                
                file.file.seek(0, 2)
                file_size = file.file.tell()
                file.file.seek(0)
                
                if file_size > 10 * 1024 * 1024:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "File size exceeds 10MB"
                    })
                    continue
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Create document record
                document = file_processor.create_document_record(
                    db=db,
                    file_path=str(file_path),
                    title=file.filename,
                    content_type=content_type,
                    category_id=category_id,
                    file_size=file_size,
                    file_type=file_ext.replace('.', '')
                )
                
                # Queue processing
                background_tasks.add_task(file_processor.process_document_async, document.id)
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "document_id": document.id,
                    "status": "pending"
                })
                
            except Exception as e:
                logger.error(f"Error uploading {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_files": len(files),
            "uploaded": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Bulk upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get document details including metadata and status
    """
    try:
        document = document_service.get_document(db, document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")


@router.get("/documents/{document_id}/preview", response_model=DocumentPreviewResponse)
async def get_document_preview(
    document_id: int,
    length: int = Query(1000, ge=100, le=5000),
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get document content preview
    
    - **length**: Number of characters to preview (100-5000)
    """
    try:
        preview = document_service.get_document_preview(db, document_id, length)
        
        if not preview:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return preview
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview for {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get preview: {str(e)}")


@router.get("/documents/{document_id}/chunks", response_model=List[DocumentChunkResponse])
async def get_document_chunks(
    document_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get all chunks for a document
    
    Shows how the document was split for embedding
    """
    try:
        chunks = document_service.get_document_chunks(db, document_id)
        return chunks
        
    except Exception as e:
        logger.error(f"Error getting chunks for {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")


@router.put("/documents/{document_id}")
async def update_document(
    document_id: int,
    update_data: DocumentUpdate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Update document metadata
    
    Can update: title, content_type, category_id, metadata
    """
    try:
        document = document_service.update_document(
            db=db,
            document_id=document_id,
            title=update_data.title,
            content_type=update_data.content_type,
            category_id=update_data.category_id,
            metadata=update_data.doc_metadata
        )
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document_id": document.id,
            "message": "Document updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Delete document (soft delete) and remove from Qdrant
    """
    try:
        success = document_service.delete_document(db, document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document_id": document_id,
            "message": "Document deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Re-process and re-embed a document
    
    Useful when:
    - Document processing failed
    - Want to use updated chunking/embedding settings
    - Qdrant data was lost
    """
    try:
        success = file_processor.reindex_document(db, document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Queue reprocessing
        background_tasks.add_task(file_processor.process_document_async, document_id)
        
        return {
            "success": True,
            "document_id": document_id,
            "status": "pending",
            "message": "Document queued for reindexing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reindexing document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


@router.post("/documents/bulk-delete", response_model=DocumentBulkDeleteResponse)
async def bulk_delete_documents(
    request: DocumentBulkDeleteRequest,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Delete multiple documents at once
    """
    try:
        result = document_service.bulk_delete_documents(db, request.document_ids)
        
        return DocumentBulkDeleteResponse(
            success=True,
            deleted_count=result["deleted_count"],
            failed_ids=result["failed_ids"],
            message=f"Deleted {result['deleted_count']} documents"
        )
        
    except Exception as e:
        logger.error(f"Bulk delete error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Bulk delete failed: {str(e)}")


@router.get("/documents/categories", response_model=List[DocumentCategoryResponse])
async def list_categories(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    List all document categories
    """
    try:
        categories = document_service.get_categories(db)
        return categories
    except Exception as e:
        logger.error(f"Error listing categories: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")


@router.post("/documents/categories", response_model=DocumentCategoryResponse)
async def create_category(
    category_data: DocumentCategoryCreate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Create a new document category
    """
    try:
        category = document_service.create_category(
            db=db,
            name=category_data.name,
            description=category_data.description
        )
        return category
    except Exception as e:
        logger.error(f"Error creating category: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create category: {str(e)}")


@router.get("/documents/{document_id}/usage-stats", response_model=DocumentUsageStats)
async def get_document_usage_stats(
    document_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Get usage statistics for a document
    
    Shows how often the document has been retrieved in RAG queries
    """
    try:
        stats = document_service.get_usage_stats(db, document_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting usage stats for {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get usage stats: {str(e)}")
