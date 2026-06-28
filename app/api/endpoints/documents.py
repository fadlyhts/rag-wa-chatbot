"""Document management API endpoints"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from pathlib import Path
import shutil
import logging

from app.database.session import get_db
from app.security.auth import get_current_active_admin, decode_access_token
from app.models.admin import Admin
from app.models.document import Document
from app.schemas.document import (
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentPreviewResponse,
    DocumentUploadResponse,
    DocumentBulkDeleteRequest,
    DocumentBulkDeleteResponse,
    DocumentCategoryResponse,
    DocumentCategoryCreate,
    DocumentCategoryUpdate,
    DivisionCreate,
    DivisionUpdate,
    DivisionResponse,
    DocumentChunkResponse,
    DocumentUpdate,
    DocumentUsageStats
)
from app.services.document_service import document_service
from app.services.file_processor import file_processor

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory - use absolute path for production compatibility
UPLOAD_DIR = Path("uploads").resolve()
UPLOAD_DIR.mkdir(exist_ok=True)


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
        raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")

@router.get("/documents/divisions")
async def list_divisions(
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    List all document divisions
    """
    try:
        from app.models.division import Division
        divisions = db.query(Division).all()
        return [{"id": d.id, "name": d.name} for d in divisions]
    except Exception as e:
        logger.error(f"Error listing divisions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list divisions: {str(e)}")

@router.post("/documents/divisions", response_model=DivisionResponse)
async def create_division(
    division_data: DivisionCreate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Create a new document division
    """
    if current_admin.role.value != "super_admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    try:
        from app.models.division import Division
        division = Division(name=division_data.name)
        db.add(division)
        db.commit()
        db.refresh(division)
        return division
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating division: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create division: {str(e)}")

@router.put("/documents/divisions/{division_id}", response_model=DivisionResponse)
async def update_division(
    division_id: int,
    division_data: DivisionUpdate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Update a document division
    """
    if current_admin.role.value != "super_admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    try:
        from app.models.division import Division
        division = db.query(Division).filter(Division.id == division_id).first()
        if not division:
            raise HTTPException(status_code=404, detail="Division not found")
        
        if division_data.name:
            division.name = division_data.name
            
        db.commit()
        db.refresh(division)
        return division
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating division: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update division: {str(e)}")

@router.delete("/documents/divisions/{division_id}")
async def delete_division(
    division_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Delete a document division
    """
    if current_admin.role.value != "super_admin":
        raise HTTPException(status_code=403, detail="Not authorized")
    try:
        from app.models.division import Division
        division = db.query(Division).filter(Division.id == division_id).first()
        if not division:
            raise HTTPException(status_code=404, detail="Division not found")
            
        db.delete(division)
        db.commit()
        return {"status": "success", "message": "Division deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting division: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete division: {str(e)}")


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

@router.put("/documents/categories/{category_id}", response_model=DocumentCategoryResponse)
async def update_category(
    category_id: int,
    category_data: DocumentCategoryUpdate,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Update a document category
    """
    try:
        category = document_service.update_category(
            db=db,
            category_id=category_id,
            name=category_data.name,
            description=category_data.description
        )
        return category
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating category: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update category: {str(e)}")

@router.delete("/documents/categories/{category_id}")
async def delete_category(
    category_id: int,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Delete a document category
    """
    try:
        document_service.delete_category(db=db, category_id=category_id)
        return {"status": "success", "message": "Category deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting category: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete category: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[int] = None,
    division: Optional[int] = None,
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
        effective_division_id = None
        if not current_admin.is_super_admin:
            effective_division_id = current_admin.division_id
        elif division is not None:
            effective_division_id = division

        result = document_service.list_documents(
            db=db,
            page=page,
            limit=limit,
            search=search,
            status=status,
            category_id=category,
            division_id=effective_division_id
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
    division_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Upload and process a document
    
    Supports: PDF, DOCX, TXT, MD, Images (JPG, PNG, TIFF, BMP, GIF) for OCR
    Max size: 10MB
    
    Processing happens in background:
    1. File is saved
    2. Document record created with status='pending'
    3. Background task processes file (with OCR for images/scanned PDFs)
    4. Status updates to 'completed' or 'failed'
    """
    try:
        # Validate file type - now includes image formats for OCR
        allowed_types = ['.pdf', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_types)}"
            )
        
        # Get file size without reading entire file into memory
        # For SpooledTemporaryFile, seek to end to get size
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
        logger.info(f"Receiving file: {file.filename}, size: {file_size / (1024*1024):.2f}MB")
        
        if file_size > 200 * 1024 * 1024:  # 200MB
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 200MB limit"
            )
        
        # Save file with streaming to handle large files
        timestamp = int(Path(file.filename).stem.encode().hex(), 16) % 1000000
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Stream file to disk in chunks for large files
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
        
        logger.info(f"Saved uploaded file: {safe_filename}")
        
        # Create document record with pending status
        document = file_processor.create_document_record(
            db=db,
            file_path=str(file_path),
            title=title or file.filename,
            content_type=content_type,
            category_id=category_id,
            division_id=current_admin.division_id if current_admin.division_id else division_id,
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
    division_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_admin: Admin = Depends(get_current_active_admin)
):
    """
    Upload multiple documents at once
    
    Supports: PDF, DOCX, TXT, MD, Images (JPG, PNG, TIFF, BMP, GIF) for OCR
    Each file is processed independently in background
    """
    try:
        results = []
        
        for file in files:
            try:
                # Validate and save each file - includes image formats
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in ['.pdf', '.docx', '.txt', '.md', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']:
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
                    division_id=current_admin.division_id if current_admin.division_id else division_id,
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


@router.get("/documents/{document_id}/download")
async def download_document(
    document_id: int,
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Download the original document file.
    
    Supports authentication via:
    - Query parameter `token` (for browser-initiated requests like PDF viewers)
    - Standard Authorization Bearer header
    
    Returns the file with appropriate content-type.
    """
    # Authenticate: token query param is the primary method since
    # vue-pdf-embed loads PDFs via URL and can't send custom headers
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Provide 'token' query parameter."
        )
    
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    admin = db.query(Admin).filter(Admin.username == username, Admin.is_active == True).first()
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.is_active == True
        ).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if not document.file_path:
            raise HTTPException(status_code=404, detail="Document file path not available")
        
        # Resolve file path - try absolute first, then relative to CWD
        file_path = Path(document.file_path)
        if not file_path.exists():
            file_path = Path.cwd() / document.file_path
            if not file_path.exists():
                # Also try relative to the uploads directory
                file_path = UPLOAD_DIR / Path(document.file_path).name
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="Document file not found on disk")
        
        # Determine media type based on file extension
        media_type_map = {
            'pdf': 'application/pdf',
            'txt': 'text/plain',
            'md': 'text/markdown',
            'csv': 'text/csv',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'tiff': 'image/tiff',
            'tif': 'image/tiff',
            'bmp': 'image/bmp',
        }
        
        file_ext = document.file_type or file_path.suffix.lstrip('.')
        media_type = media_type_map.get(file_ext.lower(), 'application/octet-stream')
        
        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=f"{document.title}.{file_ext}" if document.title else file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


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
