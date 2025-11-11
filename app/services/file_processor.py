"""Enhanced file processing service for background tasks"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import logging
import hashlib

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.rag.document_processor import document_processor
from app.rag.vector_store import vector_store
from app.rag.factory import get_embeddings_service
from app.database.session import SessionLocal

logger = logging.getLogger(__name__)


class FileProcessor:
    """Enhanced file processor with background task support"""
    
    def __init__(self):
        self.document_processor = document_processor
        self.vector_store = vector_store
    
    @property
    def embeddings(self):
        """Get embeddings service lazily"""
        return get_embeddings_service()
    
    def create_document_record(
        self,
        db: Session,
        file_path: str,
        title: str,
        content_type: str = "document",
        category_id: Optional[int] = None,
        file_size: Optional[int] = None,
        file_type: Optional[str] = None
    ) -> Document:
        """
        Create initial document record with pending status
        
        Args:
            db: Database session
            file_path: Path to uploaded file
            title: Document title
            content_type: Content type
            category_id: Category ID
            file_size: File size in bytes
            file_type: File extension
            
        Returns:
            Created document
        """
        doc = Document(
            title=title,
            content="",  # Will be filled during processing
            content_type=content_type,
            category_id=category_id,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type,
            embedding_status="pending",
            is_active=True,
            upload_date=datetime.utcnow()
        )
        
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        logger.info(f"Created document record {doc.id} with status 'pending'")
        return doc
    
    async def process_document_async(self, document_id: int):
        """
        Process document in background (async)
        
        Args:
            document_id: Document ID to process
        """
        db = SessionLocal()
        
        try:
            doc = db.query(Document).filter(Document.id == document_id).first()
            
            if not doc:
                logger.error(f"Document {document_id} not found")
                return
            
            # Update status to processing
            doc.embedding_status = "processing"
            db.commit()
            
            logger.info(f"Processing document {document_id}: {doc.title}")
            
            # Extract text from file
            try:
                text = self.document_processor.read_file(doc.file_path)
                doc.content = text
                db.commit()
                logger.info(f"Extracted {len(text)} characters from {doc.title}")
            except Exception as e:
                raise Exception(f"Text extraction failed: {str(e)}")
            
            if not text.strip():
                raise Exception("Document is empty after extraction")
            
            # Chunk text
            chunks = self.document_processor.chunk_text(text)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = await self.embeddings.generate_embeddings_batch_async(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Generate unique IDs for chunks
            doc_hash = hashlib.md5(f"{doc.id}_{doc.title}".encode()).hexdigest()[:8]
            chunk_ids = [f"doc_{doc.id}_{doc_hash}_chunk_{i}" for i in range(len(chunks))]
            
            # Prepare payloads for Qdrant
            payloads = []
            for i, chunk in enumerate(chunks):
                payload = {
                    'document_id': doc.id,
                    'title': doc.title,
                    'content': chunk,
                    'content_type': doc.content_type or "document",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'category_id': doc.category_id
                }
                payloads.append(payload)
            
            # Insert into Qdrant
            self.vector_store.insert_documents(
                ids=chunk_ids,
                vectors=embeddings,
                payloads=payloads
            )
            logger.info(f"Inserted {len(chunks)} chunks into Qdrant")
            
            # Create chunk records in database
            for i, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
                chunk_record = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    chunk_text=chunk,
                    chunk_size=self.document_processor.count_tokens(chunk),
                    qdrant_point_id=chunk_id
                )
                db.add(chunk_record)
            
            # Update document status
            doc.embedding_status = "completed"
            doc.chunks_count = len(chunks)
            doc.processed_at = datetime.utcnow()
            doc.failed_reason = None
            
            db.commit()
            
            logger.info(f"Successfully processed document {document_id}")
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}", exc_info=True)
            
            # Update document with error status
            try:
                doc = db.query(Document).filter(Document.id == document_id).first()
                if doc:
                    doc.embedding_status = "failed"
                    doc.failed_reason = str(e)[:1000]  # Limit error message length
                    db.commit()
            except Exception as db_error:
                logger.error(f"Failed to update error status: {str(db_error)}")
        
        finally:
            db.close()
    
    def reindex_document(self, db: Session, document_id: int) -> bool:
        """
        Re-process and re-embed a document
        
        Args:
            db: Database session
            document_id: Document ID
            
        Returns:
            True if reindexing started successfully
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        
        if not doc:
            return False
        
        # Delete existing chunks from Qdrant
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).all()
        
        point_ids = [chunk.qdrant_point_id for chunk in chunks if chunk.qdrant_point_id]
        
        if point_ids:
            try:
                self.vector_store.delete_documents(point_ids)
                logger.info(f"Deleted {len(point_ids)} old chunks from Qdrant")
            except Exception as e:
                logger.error(f"Failed to delete old chunks: {e}")
        
        # Delete chunk records
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        
        # Reset document status
        doc.embedding_status = "pending"
        doc.chunks_count = 0
        doc.processed_at = None
        doc.failed_reason = None
        
        db.commit()
        
        logger.info(f"Reset document {document_id} for reindexing")
        return True


# Global file processor instance
file_processor = FileProcessor()
