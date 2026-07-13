"""Enhanced file processing service for background tasks"""

from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import uuid
import os
import asyncio

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.rag.document_processor import document_processor
from app.rag.vector_store import vector_store
from app.rag.factory import get_embeddings_service
from app.database.session import SessionLocal
from app.rag.config import rag_config

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
        division_id: Optional[int] = None,
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
            division_id=division_id,
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
            
            # 1. Delete existing chunks in Qdrant
            existing_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).all()
            if existing_chunks:
                try:
                    qdrant_ids = [chunk.qdrant_point_id for chunk in existing_chunks if chunk.qdrant_point_id]
                    if qdrant_ids:
                        self.vector_store.delete_documents(qdrant_ids)
                        logger.info(f"Deleted {len(qdrant_ids)} existing points in Qdrant")
                except Exception as e:
                    logger.warning(f"Failed to delete existing points in Qdrant: {e}")
                    
                # 2. Delete existing chunks in DB
                db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
                
            db.commit()
            
            logger.info(f"Processing document {document_id}: {doc.title}")
            
            # Resolve file path
            try:
                file_path = Path(doc.file_path)
                if not file_path.exists():
                    file_path = Path.cwd() / doc.file_path
                    if not file_path.exists():
                        raise FileNotFoundError(f"File not found: {doc.file_path}")
            except Exception as e:
                raise Exception(f"File resolution failed: {str(e)}")
            
            # Extract text with page-level granularity
            # PENTING: Jalankan di thread terpisah agar TIDAK memblokir event loop FastAPI
            # Tanpa ini, OCR yang CPU-intensive akan memblokir healthcheck dan menyebabkan container di-kill
            try:
                from app.rag.config import rag_config
                
                # Switch between Docling and original pytesseract parser
                if getattr(rag_config, 'use_docling', False):
                    logger.info(f"Using DoclingProcessor for document {document_id}")
                    from app.rag.docling_processor import DoclingProcessor
                    
                    # Instantiate directly (removes singleton behavior)
                    docling_proc = DoclingProcessor()
                    page_chunks = await asyncio.to_thread(
                        docling_proc.process_document, str(file_path)
                    )
                    
                    # Mock full_text so doc.content isn't empty
                    for c in page_chunks:
                        if "text" in c and isinstance(c["text"], str):
                            c["text"] = c["text"].replace("PTPNI", "PTPN1").replace("PTPN I", "PTPN 1")
                            import re
                            c["text"] = re.sub(r'([a-zA-Z])O(\d)', r'\g<1>0\2', c["text"], flags=re.IGNORECASE)
                    
                    full_text = "\n".join(c["text"] for c in page_chunks)
                    doc.content = full_text
                    
                    # Also run the standard cover page extractor to guarantee metadata like signatures is caught
                    logger.info("Extracting cover page metadata for Docling chunks...")
                    cover_pages = await asyncio.to_thread(
                        self.document_processor.read_file_pages, str(file_path), max_pages=2
                    )
                    cover_info = self.document_processor._extract_cover_page_info(cover_pages)
                    
                    try:
                        metadata = self.document_processor.extract_document_metadata(full_text[:5000])
                        if metadata:
                            doc.doc_metadata = metadata
                    except:
                        pass
                        
                    if cover_info:
                        judul = doc.doc_metadata.get('Judul', doc.title) if doc.doc_metadata else doc.title
                        no_dok = doc.doc_metadata.get('No. Dokumen', '') if doc.doc_metadata else ''
                        
                        qa_text = f"Informasi Pengesahan Dokumen SOP {judul}:\n"
                        if no_dok:
                            qa_text += f"Nomor Dokumen: {no_dok}\n"
                        if cover_info.get('disusun_oleh'):
                            qa_text += f"Pertanyaan: Siapa yang menyusun (pembuat) dokumen ini?\nJawaban: Dokumen SOP ini disusun oleh {cover_info['disusun_oleh']}.\n"
                        if cover_info.get('ditinjau_oleh'):
                            qa_text += f"Pertanyaan: Siapa yang meninjau (mereview) dokumen ini?\nJawaban: Dokumen SOP ini ditinjau oleh {cover_info['ditinjau_oleh']}.\n"
                        if cover_info.get('disetujui_oleh'):
                            qa_text += f"Pertanyaan: Siapa yang menyetujui (mengakui) dokumen ini?\nJawaban: Dokumen SOP ini disetujui oleh {cover_info['disetujui_oleh']}.\n"
                        
                        synthetic_chunk = {
                            "text": qa_text,
                            "page_number": 2,
                            "page_numbers": [2],
                            "heading": "Pengesahan Dokumen (Metadata)",
                            "parent_heading": None,
                        }
                        page_chunks.insert(0, synthetic_chunk)
                    
                    doc.ocr_progress_current = len(page_chunks)
                    doc.ocr_progress_total = len(page_chunks)
                    db.commit()
                else:
                    logger.info(f"Using standard DocumentProcessor for document {document_id}")
                    # Use PDFPlumber + PyTesseract for extraction
                    pages = await asyncio.to_thread(
                        self.document_processor.read_file_pages, str(file_path)
                    )
                    
                    # Save combined text to DB for preview/search
                    full_text = "\n".join(p["text"] for p in pages)
                    doc.content = full_text
                    
                    # Extract structured metadata (Judul, No Dokumen, dll) from the first few pages
                    try:
                        metadata = self.document_processor.extract_document_metadata(full_text[:5000])
                        if metadata:
                            doc.doc_metadata = metadata
                            logger.info(f"Extracted document metadata: {metadata}")
                    except Exception as e:
                        logger.warning(f"Failed to extract document metadata: {e}")
                        
                    doc.ocr_progress_current = len(pages)
                    doc.ocr_progress_total = len(pages)
                    db.commit()
                    
                    # Falls back to sentence-based chunking for non-structured docs
                    for p in pages:
                        if "text" in p and isinstance(p["text"], str):
                            p["text"] = p["text"].replace("PTPNI", "PTPN1").replace("PTPN I", "PTPN 1")
                            import re
                            p["text"] = re.sub(r'([a-zA-Z])O(\d)', r'\g<1>0\2', p["text"], flags=re.IGNORECASE)
                            
                    page_chunks = self.document_processor.chunk_text_semantic_with_pages(pages)
                    
                    cover_info = self.document_processor._extract_cover_page_info(pages[:2])
                    
                    if cover_info:
                        judul = doc.doc_metadata.get('Judul', doc.title) if doc.doc_metadata else doc.title
                        no_dok = doc.doc_metadata.get('No. Dokumen', '') if doc.doc_metadata else ''
                        
                        qa_text = f"Informasi Pengesahan Dokumen SOP {judul}:\n"
                        if no_dok:
                            qa_text += f"Nomor Dokumen: {no_dok}\n"
                        if cover_info.get('disusun_oleh'):
                            qa_text += f"Pertanyaan: Siapa yang menyusun (pembuat) dokumen ini?\nJawaban: Dokumen SOP ini disusun oleh {cover_info['disusun_oleh']}.\n"
                        if cover_info.get('ditinjau_oleh'):
                            qa_text += f"Pertanyaan: Siapa yang meninjau (mereview) dokumen ini?\nJawaban: Dokumen SOP ini ditinjau oleh {cover_info['ditinjau_oleh']}.\n"
                        if cover_info.get('disetujui_oleh'):
                            qa_text += f"Pertanyaan: Siapa yang menyetujui (mengakui) dokumen ini?\nJawaban: Dokumen SOP ini disetujui oleh {cover_info['disetujui_oleh']}.\n"
                        
                        synthetic_chunk = {
                            "text": qa_text,
                            # Label this chunk as page 2 so the LLM cites it properly
                            "page_number": 2,
                            "page_numbers": [2],
                            "heading": "Pengesahan Dokumen (Metadata)",
                            "parent_heading": None,
                        }
                        page_chunks.insert(0, synthetic_chunk)
            except Exception as e:
                raise Exception(f"Text extraction failed: {str(e)}")

            # Prepend document title to chunk text to improve vector search relevance
            chunk_texts = []
            valid_page_chunks = []
            
            for c in page_chunks:
                c_text = c.get('text', '').strip()
                text_len = len(c_text)
                text_lower = c_text.lower()
                
                # Keywords that indicate a chunk is important even if it's short
                important_kws = ['tujuan', 'ruang lingkup', 'definisi', 'referensi', 'ketentuan', 
                               'pengertian', 'prosedur', 'tanggung jawab', 'kebijakan', 'lampiran']
                is_important = any(kw in text_lower for kw in important_kws)
                
                # Skip tiny noise chunks and repeated footers/headers
                if text_len < 100 and not is_important:
                    continue
                    
                # Extra check: skip if the chunk is just the document title (footer)
                judul = doc.doc_metadata.get('Judul') if doc.doc_metadata else None
                if judul and text_lower == judul.lower():
                    continue
                    
                valid_page_chunks.append(c)
                kategori = doc.doc_metadata.get('Jenis Dokumen') if doc.doc_metadata else None
                
                header_parts = []
                if judul:
                    header_parts.append(f"Judul Dokumen: {judul}")
                else:
                    header_parts.append(f"File Dokumen: {doc.title}")
                    
                if kategori:
                    header_parts.append(f"Kategori: {kategori}")
                
                if c.get('heading'):
                    header_parts.append(f"Bagian: {c['heading']}")
                elif c.get('parent_heading'):
                    header_parts.append(f"Bagian: {c['parent_heading']}")
                    
                header = "\n".join(header_parts)
                chunk_texts.append(f"{header}\n\n{c['text']}")
            
            page_chunks = valid_page_chunks
            logger.info(f"Filtered tiny chunks. Remaining: {len(page_chunks)} semantic chunks")
            
            # Generate embeddings
            logger.info(f"Starting embedding generation for {len(chunk_texts)} chunks...")
            embeddings = await self.embeddings.generate_embeddings_batch_async(chunk_texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            
            # Filter out chunks with failed (None) embeddings
            valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
            if len(valid_indices) < len(embeddings):
                logger.warning(f"Skipping {len(embeddings) - len(valid_indices)} chunks with failed embeddings")
                page_chunks = [page_chunks[i] for i in valid_indices]
                embeddings = [embeddings[i] for i in valid_indices]
                chunk_texts = [chunk_texts[i] for i in valid_indices]
            
            if not embeddings:
                raise Exception("All embeddings failed to generate")
            
            # Prepare Qdrant payloads with page metadata
            chunk_ids = [str(uuid.uuid4()) for _ in range(len(page_chunks))]
            
            # File name without extension for display
            file_name = Path(doc.file_path).stem if doc.file_path else doc.title
            
            payloads = []
            for i, chunk_info in enumerate(page_chunks):
                payload = {
                    'document_id': doc.id,
                    'title': doc.title,
                    'file_name': file_name,
                    'content': chunk_texts[i],
                    'content_type': doc.content_type or "document",
                    'chunk_index': i,
                    'total_chunks': len(page_chunks),
                    'page_number': chunk_info["page_number"],
                    'page_numbers': chunk_info["page_numbers"],
                    'category_id': doc.category_id,
                    'division_id': doc.division_id,
                    # Heading metadata from semantic chunking
                    'heading': chunk_info.get("heading"),
                    'parent_heading': chunk_info.get("parent_heading"),
                    # Document level metadata (Judul, No Dokumen, dll)
                    'doc_metadata': doc.doc_metadata or {},
                }
                payloads.append(payload)
            
            # Generate Sparse Embeddings for Hybrid Search if enabled
            sparse_vectors = None
            if getattr(rag_config, 'RAG_HYBRID_SEARCH', False):
                try:
                    from app.rag.embeddings_sparse import SparseEmbeddings
                    sparse_embedder = SparseEmbeddings(model_name=rag_config.RAG_SPARSE_MODEL_NAME)
                    logger.info(f"Generating sparse embeddings for {len(chunk_texts)} chunks")
                    sparse_vectors = sparse_embedder.generate_sparse_embeddings_batch(chunk_texts)
                except Exception as e:
                    logger.error(f"Failed to generate sparse embeddings: {e}")
            
            # Insert into Qdrant
            self.vector_store.insert_documents(
                ids=chunk_ids,
                vectors=embeddings,
                payloads=payloads,
                sparse_vectors=sparse_vectors
            )
            logger.info(f"Inserted {len(page_chunks)} chunks into Qdrant")
            
            # Create chunk records in database
            for i, (chunk_info, chunk_id) in enumerate(zip(page_chunks, chunk_ids)):
                chunk_record = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    chunk_text=chunk_texts[i],
                    chunk_size=self.document_processor.count_tokens(chunk_texts[i]),
                    qdrant_point_id=chunk_id
                )
                db.add(chunk_record)
            
            # Update document status
            doc.embedding_status = "completed"
            doc.chunks_count = len(page_chunks)
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
