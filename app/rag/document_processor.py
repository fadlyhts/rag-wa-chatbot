"""Document processing and ingestion"""

from typing import List, Dict, Any, Optional
import os
import hashlib
import logging
from pathlib import Path
import PyPDF2
import docx
import tiktoken

from app.rag.embeddings import embeddings_service
from app.rag.vector_store import vector_store
from app.rag.config import rag_config
from app.rag.ocr_config import ocr_config

logger = logging.getLogger(__name__)

# OCR imports - moved after logger initialization
try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR libraries not available. Install pytesseract, Pillow, and pdf2image for OCR support.")


class DocumentProcessor:
    """Process and ingest documents into vector store"""
    
    def __init__(self):
        self.embeddings = embeddings_service
        self.vector_store = vector_store
        self.chunk_size = rag_config.chunk_size
        self.chunk_overlap = rag_config.chunk_overlap
        
        # Token counter
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Rough estimate
            return len(text) // 4
    
    def read_file(self, file_path: str, progress_callback=None) -> str:
        """
        Read file content based on extension
        
        Args:
            file_path: Path to file
            progress_callback: Optional callback for progress updates during OCR
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._read_pdf(file_path, progress_callback)
            elif extension == '.docx':
                return self._read_docx(file_path)
            elif extension in ['.txt', '.md']:
                return self._read_text(file_path)
            elif extension in ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']:
                return self._read_image_ocr(file_path)
            else:
                logger.warning(f"Unsupported file type: {extension}")
                return ""
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def _read_pdf(self, file_path: Path, progress_callback=None) -> str:
        """Read PDF file with OCR fallback for scanned PDFs"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text + "\n"
        
        # If PDF has very little text, it's likely scanned - use OCR
        if len(text.strip()) < 100 and OCR_AVAILABLE:
            logger.info(f"PDF appears to be scanned, using OCR: {file_path}")
            text = self._read_pdf_ocr(file_path, progress_callback)
        
        return text
    
    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file"""
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    
    def _read_text(self, file_path: Path) -> str:
        """Read text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _read_image_ocr(self, file_path: Path) -> str:
        """
        Read image file using OCR
        
        Args:
            file_path: Path to image file
            
        Returns:
            Extracted text from image
        """
        if not OCR_AVAILABLE:
            raise Exception("OCR libraries not installed. Please install: pip install pytesseract Pillow pdf2image")
        
        try:
            logger.info(f"Performing OCR on image: {file_path}")
            image = Image.open(file_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang='eng+ind')  # English + Indonesian
            
            logger.info(f"OCR extracted {len(text)} characters from image")
            return text
            
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            raise Exception(f"Failed to extract text from image: {str(e)}")
    
    def _read_pdf_ocr(self, file_path: Path, progress_callback=None) -> str:
        """
        Read scanned PDF using optimized OCR with parallel processing
        
        Args:
            file_path: Path to PDF file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Extracted text from scanned PDF
        """
        if not OCR_AVAILABLE:
            raise Exception("OCR libraries not installed. Please install: pip install pytesseract Pillow pdf2image")
        
        import asyncio
        import concurrent.futures
        import gc
        from typing import Tuple
        
        try:
            logger.info(f"Performing OCR on scanned PDF: {file_path}")
            
            # Convert PDF pages to images with configurable DPI
            logger.info(f"Converting PDF pages to images (DPI: {ocr_config.dpi})...")
            images = convert_from_path(file_path, dpi=ocr_config.dpi)
            total_pages = len(images)
            
            logger.info(f"PDF has {total_pages} pages - starting OCR processing (DPI: {ocr_config.dpi})")
            
            # Initialize progress
            if progress_callback:
                progress_callback(0, total_pages, "Starting OCR processing...")
            
            # Process pages sequentially (more reliable than parallel)
            return self._process_ocr_sequential(images, total_pages, progress_callback)
            
        except Exception as e:
            logger.error(f"OCR failed for PDF {file_path}: {e}")
            raise Exception(f"Failed to extract text from scanned PDF: {str(e)}")
    
    def _process_ocr_in_batches(self, images: list, total_pages: int, progress_callback=None) -> str:
        """
        Process OCR in batches with parallel processing and memory management
        
        Args:
            images: List of PIL images
            total_pages: Total number of pages
            progress_callback: Optional callback for progress updates
            
        Returns:
            Extracted text
        """
        import concurrent.futures
        import gc
        from typing import Tuple
        
        # Configure batch settings using OCR config
        batch_size = ocr_config.get_batch_size(total_pages)
        max_workers = ocr_config.get_worker_count(batch_size)
        
        logger.info(f"Processing {total_pages} pages in batches of {batch_size} with {max_workers} parallel workers")
        
        all_text = ""
        processed_pages = 0
        
        # Process in batches
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_images = images[batch_start:batch_end]
            
            logger.info(f"Processing batch: pages {batch_start + 1}-{batch_end} ({len(batch_images)} pages)")
            
            try:
                # Process this batch in parallel
                batch_text = self._process_ocr_batch_parallel(batch_images, batch_start, max_workers)
                all_text += batch_text
                
                processed_pages += len(batch_images)
                
                # Log progress
                logger.info(f"OCR progress: {processed_pages}/{total_pages} pages completed, {len(all_text)} characters extracted so far")
                
                # Update progress via callback
                if progress_callback:
                    progress_callback(processed_pages, total_pages, f"Completed batch {batch_start//batch_size + 1}, {processed_pages}/{total_pages} pages processed")
                
                # Free memory after each batch
                for img in batch_images:
                    try:
                        img.close()
                    except:
                        pass
                del batch_images
                gc.collect()
                
            except Exception as batch_error:
                logger.error(f"Failed to process batch {batch_start + 1}-{batch_end}: {batch_error}")
                # Continue with next batch instead of failing entirely
                processed_pages += len(batch_images)
                continue
        
        logger.info(f"OCR completed: extracted {len(all_text)} characters from {total_pages} pages")
        return all_text
    
    def _process_ocr_batch_parallel(self, images: list, start_page: int, max_workers: int) -> str:
        """
        Process a batch of images in parallel using ThreadPoolExecutor
        
        Args:
            images: List of PIL images for this batch
            start_page: Starting page number (for logging)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Combined text from all images in batch
        """
        import concurrent.futures
        from typing import Tuple
        
        def ocr_single_page(page_data: Tuple[int, any]) -> Tuple[int, str]:
            """OCR a single page and return (page_number, text)"""
            page_idx, image = page_data
            page_number = start_page + page_idx + 1
            
            try:
                logger.debug(f"OCR processing page {page_number}")
                # Use pytesseract's built-in timeout parameter
                text = pytesseract.image_to_string(
                    image, 
                    lang=ocr_config.languages,
                    timeout=ocr_config.page_timeout
                )
                
                return (page_idx, text)
                
            except Exception as e:
                logger.warning(f"Failed to OCR page {page_number}: {str(e)}")
                return (page_idx, "")  # Return empty text for failed pages
        
        # Process pages in parallel
        page_texts = [""] * len(images)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all pages for processing
            future_to_page = {
                executor.submit(ocr_single_page, (i, img)): i 
                for i, img in enumerate(images)
            }
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_page, timeout=ocr_config.batch_timeout):
                try:
                    page_idx, text = future.result(timeout=5)  # 5 second timeout to get result
                    page_texts[page_idx] = text
                except Exception as e:
                    page_idx = future_to_page[future]
                    logger.warning(f"Failed to get OCR result for page {start_page + page_idx + 1}: {e}")
                    page_texts[page_idx] = ""
        
        # Combine all page texts
        combined_text = "\n\n".join(page_texts)
        return combined_text
    
    def _process_ocr_sequential(self, images: list, total_pages: int, progress_callback=None) -> str:
        """
        Process images sequentially (original reliable method)
        
        Args:
            images: List of PIL images
            total_pages: Total pages for logging
            progress_callback: Optional callback for progress updates
            
        Returns:
            Combined text
        """
        text = ""
        for i, image in enumerate(images, 1):
            try:
                logger.info(f"OCR processing page {i}/{total_pages}")
                
                # Simple OCR - no timeout to avoid issues
                page_text = pytesseract.image_to_string(image, lang=ocr_config.languages)
                text += page_text + "\n\n"
                
                # Update progress via callback every 10 pages
                if progress_callback and (i % 10 == 0 or i == total_pages):
                    progress_callback(i, total_pages, f"Processed {i}/{total_pages} pages")
                
                # Free memory after each page
                try:
                    image.close()
                except:
                    pass
                
                # Log progress every 10 pages
                if i % 10 == 0:
                    logger.info(f"OCR progress: {i}/{total_pages} pages completed, {len(text)} characters extracted so far")
                
            except Exception as page_error:
                logger.warning(f"Failed to OCR page {i}, skipping: {page_error}")
                continue
        
        return text
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        # Split by sentences first
        sentences = text.replace('\n', ' ').split('. ')
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + ". "
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= chunk_size:
                current_chunk += sentence
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    # Take last sentences from previous chunk for overlap
                    overlap_text = current_chunk[-overlap * 4:]  # Rough character estimate
                    current_chunk = overlap_text + sentence
                    current_tokens = self.count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
        return chunks
    
    def process_document(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        title: Optional[str] = None,
        content_type: str = "document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document and ingest into vector store
        
        Args:
            file_path: Path to document file
            text: Direct text content (if no file_path)
            title: Document title
            content_type: Type of content (faq, policy, product, etc.)
            metadata: Additional metadata
            
        Returns:
            Dict with processing results
        """
        try:
            # Extract text
            if file_path:
                text = self.read_file(file_path)
                if not title:
                    title = Path(file_path).stem
            elif not text:
                raise ValueError("Either file_path or text must be provided")
            
            if not text.strip():
                raise ValueError("Document is empty")
            
            logger.info(f"Processing document: {title} ({len(text)} chars)")
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = self.embeddings.generate_embeddings_batch(chunks)
            
            # Generate unique IDs for chunks
            doc_id = hashlib.md5(title.encode()).hexdigest()[:8]
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Prepare payloads
            payloads = []
            for i, chunk in enumerate(chunks):
                payload = {
                    'document_id': doc_id,
                    'title': title,
                    'content': chunk,
                    'content_type': content_type,
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                
                # Add custom metadata
                if metadata:
                    payload.update(metadata)
                
                payloads.append(payload)
            
            # Insert into vector store
            logger.info(f"Inserting {len(chunks)} chunks into vector store")
            self.vector_store.insert_documents(
                ids=chunk_ids,
                vectors=embeddings,
                payloads=payloads
            )
            
            result = {
                'document_id': doc_id,
                'title': title,
                'chunks_created': len(chunks),
                'total_tokens': sum(self.count_tokens(chunk) for chunk in chunks),
                'chunk_ids': chunk_ids
            }
            
            logger.info(f"Successfully processed document: {title}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            raise
    
    def process_documents_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch
        
        Args:
            documents: List of document dicts with keys:
                - file_path or text
                - title
                - content_type
                - metadata (optional)
        
        Returns:
            List of processing results
        """
        results = []
        
        for i, doc in enumerate(documents, 1):
            logger.info(f"Processing document {i}/{len(documents)}")
            
            try:
                result = self.process_document(
                    file_path=doc.get('file_path'),
                    text=doc.get('text'),
                    title=doc.get('title'),
                    content_type=doc.get('content_type', 'document'),
                    metadata=doc.get('metadata')
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process document {i}: {e}")
                results.append({
                    'error': str(e),
                    'title': doc.get('title', 'Unknown')
                })
        
        logger.info(f"Batch processing complete: {len(results)} documents")
        return results
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks of a document
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            # This is a simplified version - in production you'd need to
            # query for all chunks with this document_id first
            logger.warning(f"Delete document not fully implemented: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            raise


# Global document processor instance
document_processor = DocumentProcessor()
