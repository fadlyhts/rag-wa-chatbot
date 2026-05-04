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
    
    def read_file(self, file_path: str) -> str:
        """
        Read file content based on extension
        
        Args:
            file_path: Path to file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._read_pdf(file_path)
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
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file with OCR fallback for scanned PDFs"""
        pages = self._read_pdf_pages(file_path)
        return "\n".join(p["text"] for p in pages)

    def _read_pdf_pages(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read PDF file and return text per page.
        
        Returns:
            List of dicts: [{"page_number": 1, "text": "..."}, ...]
        """
        pages: List[Dict[str, Any]] = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text() or ""
                pages.append({"page_number": i, "text": page_text})
        
        # If PDF has very little text overall, it's likely scanned - use OCR
        total_text = "".join(p["text"] for p in pages)
        if len(total_text.strip()) < 100 and OCR_AVAILABLE:
            logger.info(f"PDF appears to be scanned, using OCR: {file_path}")
            pages = self._read_pdf_ocr_pages(file_path)
        
        return pages
    
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
    
    def _read_pdf_ocr(self, file_path: Path) -> str:
        """Read scanned PDF using OCR (returns single string)"""
        pages = self._read_pdf_ocr_pages(file_path)
        return "\n\n".join(p["text"] for p in pages)

    def _read_pdf_ocr_pages(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read scanned PDF using OCR, returning text per page.
        
        Returns:
            List of dicts: [{"page_number": 1, "text": "..."}, ...]
        """
        if not OCR_AVAILABLE:
            raise Exception("OCR libraries not installed. Please install: pip install pytesseract Pillow pdf2image")
        
        try:
            logger.info(f"Performing OCR on scanned PDF: {file_path}")
            
            images = convert_from_path(file_path, dpi=150)
            total_pages = len(images)
            
            logger.info(f"PDF has {total_pages} pages - starting OCR processing (DPI: 150)")
            
            pages: List[Dict[str, Any]] = []
            for i, image in enumerate(images, 1):
                try:
                    logger.info(f"OCR processing page {i}/{total_pages}")
                    page_text = pytesseract.image_to_string(image, lang='eng+ind')
                    pages.append({"page_number": i, "text": page_text})
                    
                    image.close()
                    del image
                    
                    if i % 10 == 0:
                        logger.info(f"OCR progress: {i}/{total_pages} pages completed")
                        
                except Exception as page_error:
                    logger.warning(f"Failed to OCR page {i}, skipping: {page_error}")
                    pages.append({"page_number": i, "text": ""})
                    continue
            
            total_chars = sum(len(p["text"]) for p in pages)
            logger.info(f"OCR completed: extracted {total_chars} characters from {total_pages} pages")
            return pages
            
        except Exception as e:
            logger.error(f"OCR failed for PDF {file_path}: {e}")
            raise Exception(f"Failed to extract text from scanned PDF: {str(e)}")
    
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

    def chunk_text_with_pages(
        self,
        pages: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Split page-aware text into chunks while preserving page number info.
        
        Args:
            pages: List of {"page_number": int, "text": str}
            chunk_size: Size of each chunk in tokens
            overlap: Overlap between chunks in tokens
            
        Returns:
            List of {"text": str, "page_number": int, "page_numbers": List[int]}
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        # Build a list of (sentence, page_number) tuples
        sentence_page_pairs = []
        for page_info in pages:
            page_num = page_info["page_number"]
            page_text = page_info["text"].replace('\n', ' ')
            
            if not page_text.strip():
                continue
            
            sentences = page_text.split('. ')
            for sent in sentences:
                sent = sent.strip()
                if sent:
                    sentence_page_pairs.append((sent + ". ", page_num))
        
        if not sentence_page_pairs:
            return []
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        current_pages = set()
        
        for sentence, page_num in sentence_page_pairs:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= chunk_size:
                current_chunk += sentence
                current_tokens += sentence_tokens
                current_pages.add(page_num)
            else:
                if current_chunk:
                    sorted_pages = sorted(current_pages)
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page_number": sorted_pages[0],
                        "page_numbers": sorted_pages,
                    })
                
                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = current_chunk[-overlap * 4:]
                    current_chunk = overlap_text + sentence
                    current_tokens = self.count_tokens(current_chunk)
                    # Overlap text comes from previous pages + new page
                    current_pages = set(current_pages)  # carry over
                    current_pages.add(page_num)
                else:
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                    current_pages = {page_num}
        
        if current_chunk:
            sorted_pages = sorted(current_pages)
            chunks.append({
                "text": current_chunk.strip(),
                "page_number": sorted_pages[0],
                "page_numbers": sorted_pages,
            })
        
        logger.info(
            f"Split {len(pages)} pages into {len(chunks)} page-aware chunks "
            f"(size: {chunk_size}, overlap: {overlap})"
        )
        return chunks

    def read_file_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read file and return page-aware data.
        For PDFs: returns actual page numbers.
        For other formats: returns single page (page_number=1).
        
        Returns:
            List of {"page_number": int, "text": str}
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._read_pdf_pages(file_path)
            else:
                # Non-PDF: treat as single page
                text = self.read_file(str(file_path))
                return [{"page_number": 1, "text": text}]
        except Exception as e:
            logger.error(f"Error reading file pages {file_path}: {e}")
            raise
    
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
