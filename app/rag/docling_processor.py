import logging
from typing import List, Dict, Any, Optional
import os

try:
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """
    Processor using IBM Docling for advanced document layout parsing.
    Requires docling and easyocr installed.
    
    Key configurations:
    - OCR: EasyOCR with Indonesian + English language support
    - Table: Explicit table structure recognition enabled
    - Chunker: HybridChunker (token-aware, structure-respecting)
    - Reading Order: Handled automatically by Docling's Heron layout model
    """
    
    # Token limit for chunks — aligned with typical embedding model limits
    MAX_CHUNK_TOKENS = 512
    
    def __init__(self):
        if not HAS_DOCLING:
            logger.error("Docling is not installed. Please install with `pip install docling`.")
            raise ImportError("docling is not installed")
        
        # Use HybridChunker: respects document hierarchy AND enforces token limits.
        # This prevents chunks that are too large (overflow embedding context)
        # or too small (lose semantic meaning).
        self.chunker = HybridChunker(
            max_tokens=self.MAX_CHUNK_TOKENS,
            merge_peers=True,  # Merge adjacent small sections into a single chunk
        )
        
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a document using Docling and chunks it with HybridChunker.
        Returns a list of chunks matching the expected format.
        """
        from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        import PyPDF2
        
        # Auto-detect if document is scanned by checking PyPDF2 text density
        force_ocr = False
        try:
            reader = PyPDF2.PdfReader(file_path)
            chars = sum(len(p.extract_text() or '') for p in reader.pages)
            avg_chars = chars / len(reader.pages) if len(reader.pages) > 0 else 0
            
            if avg_chars < 500:
                logger.info(f"Auto-detected SCANNED document (avg chars/page: {avg_chars:.1f}). Enabling force OCR.")
                force_ocr = True
            else:
                logger.info(f"Auto-detected DIGITAL document (avg chars/page: {avg_chars:.1f}). Disabling force OCR.")
        except Exception as e:
            logger.warning(f"Failed to auto-detect document type, defaulting to False: {e}")
            
        # --- Pipeline Options ---
        pipeline_options = PdfPipelineOptions()
        
        # OCR: Enable with Indonesian + English language support
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=["id", "en"],              # Indonesian + English (critical for SOP/IK documents)
            use_gpu=True,                   # Leverage GPU for faster OCR
            force_full_page_ocr=force_ocr,  # Force full-page OCR for scanned docs
        )
        
        # Table structure: Explicitly enable for accurate table extraction
        pipeline_options.do_table_structure = True
        
        # Image scale: Higher resolution for scanned documents improves OCR accuracy
        if force_ocr:
            pipeline_options.images_scale = 1.5  # 1.5x resolution for scanned docs
        
        # Disable image generation (not needed for text RAG, saves memory)
        pipeline_options.generate_page_images = False
        pipeline_options.generate_picture_images = False
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        logger.info(f"Docling is converting file: {file_path}")
        logger.info(f"  OCR languages: ['id', 'en'], force_ocr: {force_ocr}, images_scale: {pipeline_options.images_scale}")
        result = converter.convert(file_path)
        
        logger.info("Chunking document using HybridChunker (max_tokens=%d, merge_peers=True)...", self.MAX_CHUNK_TOKENS)
        doc_chunks = list(self.chunker.chunk(result.document))
        
        logger.info(f"Generated {len(doc_chunks)} hybrid chunks.")
        
        page_chunks = []
        for c in doc_chunks:
            text = c.text
            
            # Extract full heading hierarchy if available
            heading = None
            parent_heading = None
            if hasattr(c.meta, 'headings') and c.meta.headings:
                heading = c.meta.headings[-1]  # Most specific (deepest) heading
                if len(c.meta.headings) > 1:
                    parent_heading = c.meta.headings[-2]  # Parent heading for context
            
            # Extract page numbers from provenance
            page_numbers = []
            if hasattr(c.meta, 'doc_items'):
                for item in c.meta.doc_items:
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if hasattr(prov, 'page_no'):
                                page_numbers.append(prov.page_no)
            
            # Deduplicate and sort page numbers
            page_numbers = sorted(set(page_numbers))
            primary_page = page_numbers[0] if page_numbers else 1
            
            page_chunks.append({
                "text": text,
                "page_number": primary_page,
                "page_numbers": page_numbers,
                "heading": heading,
                "parent_heading": parent_heading,
            })
            
        return page_chunks
