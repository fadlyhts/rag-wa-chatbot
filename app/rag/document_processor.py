"""Document processing and ingestion"""

from typing import List, Dict, Any, Optional, Tuple
import os
import hashlib
import logging
import gc
import re
import tempfile
import time
from pathlib import Path
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
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
    from pdf2image import convert_from_path, pdfinfo_from_path
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
            
    def extract_document_metadata(self, text: str) -> Dict[str, str]:
        """
        Extract structured metadata from the beginning of a document (like SOP/Instruksi Kerja headers)
        using regex patterns.
        """
        metadata = {}
        
        # We only search the first 3000 characters to avoid matching random text deep in the document
        search_area = text[:3000]
        
        # Patterns for SOP / Instruksi Kerja headers
        patterns = {
            "Jenis Dokumen": r'\b(INSTRUKSI\s+KERJA|STANDARD\s+OPERATING\s+PROCEDURE|SOP)\b',
            "Judul": r'Judul\s*:\s*([^\n]+)',
            "No. Dokumen": r'No\.?\s*Dokumen\s*:\s*([^\n]+)',
            "No. Revisi": r'No\.?\s*Revisi\s*:\s*([^\n]+)',
            "Tanggal Terbit": r'Tanggal\s*Terbit\s*:\s*([^\n]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, search_area, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                
                # Special fix for "Judul" which often catches the "Cap :" column in the same line
                if key == "Judul":
                    val = re.sub(r'\s*Cap\b.*$', '', val, flags=re.IGNORECASE).strip()
                    
                # Special fix for OCR misreading numbers in "No. Revisi" (e.g., 'OL' -> '01')
                if key == "No. Revisi":
                    val = val.replace('O', '0').replace('o', '0').replace('L', '1').replace('l', '1')
                
                # Clean up if Tesseract added weird characters at the end
                val = re.sub(r'[^a-zA-Z0-9\s\.\-\/]', '', val).strip()
                
                if val:
                    metadata[key] = val
                    
        return metadata
    
    # ─── Boilerplate / Cover-Page Noise Patterns ────────────────────────
    # Only GENERIC patterns that apply to ANY document type.
    # Content-specific phrases (company names, director titles) are NOT
    # listed here — instead, the preamble-truncation strategy below handles
    # them structurally by discarding all text before the first heading.
    _BOILERPLATE_PATTERNS: List[re.Pattern] = [
        # Phone / fax numbers (with or without country code)
        re.compile(r'(?:Telp|Tel|Fax|Faks|Phone)[\s.:]*[\(\)\d\s\-\+]+', re.IGNORECASE),
        # URLs and email addresses
        re.compile(r'https?://\S+', re.IGNORECASE),
        re.compile(r'Homepage\s*:\s*\S+', re.IGNORECASE),
        re.compile(r'E-?mail\s*:\s*\S+', re.IGNORECASE),
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', re.IGNORECASE),
        # Postal / mailing address lines
        re.compile(r'(?:Jl\.|Jalan)\s+.{5,}', re.IGNORECASE),
        re.compile(r'Kotak\s+Pos\s+\d+', re.IGNORECASE),
        # Page numbers / document control lines
        re.compile(r'^\s*(?:Page|Halaman)\s+\d+', re.IGNORECASE),
        re.compile(r'^\s*(?:.{0,15})?\bNo\.\s+(?:Dokumen|Revisi)\b', re.IGNORECASE),
        re.compile(r'^\s*(?:.{0,15})?\bTanggal\s+(?:Efektif|Terbit)\b', re.IGNORECASE),
        re.compile(r'^\s*(?:.{0,15})?\bJudul\s*:', re.IGNORECASE),
        # Ensure company/doc types only match if they are standalone headers (short lines) or followed by typical header symbols
        re.compile(r'^\s*(?:SISTEM MANAJEMEN|PT PERKEBUNAN NUSANTARA|INSTRUKSI KERJA|STANDARD OPERATING PROCEDURE|SOP)\s*(?:$|:|-)', re.IGNORECASE),
    ]

    def _clean_pages_boilerplate(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove noise from document pages before semantic chunking.
        Returns a new list of pages with cleaned text.
        """
        # Phase 1: Check if there's any level-1 heading to allow preamble truncation
        has_heading = False
        for p in pages:
            for line in p["text"].split('\n'):
                heading_info = self._detect_heading(line)
                if heading_info and heading_info["level"] == 1:
                    has_heading = True
                    break
            if has_heading:
                break
                
        found_heading = not has_heading
        
        cleaned_pages = []
        total_removed = 0
        
        for p in pages:
            lines = p["text"].split('\n')
            cleaned_lines = []
            
            for line in lines:
                if not found_heading:
                    heading_info = self._detect_heading(line)
                    if heading_info and heading_info["level"] == 1:
                        found_heading = True
                    else:
                        total_removed += len(line) + 1
                        continue
                
                stripped = line.strip()
                if not stripped:
                    cleaned_lines.append(line)
                    continue
                    
                is_noise = False
                for pat in self._BOILERPLATE_PATTERNS:
                    if pat.search(stripped):
                        is_noise = True
                        break
                        
                if not is_noise and len(stripped) < 15 and stripped.isupper():
                    if not re.match(r'^\d+\.', stripped):
                        is_noise = True
                        
                if is_noise:
                    total_removed += len(line) + 1
                    continue
                    
                cleaned_lines.append(line)
                
            cleaned_pages.append({
                "page_number": p["page_number"],
                "text": "\n".join(cleaned_lines)
            })
            
        if total_removed > 0:
            logger.info(f"Boilerplate cleaning removed ~{total_removed} characters total")
            
        return cleaned_pages

    # ─── Heading Detection for Indonesian SOP Documents ─────────────────
    # Patterns ordered from most specific (deepest heading) to broadest.
    # Each tuple: (compiled_regex, heading_level)
    #   level 1 = top chapter,  level 2 = section,  level 3+ = sub-section
    HEADING_PATTERNS: List[Tuple[re.Pattern, int]] = [
        # "BAB I. Pendahuluan" or "BAB II PENDAHULUAN"
        (re.compile(r'^(BAB\s+[IVXLC]+)\.?\s*(.*)', re.IGNORECASE), 1),
        # "1. PERSIAPAN KEBUN PERBANYAKAN" (single digit + ALL CAPS title)
        (re.compile(r'^(\d+)\.\s+([A-Z][A-Z\s]{3,})$'), 1),
        # "1.1.1. Sub-sub-heading" (three-level numbering)
        (re.compile(r'^(\d+\.\d+\.\d+)\.?\s+(.+)'), 3),
        # "1.1. Definisi" or "1.1 Definisi" (two-level numbering)
        (re.compile(r'^(\d+\.\d+)\.?\s+(.+)'), 2),
        # "A. Major heading" (uppercase letter)
        (re.compile(r'^([A-Z])\.\s+(.{3,})'), 2),
        # "a. Penentuan blok kebun" (lowercase letter sub-item)
        (re.compile(r'^([a-z])\.\s+(.{3,})'), 3),
    ]

    def _detect_heading(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Check if a line is a heading. Returns heading info or None.
        
        Returns:
            {"number": "1.1", "title": "Definisi", "level": 2, "full": "1.1. Definisi"}
            or None if not a heading
        """
        stripped = line.strip()
        if not stripped or len(stripped) < 2:
            return None
        
        for pattern, level in self.HEADING_PATTERNS:
            match = pattern.match(stripped)
            if match:
                number = match.group(1).strip()
                title = match.group(2).strip() if match.lastindex >= 2 else ""
                return {
                    "number": number,
                    "title": title,
                    "level": level,
                    "full": stripped,
                }
        return None

    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse text into a flat list of sections delimited by headings.
        
        Each section dict:
            {
                "heading": "1.1. Definisi" | None (for preamble text),
                "heading_number": "1.1" | None,
                "heading_level": 2 | None,
                "body": "Persiapan kebun perbanyakan adalah ...",
            }
        """
        lines = text.split('\n')
        sections: List[Dict[str, Any]] = []
        current_heading: Optional[Dict[str, Any]] = None
        body_lines: List[str] = []
        
        current_offset = 0
        section_start_offset = 0

        for line in lines:
            line_len = len(line) + 1  # +1 for the '\n'
            heading_info = self._detect_heading(line)
            if heading_info:
                # Flush the previous section
                body_text = '\n'.join(body_lines).strip()
                if body_text or current_heading:
                    sections.append({
                        "heading": current_heading["full"] if current_heading else None,
                        "heading_number": current_heading["number"] if current_heading else None,
                        "heading_level": current_heading["level"] if current_heading else None,
                        "body": body_text,
                        "start_offset": section_start_offset
                    })
                current_heading = heading_info
                body_lines = []
                section_start_offset = current_offset
            else:
                body_lines.append(line)
                
            current_offset += line_len

        # Flush the last section
        body_text = '\n'.join(body_lines).strip()
        if body_text or current_heading:
            sections.append({
                "heading": current_heading["full"] if current_heading else None,
                "heading_number": current_heading["number"] if current_heading else None,
                "heading_level": current_heading["level"] if current_heading else None,
                "body": body_text,
                "start_offset": section_start_offset
            })


        return sections

    def _build_parent_map(self, sections: List[Dict[str, Any]]) -> Dict[int, str]:
        """
        Build a mapping: section_index -> parent heading string.
        A "parent" is the nearest preceding section with a *lower* heading level.
        """
        parent_map: Dict[int, str] = {}
        # Stack of (level, heading_full_text)
        heading_stack: List[Tuple[int, str]] = []

        for idx, sec in enumerate(sections):
            level = sec.get("heading_level")
            heading = sec.get("heading")

            if level is not None and heading is not None:
                # Pop everything with level >= current (siblings/children)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                # The top of the stack is the parent
                if heading_stack:
                    parent_map[idx] = heading_stack[-1][1]
                # Push current onto stack
                heading_stack.append((level, heading))
            else:
                # No heading (preamble) — parent is whatever is on stack
                if heading_stack:
                    parent_map[idx] = heading_stack[-1][1]

        return parent_map

    def _is_structured_document(self, text: str, min_headings: int = 3) -> bool:
        """
        Quick heuristic: does this text have enough heading-like lines
        to justify semantic chunking?
        """
        count = 0
        for line in text.split('\n'):
            if self._detect_heading(line):
                count += 1
                if count >= min_headings:
                    return True
        return False

    # ─── OCR Post-Processing ───────────────────────────────────────────
    # Tesseract often misreads scanned SOP documents: heading numbers
    # (1.1., 1.2., ...) appear as a separate block from their titles
    # (Definisi, Tujuan, ...). This method reassembles them.

    # Pattern: a line that is ONLY a section number (possibly OCR-garbled)
    _RE_ORPHAN_NUMBER = re.compile(
        r'^(\d+[\.,]\d*[\.,]?\d*)[\.,]?\s*$'   # "1.1." or "11," or "14." or "2.4,"
    )
    # Pattern: a line that looks like a title (Capitalised word, no number prefix)
    _RE_ORPHAN_TITLE = re.compile(
        r'^([A-Z][a-z].{2,})$'                  # "Definisi", "Tujuan", "Sasaran"
    )

    def _fix_ocr_number(self, raw: str) -> str:
        """
        Fix common Tesseract misreads of section numbers.
        Examples: '11,' -> '1.1.', '14.' -> '1.4.', '2.4,' -> '2.4.'
        """
        # Remove trailing comma/period and whitespace
        cleaned = raw.strip().rstrip('.,')
        
        # Case: "11" → could be "1.1" (two single digits merged)
        if cleaned.isdigit() and len(cleaned) == 2:
            return f"{cleaned[0]}.{cleaned[1]}."
        
        # Case: "14" → "1.4"
        if cleaned.isdigit() and len(cleaned) == 2:
            return f"{cleaned[0]}.{cleaned[1]}."
        
        # Case: "111" → "1.1.1"  (three digits merged)
        if cleaned.isdigit() and len(cleaned) == 3:
            return f"{cleaned[0]}.{cleaned[1]}.{cleaned[2]}."
        
        # Case: already has dots like "2.4" or "1.2" → just ensure trailing dot
        if re.match(r'^\d+\.\d+(\.\d+)?$', cleaned):
            return cleaned + '.'
        
        # Case: comma instead of dot: "2,4" → "2.4."
        if re.match(r'^\d+,\d+$', cleaned):
            return cleaned.replace(',', '.') + '.'
        
        return cleaned + '.'

    def _postprocess_ocr_text(self, text: str) -> str:
        """
        Fix Tesseract layout analysis issues in scanned SOP documents.
        
        Problem: Tesseract reads heading numbers as a column, then titles
        as a separate column, producing:
            1.1.
            1.2.
            1.3.
            Definisi
            Body text about definisi...
            Tujuan
            Body text about tujuan...
        
        Fix: Two-pass approach:
        1. Detect and collect blocks of orphan numbers
        2. Scan forward to find short title-like lines and pair them with numbers
        """
        lines = text.split('\n')
        
        # ── Pass 1: Find orphan number blocks ──────────────────────────
        # An "orphan block" is 2+ consecutive number-only lines 
        # (with possible blank lines between them)
        orphan_blocks: List[Dict[str, Any]] = []  # {start_line, end_line, numbers[]}
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            match = self._RE_ORPHAN_NUMBER.match(line)
            
            if match:
                # Potential start of orphan block — look ahead for more numbers
                block_numbers = [match.group(1)]
                block_start = i
                j = i + 1
                
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:  # skip blanks between numbers
                        j += 1
                        continue
                    next_match = self._RE_ORPHAN_NUMBER.match(next_line)
                    if next_match:
                        block_numbers.append(next_match.group(1))
                        j += 1
                    else:
                        break
                
                if len(block_numbers) >= 2:
                    # Found an orphan block
                    orphan_blocks.append({
                        'start_line': block_start,
                        'end_line': j - 1,  # last number line
                        'numbers': block_numbers,
                    })
                    i = j
                    continue
            
            i += 1
        
        if not orphan_blocks:
            return text  # No orphan blocks found, return unchanged
        
        # ── Pass 2: Remove orphan blocks and pair numbers with titles ──
        # Title heuristic: a short line (< 60 chars) that starts with a 
        # capital letter followed by lowercase, and doesn't look like
        # a continuation sentence.
        
        # Mark lines that belong to orphan blocks for removal
        orphan_line_set = set()
        for block in orphan_blocks:
            for li in range(block['start_line'], block['end_line'] + 1):
                orphan_line_set.add(li)
            # Also mark blank lines between block end and next content
            j = block['end_line'] + 1
            while j < len(lines) and not lines[j].strip():
                orphan_line_set.add(j)
                j += 1
        
        # Build clean lines (without orphan number blocks)
        clean_lines = []
        line_mapping = []  # maps clean_line_index -> original_line_index
        for i, line in enumerate(lines):
            if i not in orphan_line_set:
                clean_lines.append(line)
                line_mapping.append(i)
        
        # Now find title-like lines in clean_lines and pair with orphan numbers
        # A title line: short, starts with uppercase, is NOT a numbered list item,
        # and is either preceded by a blank line, a heading, or is the first line
        
        # Flatten all orphan numbers in order
        all_numbers: List[str] = []
        for block in orphan_blocks:
            all_numbers.extend(block['numbers'])
        
        number_idx = 0  # pointer into all_numbers
        result_lines: List[str] = []
        # Track if previous line was a heading (to help title detection)
        prev_was_heading = False
        
        for ci, cline in enumerate(clean_lines):
            stripped = cline.strip()
            
            # Check if current line is itself a heading (already has number)
            current_is_heading = bool(self._detect_heading(stripped)) if stripped else False
            
            if number_idx < len(all_numbers) and stripped:
                # Blacklist pattern for headers, footers, metadata to avoid incorrect heading assignment
                blacklist_patterns = [
                    r'standard\s+operating\s+procedure',
                    r'kultur\s+teknis',
                    r'tanaman\s+teh',
                    r'perkebunan\s+nusantara',
                    r'direktur',
                    r'komoditi',
                    r'januari',
                    r'bandung',
                    r'page\s+\d+',
                    r'halaman\s+\d+',
                    r'^no\.\s+dokumen',
                    r'^no\.\s+revisi',
                    r'^tanggal\s+efektif'
                ]
                is_blacklisted = any(re.search(pat, stripped.lower()) for pat in blacklist_patterns)

                # Check if this line looks like a section title
                # Title characteristics: short, capitalised, standalone word/phrase
                is_title = (
                    len(stripped) < 60 and
                    stripped[0].isupper() and
                    not stripped[0].isdigit() and
                    not is_blacklisted and
                    # Not a numbered list item
                    not re.match(r'^\d+\)', stripped) and
                    # Not a lettered sub-item
                    not re.match(r'^[a-d]\.\s', stripped) and
                    # Previous line is blank, or this is first line, or prev was heading
                    (ci == 0 or not clean_lines[ci - 1].strip() or prev_was_heading)
                )
                
                if is_title:
                    fixed_num = self._fix_ocr_number(all_numbers[number_idx])
                    result_lines.append(f"{fixed_num} {stripped}")
                    number_idx += 1
                    prev_was_heading = True
                    continue
            
            prev_was_heading = current_is_heading
            result_lines.append(cline)
        
        return '\n'.join(result_lines)


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
        if PDFPLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages, 1):
                        page_text = page.extract_text() or ""
                        pages.append({"page_number": i, "text": page_text})
            except Exception as e:
                logger.warning(f"pdfplumber failed: {e}. Trying fallback.")
                with open(file_path, 'rb') as file:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(file)
                    for i, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text() or ""
                        pages.append({"page_number": i, "text": page_text})
        else:
            with open(file_path, 'rb') as file:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text() or ""
                    pages.append({"page_number": i, "text": page_text})
        
        # If PDF has very little text per page on average, it's likely scanned with a digital watermark - use OCR
        total_text = "".join(p["text"] for p in pages)
        avg_chars_per_page = len(total_text.strip()) / max(len(pages), 1)
        
        # A normal text page usually has 1500-3000 chars. If < 500, it's likely just a footer/watermark.
        if avg_chars_per_page < 500 and OCR_AVAILABLE:
            logger.info(f"PDF appears to be scanned (avg {avg_chars_per_page:.1f} chars/page), using OCR: {file_path}")
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
            
            # Perform OCR (PSM 4 = Assume a single column of text of variable sizes)
            text = pytesseract.image_to_string(image, lang='eng+ind', config='--psm 4')  # English + Indonesian
            
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
            
            # Get total pages without loading images
            info = pdfinfo_from_path(file_path)
            total_pages = info["Pages"]
            
            logger.info(f"PDF has {total_pages} pages - starting OCR processing (DPI: 200)")
            
            # Sangat Penting: Batasi Tesseract agar hanya menggunakan 1 thread CPU
            # Jika tidak dibatasi, Tesseract akan memakai semua core (100%) dan memicu crash/healthcheck timeout
            os.environ['OMP_THREAD_LIMIT'] = '1'
            os.environ['TESSCORE_LIMIT'] = '1'
            
            pages: List[Dict[str, Any]] = []
            
            # Create a temporary directory to store extracted images on disk
            with tempfile.TemporaryDirectory() as tmp_dir:
                for i in range(1, total_pages + 1):
                    try:
                        logger.info(f"OCR processing page {i}/{total_pages}")
                        
                        # paths_only=True means it writes direct to disk and returns paths, NOT loading into Python RAM
                        image_paths = convert_from_path(
                            file_path, 
                            dpi=200, 
                            first_page=i, 
                            last_page=i, 
                            output_folder=tmp_dir,
                            fmt="jpeg",
                            paths_only=True
                        )
                        
                        if not image_paths:
                            continue
                            
                        img_path = image_paths[0]
                        
                        # Pass the file path directly to Tesseract, bypassing Python memory completely
                        # Use PSM 4 to preserve top-to-bottom single column reading order
                        page_text = pytesseract.image_to_string(img_path, lang='eng+ind', config='--psm 4', timeout=60)
                        
                        # Fix Tesseract layout issues (orphan numbers separated from titles)
                        page_text = self._postprocess_ocr_text(page_text)
                        
                        pages.append({"page_number": i, "text": page_text})
                        
                        # Clean up the temp image file immediately
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            
                        # Force garbage collection
                        gc.collect()
                        
                        # Beri nafas untuk CPU (1 detik) agar FastAPI bisa merespon Healthcheck dari server Easypanel
                        time.sleep(1)
                        
                        if i % 10 == 0:
                            logger.info(f"OCR progress: {i}/{total_pages} pages completed")
                            
                    except Exception as page_error:
                        logger.warning(f"Failed to OCR page {i}, skipping: {page_error}")
                        pages.append({"page_number": i, "text": ""})
                        # Force garbage collection even on error
                        gc.collect()
                        time.sleep(1)
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
        current_sentences = []  # List of (sentence_text, page_num, tokens)
        current_tokens = 0
        
        for sentence, page_num in sentence_page_pairs:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= chunk_size:
                current_sentences.append((sentence, page_num, sentence_tokens))
                current_tokens += sentence_tokens
            else:
                if current_sentences:
                    chunk_text = "".join(s for s, p, t in current_sentences).strip()
                    chunk_pages = sorted(list(set(p for s, p, t in current_sentences)))
                    chunks.append({
                        "text": chunk_text,
                        "page_number": chunk_pages[0] if chunk_pages else page_num,
                        "page_numbers": chunk_pages,
                    })
                
                # Start new chunk with overlap
                if overlap > 0 and current_sentences:
                    overlap_sentences = []
                    overlap_tokens = 0
                    # Traverse backwards to get overlap sentences
                    for s, p, t in reversed(current_sentences):
                        if overlap_tokens + t <= overlap:
                            overlap_sentences.insert(0, (s, p, t))
                            overlap_tokens += t
                        else:
                            break
                    
                    # We do not force carry-over if a sentence exceeds the overlap limit.
                    # This prevents massive chunks if a single 'sentence' is huge.
                        
                    current_sentences = overlap_sentences
                    current_tokens = overlap_tokens
                    
                    # Add the new sentence
                    current_sentences.append((sentence, page_num, sentence_tokens))
                    current_tokens += sentence_tokens
                else:
                    current_sentences = [(sentence, page_num, sentence_tokens)]
                    current_tokens = sentence_tokens
        
        if current_sentences:
            chunk_text = "".join(s for s, p, t in current_sentences).strip()
            chunk_pages = sorted(list(set(p for s, p, t in current_sentences)))
            chunks.append({
                "text": chunk_text,
                "page_number": chunk_pages[0] if chunk_pages else 1,
                "page_numbers": chunk_pages,
            })
        
        logger.info(
            f"Split {len(pages)} pages into {len(chunks)} page-aware chunks "
            f"(size: {chunk_size}, overlap: {overlap})"
        )
        return chunks

    def chunk_text_semantic_with_pages(
        self,
        pages: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Heading-aware semantic chunking for structured documents (SOP, manuals).
        
        Detects heading patterns (1., 1.1., a., BAB, etc.) and uses them as
        chunk boundaries. Each chunk contains a coherent section with its heading
        and parent heading context.
        
        Falls back to sentence-based chunking for non-structured documents.
        
        Args:
            pages: List of {"page_number": int, "text": str}
            chunk_size: Max chunk size in tokens
            overlap: Overlap in tokens (used for fallback)
            
        Returns:
            List of {
                "text": str,
                "page_number": int,
                "page_numbers": List[int],
                "heading": Optional[str],
                "parent_heading": Optional[str],
            }
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        # Remove cover-page boilerplate and noise per page BEFORE joining
        pages = self._clean_pages_boilerplate(pages)
        
        # Combine all page text while tracking page boundaries
        full_text = "\n".join(p["text"] for p in pages)
        
        # Check if the document has enough headings for semantic chunking
        if not self._is_structured_document(full_text):
            logger.info("Document is not structured — falling back to sentence-based chunking")
            fallback = self.chunk_text_with_pages(pages, chunk_size, overlap)
            # Add empty heading fields for consistent interface
            for chunk in fallback:
                chunk.setdefault("heading", None)
                chunk.setdefault("parent_heading", None)
            return fallback
        
        logger.info("Document has structured headings — using semantic chunking")
        
        # Build a character-offset → page_number map
        # This lets us figure out which page a section belongs to.
        page_char_offsets: List[Tuple[int, int, int]] = []  # (start, end, page_num)
        offset = 0
        for p in pages:
            text = p["text"]
            page_char_offsets.append((offset, offset + len(text), p["page_number"]))
            offset += len(text) + 1  # +1 for the '\n' joiner
        
        def _get_pages_for_range(start: int, end: int) -> List[int]:
            """Given character range in full_text, return page numbers."""
            result = []
            for pstart, pend, pnum in page_char_offsets:
                if pstart < end and pend > start:
                    result.append(pnum)
            return sorted(set(result)) if result else [1]
        
        # Parse into sections
        sections = self._parse_sections(full_text)
        parent_map = self._build_parent_map(sections)
        
        logger.info(f"Parsed {len(sections)} sections from document")
        
        # Extract exact section character offsets computed during parsing
        section_char_starts = [sec.get("start_offset", 0) for sec in sections]
        
        # Build chunks by merging/splitting sections to fit chunk_size
        chunks: List[Dict[str, Any]] = []
        
        # Buffer for accumulating small consecutive sections
        buf_sections: List[int] = []  # indices into sections[]
        buf_tokens: int = 0
        
        def _section_text(idx: int) -> str:
            """Build the full text for a section: heading + body."""
            sec = sections[idx]
            parts = []
            if sec["heading"]:
                parts.append(sec["heading"])
            if sec["body"]:
                parts.append(sec["body"])
            return "\n".join(parts)
        
        def _flush_buffer():
            """Emit the buffered sections as a single chunk."""
            nonlocal buf_sections, buf_tokens
            if not buf_sections:
                return
            
            # Build chunk text
            chunk_parts = [_section_text(i) for i in buf_sections]
            chunk_text = "\n\n".join(chunk_parts).strip()
            
            if not chunk_text:
                buf_sections = []
                buf_tokens = 0
                return
            
            # Determine heading & parent from the first section in buffer
            first_sec = sections[buf_sections[0]]
            heading = first_sec["heading"]
            parent = parent_map.get(buf_sections[0])
            
            # Determine page numbers
            first_start = section_char_starts[buf_sections[0]]
            last_idx = buf_sections[-1]
            last_end = section_char_starts[last_idx] + len(_section_text(last_idx))
            chunk_pages = _get_pages_for_range(first_start, last_end)
            
            chunks.append({
                "text": chunk_text,
                "page_number": chunk_pages[0],
                "page_numbers": chunk_pages,
                "heading": heading,
                "parent_heading": parent,
            })
            
            buf_sections = []
            buf_tokens = 0
        
        def _split_large_section(idx: int):
            """
            When a single section exceeds chunk_size, split its body by
            paragraphs and emit multiple chunks, each prefixed with the
            section heading for context.
            """
            sec = sections[idx]
            heading = sec["heading"] or ""
            parent = parent_map.get(idx)
            body = sec["body"] or ""
            
            # Split by double-newline (paragraphs) first, then single newline
            paragraphs = re.split(r'\n\s*\n', body)
            if len(paragraphs) <= 1:
                paragraphs = body.split('\n')
            
            heading_prefix = f"{heading}\n" if heading else ""
            heading_tokens = self.count_tokens(heading_prefix)
            
            current_parts: List[str] = []
            current_tokens = heading_tokens
            
            sec_start = section_char_starts[idx]
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                para_tokens = self.count_tokens(para)
                
                if current_tokens + para_tokens <= chunk_size:
                    current_parts.append(para)
                    current_tokens += para_tokens
                else:
                    # Emit current accumulated paragraphs
                    if current_parts:
                        chunk_text = heading_prefix + "\n".join(current_parts)
                        chunk_pages = _get_pages_for_range(
                            sec_start, sec_start + len(chunk_text)
                        )
                        chunks.append({
                            "text": chunk_text.strip(),
                            "page_number": chunk_pages[0],
                            "page_numbers": chunk_pages,
                            "heading": heading or None,
                            "parent_heading": parent,
                        })
                    
                    # Start new accumulation with this paragraph
                    current_parts = [para]
                    current_tokens = heading_tokens + para_tokens
            
            # Flush remaining
            if current_parts:
                chunk_text = heading_prefix + "\n".join(current_parts)
                chunk_pages = _get_pages_for_range(
                    sec_start, sec_start + len(chunk_text)
                )
                chunks.append({
                    "text": chunk_text.strip(),
                    "page_number": chunk_pages[0],
                    "page_numbers": chunk_pages,
                    "heading": heading or None,
                    "parent_heading": parent,
                })
        
        # Main loop: iterate sections and decide merge / emit / split
        for idx in range(len(sections)):
            sec_text = _section_text(idx)
            sec_tokens = self.count_tokens(sec_text)
            
            if sec_tokens > chunk_size:
                # Flush buffer first, then split the oversized section
                _flush_buffer()
                _split_large_section(idx)
            elif buf_tokens + sec_tokens <= chunk_size:
                # Fits in current buffer — accumulate
                buf_sections.append(idx)
                buf_tokens += sec_tokens
            else:
                # Would exceed — flush buffer, start new
                _flush_buffer()
                buf_sections = [idx]
                buf_tokens = sec_tokens
        
        # Flush any remaining buffer
        _flush_buffer()
        
        logger.info(
            f"Semantic chunking: {len(sections)} sections → {len(chunks)} chunks "
            f"(max size: {chunk_size} tokens)"
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
