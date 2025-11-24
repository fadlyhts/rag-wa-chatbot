"""OCR processing configuration"""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class OCRConfig:
    """Configuration settings for OCR processing"""
    
    # Image processing settings
    dpi: int = 150  # DPI for PDF to image conversion (100-300 recommended)
    
    # Parallel processing settings
    max_workers: int = 4  # Maximum parallel OCR processes
    batch_size_small: int = 10  # Threshold for small PDFs (process all at once)
    batch_size_adaptive: bool = True  # Use adaptive batch sizing
    
    # Timeout settings (in seconds)
    page_timeout: int = 30  # Timeout per page
    batch_timeout: int = 120  # Timeout per batch
    
    # Language settings
    languages: str = "eng+ind"  # Tesseract languages (English + Indonesian)
    
    # Memory management
    enable_garbage_collection: bool = True  # Force garbage collection after batches
    log_memory_usage: bool = False  # Log memory usage (for debugging)
    
    # Progress tracking
    progress_update_frequency: int = 1  # Update progress every N pages
    
    @classmethod
    def from_env(cls) -> 'OCRConfig':
        """Create config from environment variables"""
        return cls(
            dpi=int(os.getenv('OCR_DPI', '150')),
            max_workers=int(os.getenv('OCR_MAX_WORKERS', '4')),
            batch_size_small=int(os.getenv('OCR_BATCH_SIZE_SMALL', '10')),
            batch_size_adaptive=os.getenv('OCR_BATCH_SIZE_ADAPTIVE', 'true').lower() == 'true',
            page_timeout=int(os.getenv('OCR_PAGE_TIMEOUT', '30')),
            batch_timeout=int(os.getenv('OCR_BATCH_TIMEOUT', '120')),
            languages=os.getenv('OCR_LANGUAGES', 'eng+ind'),
            enable_garbage_collection=os.getenv('OCR_ENABLE_GC', 'true').lower() == 'true',
            log_memory_usage=os.getenv('OCR_LOG_MEMORY', 'false').lower() == 'true',
            progress_update_frequency=int(os.getenv('OCR_PROGRESS_FREQUENCY', '1'))
        )
    
    def get_batch_size(self, total_pages: int) -> int:
        """Calculate optimal batch size based on total pages"""
        if not self.batch_size_adaptive:
            return min(8, max(2, total_pages // 6))
        
        # Adaptive batch sizing
        if total_pages <= 5:
            return total_pages  # Process all at once
        elif total_pages <= 20:
            return min(4, total_pages // 2)  # Small batches
        elif total_pages <= 50:
            return min(6, total_pages // 4)  # Medium batches
        else:
            return min(8, total_pages // 6)  # Large batches
    
    def get_worker_count(self, batch_size: int) -> int:
        """Get optimal worker count for given batch size"""
        return min(self.max_workers, batch_size)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'dpi': self.dpi,
            'max_workers': self.max_workers,
            'batch_size_small': self.batch_size_small,
            'batch_size_adaptive': self.batch_size_adaptive,
            'page_timeout': self.page_timeout,
            'batch_timeout': self.batch_timeout,
            'languages': self.languages,
            'enable_garbage_collection': self.enable_garbage_collection,
            'log_memory_usage': self.log_memory_usage,
            'progress_update_frequency': self.progress_update_frequency
        }


# Global OCR configuration instance
ocr_config = OCRConfig.from_env()
