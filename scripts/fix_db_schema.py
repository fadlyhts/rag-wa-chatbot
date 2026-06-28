import sys
import os
from pathlib import Path
from sqlalchemy import text, inspect

# Add parent directory to path to import app
sys.path.append(str(Path(__file__).parent.parent))

from app.database.session import engine

def fix_schema():
    print("Checking database schema...")
    
    with engine.connect() as conn:
        inspector = inspect(engine)
        columns = [c['name'] for c in inspector.get_columns('documents')]
        print(f"Existing columns: {columns}")
        
        # Columns to check and add
        new_columns = [
            ("ocr_progress_current", "INT DEFAULT 0 NOT NULL"),
            ("ocr_progress_total", "INT DEFAULT 0 NOT NULL"),
            ("file_path", "VARCHAR(500) NULL"),
            ("file_size", "INT NULL"),
            ("file_type", "VARCHAR(50) NULL"),
            ("chunks_count", "INT DEFAULT 0 NOT NULL"),
            ("processed_at", "DATETIME NULL"),
            ("failed_reason", "TEXT NULL"),
            ("upload_date", "DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL"),
            ("category_id", "INT NULL"),
            ("content", "LONGTEXT NULL"), # Ensure content exists and is LONGTEXT
            ("source_url", "TEXT NULL"),
            ("doc_metadata", "JSON NULL")
        ]
        
        for col_name, col_def in new_columns:
            if col_name not in columns:
                print(f"Adding missing column: {col_name}")
                try:
                    conn.execute(text(f"ALTER TABLE documents ADD COLUMN {col_name} {col_def}"))
                    conn.commit()
                    print(f"  [OK] Added {col_name}")
                except Exception as e:
                    print(f"  [ERROR] Failed to add {col_name}: {e}")
            else:
                # Special check for content column type if needed, but skipping for now to be safe
                print(f"Column {col_name} exists.")
                
        # Check if category_id needs index
        # This is harder to check via inspector easily in one go, usually safe to skip if it exists
        
        print("\nSchema fix complete.")

if __name__ == "__main__":
    fix_schema()
