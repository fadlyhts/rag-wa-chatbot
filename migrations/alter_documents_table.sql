-- Migration: Add new columns to documents table
-- This should be run AFTER the main migration if tables already exist

-- Add new columns to documents table (if they don't exist)
ALTER TABLE documents 
  ADD COLUMN IF NOT EXISTS category_id INT NULL,
  ADD COLUMN IF NOT EXISTS file_path VARCHAR(500) NULL,
  ADD COLUMN IF NOT EXISTS file_size INT NULL,
  ADD COLUMN IF NOT EXISTS file_type VARCHAR(50) NULL,
  ADD COLUMN IF NOT EXISTS chunks_count INT DEFAULT 0 NOT NULL,
  ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP NULL,
  ADD COLUMN IF NOT EXISTS failed_reason TEXT NULL,
  ADD COLUMN IF NOT EXISTS upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL;

-- Add foreign key constraint (if not exists)
-- Note: This will fail silently if constraint already exists, which is fine
SET @constraint_check = (
    SELECT COUNT(*)
    FROM information_schema.TABLE_CONSTRAINTS
    WHERE CONSTRAINT_SCHEMA = DATABASE()
    AND TABLE_NAME = 'documents'
    AND CONSTRAINT_NAME = 'fk_document_category'
);

SET @alter_query = IF(
    @constraint_check = 0,
    'ALTER TABLE documents ADD CONSTRAINT fk_document_category FOREIGN KEY (category_id) REFERENCES document_categories(id) ON DELETE SET NULL',
    'SELECT "Foreign key already exists" AS message'
);

PREPARE stmt FROM @alter_query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- Add indexes (if they don't exist)
CREATE INDEX IF NOT EXISTS idx_category_id ON documents(category_id);
CREATE INDEX IF NOT EXISTS idx_category_status ON documents(category_id, embedding_status);
CREATE INDEX IF NOT EXISTS idx_upload_date ON documents(upload_date);

SELECT 'Documents table migration completed!' AS status;
