"""Add OCR progress tracking columns to documents table

This migration adds columns to track OCR processing progress in real-time.
"""

from alembic import op
import sqlalchemy as sa


def upgrade():
    """Add OCR progress tracking columns"""
    try:
        # Add OCR progress columns
        op.add_column('documents', sa.Column('ocr_progress_current', sa.Integer(), nullable=False, server_default='0'))
        op.add_column('documents', sa.Column('ocr_progress_total', sa.Integer(), nullable=False, server_default='0'))
        op.add_column('documents', sa.Column('ocr_progress_message', sa.String(500), nullable=True))
        
        print("✅ Successfully added OCR progress tracking columns")
        
    except Exception as e:
        print(f"❌ Error adding OCR progress columns: {e}")
        raise


def downgrade():
    """Remove OCR progress tracking columns"""
    try:
        op.drop_column('documents', 'ocr_progress_message')
        op.drop_column('documents', 'ocr_progress_total')
        op.drop_column('documents', 'ocr_progress_current')
        
        print("✅ Successfully removed OCR progress tracking columns")
        
    except Exception as e:
        print(f"❌ Error removing OCR progress columns: {e}")
        raise


if __name__ == "__main__":
    # Direct execution for manual migration
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from app.database.session import engine
    from sqlalchemy import text
    
    print("Running OCR progress columns migration...")
    
    with engine.connect() as connection:
        try:
            # Check if columns already exist
            result = connection.execute(text("""
                SELECT COUNT(*) as count FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'documents' 
                AND COLUMN_NAME = 'ocr_progress_current'
            """))
            
            if result.fetchone()[0] > 0:
                print("OCR progress columns already exist")
                sys.exit(0)
            
            # Add columns
            connection.execute(text("""
                ALTER TABLE documents 
                ADD COLUMN ocr_progress_current INT NOT NULL DEFAULT 0,
                ADD COLUMN ocr_progress_total INT NOT NULL DEFAULT 0,
                ADD COLUMN ocr_progress_message VARCHAR(500) NULL
            """))
            
            connection.commit()
            print("✅ OCR progress columns added successfully")
            
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            connection.rollback()
            sys.exit(1)
