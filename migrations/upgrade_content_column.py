"""
Migration: Upgrade documents.content column from TEXT to LONGTEXT

This fixes the error: "Data too long for column 'content'"
Allows storing large OCR documents (up to 4GB instead of 64KB)

Run this script to upgrade your database:
    python migrations/upgrade_content_column.py
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

def upgrade_content_column():
    """Upgrade documents.content column from TEXT to LONGTEXT"""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ ERROR: DATABASE_URL not found in environment variables")
        print("   Make sure .env file exists with DATABASE_URL")
        return False
    
    print("=" * 60)
    print("Upgrading documents.content column")
    print("=" * 60)
    print(f"Database: {database_url.split('@')[-1]}")  # Hide password
    print()
    
    try:
        # Create engine
        engine = create_engine(database_url)
        
        with engine.connect() as connection:
            # Check current column type
            result = connection.execute(text("""
                SELECT COLUMN_TYPE 
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'documents' 
                AND COLUMN_NAME = 'content'
            """))
            
            current_type = result.scalar()
            print(f"Current column type: {current_type}")
            
            if current_type and 'longtext' in current_type.lower():
                print("✅ Column is already LONGTEXT - no upgrade needed")
                return True
            
            # Perform upgrade
            print()
            print("Upgrading column to LONGTEXT...")
            print("This may take a few seconds...")
            
            connection.execute(text("""
                ALTER TABLE documents 
                MODIFY COLUMN content LONGTEXT NOT NULL
            """))
            
            connection.commit()
            
            # Verify upgrade
            result = connection.execute(text("""
                SELECT COLUMN_TYPE 
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = DATABASE() 
                AND TABLE_NAME = 'documents' 
                AND COLUMN_NAME = 'content'
            """))
            
            new_type = result.scalar()
            print()
            print(f"✅ Upgrade complete!")
            print(f"   Old type: {current_type}")
            print(f"   New type: {new_type}")
            print()
            print("=" * 60)
            print("Migration successful!")
            print("=" * 60)
            print()
            print("Benefits:")
            print("  • Can now store documents up to 4GB (was 64KB)")
            print("  • OCR of large documents will work")
            print("  • No more 'Data too long' errors")
            print()
            
            return True
            
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Migration failed!")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Check DATABASE_URL in .env file")
        print("  2. Ensure database is accessible")
        print("  3. Ensure you have ALTER permissions")
        print()
        return False

if __name__ == "__main__":
    success = upgrade_content_column()
    sys.exit(0 if success else 1)
