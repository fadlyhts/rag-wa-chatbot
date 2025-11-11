#!/bin/bash
set -e

echo "==================================="
echo "Starting WhatsApp RAG Chatbot"
echo "==================================="

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
max_tries=30
count=0
until python -c "from app.database.session import engine; engine.connect()" 2>/dev/null || [ $count -eq $max_tries ]; do
  echo "Waiting for database... ($count/$max_tries)"
  sleep 2
  count=$((count + 1))
done

if [ $count -eq $max_tries ]; then
  echo "ERROR: Could not connect to database after $max_tries attempts"
  exit 1
fi

echo "[OK] Database connection established"

# Run database migrations
echo "Running database migrations..."
python -c "
from app.database.session import engine
from app.database.base import Base
from app.models import *
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create new tables (admins, categories, chunks, settings)
    logger.info('Creating/updating database tables...')
    Base.metadata.create_all(bind=engine)
    logger.info('Database tables created/updated successfully')
    
    # Alter existing documents table to add new columns
    logger.info('Altering documents table to add new columns...')
    with engine.connect() as conn:
        # Check if category_id column exists
        result = conn.execute(text('''
            SELECT COUNT(*) as count FROM information_schema.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = 'documents' 
            AND COLUMN_NAME = 'category_id'
        '''))
        column_exists = result.fetchone()[0] > 0
        
        if not column_exists:
            logger.info('Adding new columns to documents table...')
            # Add columns one by one to handle MySQL limitations
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN category_id INT NULL'))
                logger.info('Added category_id column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN file_path VARCHAR(500) NULL'))
                logger.info('Added file_path column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN file_size INT NULL'))
                logger.info('Added file_size column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN file_type VARCHAR(50) NULL'))
                logger.info('Added file_type column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN chunks_count INT DEFAULT 0 NOT NULL'))
                logger.info('Added chunks_count column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN processed_at TIMESTAMP NULL'))
                logger.info('Added processed_at column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN failed_reason TEXT NULL'))
                logger.info('Added failed_reason column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            try:
                conn.execute(text('ALTER TABLE documents ADD COLUMN upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL'))
                logger.info('Added upload_date column')
            except Exception as e:
                logger.debug(f'Column might exist: {e}')
            
            conn.commit()
            logger.info('Documents table columns added successfully')
        else:
            logger.info('Documents table already has new columns')
        
except Exception as e:
    logger.error(f'Migration error: {e}')
    raise
"

# Seed default data if needed
echo "Seeding default data..."
python -c "
from app.database.session import SessionLocal
from app.models.admin import Admin
from app.models.document_category import DocumentCategory
from app.models.settings import Settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = SessionLocal()

try:
    # Create default admin if not exists
    admin = db.query(Admin).filter(Admin.username == 'admin').first()
    if not admin:
        logger.info('Creating default admin user...')
        
        # Try to hash password with bcrypt, fallback to pre-generated hash
        try:
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
            password_hash = pwd_context.hash('admin123')
            logger.info('Generated password hash with bcrypt')
        except Exception as e:
            # Fallback to pre-generated bcrypt hash (works with bcrypt 4.x and 5.x)
            logger.warning(f'Bcrypt hashing failed: {e}')
            logger.info('Using pre-generated password hash')
            # Pre-generated with bcrypt 4.x: password = admin123
            password_hash = '\$2b\$12\$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5ztC2x3eoKiOy'
        
        admin = Admin(
            username='admin',
            email='admin@example.com',
            password_hash=password_hash,
            role='super_admin',
            is_active=True
        )
        db.add(admin)
        db.commit()
        logger.info('Default admin created (username: admin, password: admin123)')
    else:
        logger.info('Admin user already exists')
    
    # Create default categories if not exist
    default_categories = [
        ('General', 'General documents and information'),
        ('Policy', 'Company policies and procedures'),
        ('FAQ', 'Frequently asked questions'),
        ('Product', 'Product documentation and guides'),
        ('Technical', 'Technical documentation')
    ]
    
    for name, desc in default_categories:
        cat = db.query(DocumentCategory).filter(DocumentCategory.name == name).first()
        if not cat:
            cat = DocumentCategory(name=name, description=desc)
            db.add(cat)
    
    db.commit()
    logger.info('Default categories ensured')
    
    # Create default settings if not exist
    settings_defaults = {
        'rag_config': {
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 500,
            'top_k': 5,
            'min_score': 0.7
        },
        'rate_limiting': {
            'messages_per_minute': 10,
            'enabled': True
        },
        'system_info': {
            'version': '1.0.0',
            'maintenance_mode': False
        }
    }
    
    for key, value in settings_defaults.items():
        setting = db.query(Settings).filter(Settings.setting_key == key).first()
        if not setting:
            setting = Settings(setting_key=key, setting_value=value)
            db.add(setting)
    
    db.commit()
    logger.info('Default settings ensured')
    
except Exception as e:
    logger.error(f'Seeding error: {e}')
    db.rollback()
    raise
finally:
    db.close()
"

echo "==================================="
echo "[SUCCESS] Migrations and seeding complete"
echo "==================================="

# Execute the main command (uvicorn)
exec "$@"
