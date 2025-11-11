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

echo "✓ Database connection established"

# Run database migrations
echo "Running database migrations..."
python -c "
from app.database.session import engine
from app.database.base import Base
from app.models import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Create all tables
    logger.info('Creating/updating database tables...')
    Base.metadata.create_all(bind=engine)
    logger.info('✓ Database tables created/updated successfully')
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
from passlib.context import CryptContext
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = SessionLocal()
pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')

try:
    # Create default admin if not exists
    admin = db.query(Admin).filter(Admin.username == 'admin').first()
    if not admin:
        logger.info('Creating default admin user...')
        admin = Admin(
            username='admin',
            email='admin@example.com',
            password_hash=pwd_context.hash('admin123'),
            role='super_admin',
            is_active=True
        )
        db.add(admin)
        db.commit()
        logger.info('✓ Default admin created (username: admin, password: admin123)')
    else:
        logger.info('✓ Admin user already exists')
    
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
    logger.info('✓ Default categories ensured')
    
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
    logger.info('✓ Default settings ensured')
    
except Exception as e:
    logger.error(f'Seeding error: {e}')
    db.rollback()
    raise
finally:
    db.close()
"

echo "==================================="
echo "✓ Migrations and seeding complete"
echo "==================================="

# Execute the main command (uvicorn)
exec "$@"
