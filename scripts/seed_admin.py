"""Seed default admin user and data"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.database.session import engine

def seed_data():
    """Seed default admin user and categories"""
    
    print("\n" + "="*60)
    print("Seeding Admin User and Default Data")
    print("="*60)
    
    with engine.connect() as conn:
        try:
            # Insert admin user with pre-hashed password (admin123)
            print("\n1. Creating admin user...")
            conn.execute(text("""
                INSERT INTO admins (username, email, password_hash, role, is_active, created_at)
                VALUES (
                    'admin',
                    'admin@example.com',
                    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5ztC2x3eoKiOy',
                    'super_admin',
                    1,
                    NOW()
                )
                ON DUPLICATE KEY UPDATE username=username
            """))
            print("   [OK] Admin user created/verified")
            
            # Insert default categories
            print("\n2. Creating default categories...")
            categories = [
                ('General', 'General documents and information'),
                ('Policy', 'Company policies and procedures'),
                ('FAQ', 'Frequently asked questions'),
                ('Product', 'Product documentation and guides'),
                ('Technical', 'Technical documentation')
            ]
            
            for name, desc in categories:
                conn.execute(text("""
                    INSERT INTO document_categories (name, description, created_at)
                    VALUES (:name, :desc, NOW())
                    ON DUPLICATE KEY UPDATE name=name
                """), {"name": name, "desc": desc})
                print(f"   [OK] {name}")
            
            # Insert default settings
            print("\n3. Creating default settings...")
            conn.execute(text("""
                INSERT INTO settings (setting_key, setting_value, updated_at) VALUES
                ('rag_config', '{"model": "gpt-4", "temperature": 0.7, "max_tokens": 500, "top_k": 5, "min_score": 0.7}', NOW()),
                ('rate_limiting', '{"messages_per_minute": 10, "enabled": true}', NOW()),
                ('system_info', '{"version": "1.0.0", "maintenance_mode": false}', NOW())
                ON DUPLICATE KEY UPDATE setting_key=setting_key
            """))
            print("   [OK] Settings created")
            
            conn.commit()
            
            # Verify admin user
            result = conn.execute(text("SELECT username, email, role FROM admins WHERE username='admin'"))
            admin = result.fetchone()
            
            print("\n" + "="*60)
            print("[SUCCESS] SEEDING COMPLETE!")
            print("="*60)
            print("\nDefault Admin Credentials:")
            print(f"  Username: {admin[0]}")
            print("  Password: admin123")
            print(f"  Email: {admin[1]}")
            print(f"  Role: {admin[2]}")
            print("\n[!] IMPORTANT: Change the password after first login!")
            print("\nNext steps:")
            print("  1. Start server: uvicorn app.main:app --reload")
            print("  2. Test auth: python scripts/test_auth.py")
            print("  3. Access docs: http://localhost:8000/docs")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n[ERROR] Failed: {str(e)}")
            conn.rollback()
            raise


if __name__ == "__main__":
    seed_data()
