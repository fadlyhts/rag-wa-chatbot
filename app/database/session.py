"""Database session management"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Create engine with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=30,
    max_overflow=10,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=settings.DEBUG,
    poolclass=QueuePool
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Handle new database connections"""
    logger.debug("New database connection established")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Handle connection checkout from pool"""
    logger.debug("Connection checked out from pool")
