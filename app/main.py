"""FastAPI application entry point"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from apscheduler.schedulers.background import BackgroundScheduler

from app.api.endpoints import webhook, health, messages, stats, test, auth, documents, vector_db, dashboard
from app.api.endpoints import settings as settings_router, evaluation, feedback
from app.database.session import engine
from app.database.base import Base
from app.config import settings
from app.utils.logger import setup_logging
from app.exceptions import ChatbotException
from app.jobs.keep_alive import keep_waha_session_alive

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Background scheduler for periodic tasks
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    - Startup: Initialize database tables, start background jobs
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Log AI provider configuration
    from app.rag.config import rag_config
    logger.info("="*60)
    logger.info(f"🤖 AI PROVIDER: {rag_config.ai_provider.upper()}")
    if rag_config.ai_provider.lower() == "gemini":
        logger.info(f"📦 Model: {rag_config.gemini_model}")
        logger.info(f"🔢 Embedding: {rag_config.gemini_embedding_model}")
        api_key = rag_config.google_api_key
        logger.info(f"🔑 API Key: {api_key[:20]}..." if api_key else "❌ NOT SET")
    else:
        logger.info(f"📦 Model: {rag_config.llm_model}")
        logger.info(f"🔢 Embedding: {rag_config.embedding_model}")
        api_key = rag_config.openai_api_key
        logger.info(f"🔑 API Key: {api_key[:20]}..." if api_key else "❌ NOT SET")
    logger.info("="*60)
    
    # Create database tables
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
    
    # Start background scheduler
    try:
        # Add keep-alive job - runs every 5 minutes
        scheduler.add_job(
            keep_waha_session_alive,
            'interval',
            minutes=5,
            id='waha_keep_alive',
            replace_existing=True
        )
        scheduler.start()
        logger.info("Background scheduler started with keep-alive job")
    except Exception as e:
        logger.error(f"Scheduler initialization error: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}")
    
    # Stop scheduler
    try:
        scheduler.shutdown()
        logger.info("Background scheduler stopped")
    except Exception as e:
        logger.error(f"Scheduler shutdown error: {str(e)}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Intelligent RAG-powered WhatsApp chatbot using WAHA",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",   # Alternative dev port
        "https://*.easypanel.host", # Production (wildcard)
        "*"  # Allow all for now
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(webhook.router, prefix="/api", tags=["webhook"])
app.include_router(health.router, tags=["health"])
app.include_router(messages.router, prefix="/api", tags=["messages"])
app.include_router(stats.router, prefix="/api", tags=["stats"])
app.include_router(test.router, prefix="/api", tags=["test"])

# Admin panel routers (comprehensive with authentication)
app.include_router(auth.router, prefix="/api", tags=["authentication"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(vector_db.router, prefix="/api", tags=["vector-database"])
app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])
app.include_router(settings_router.router, prefix="/api", tags=["settings"])
app.include_router(evaluation.router, prefix="/api", tags=["evaluation"])
app.include_router(feedback.router, prefix="/api", tags=["feedback"])


# Exception handlers
@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request: Request, exc: ChatbotException):
    """Handle custom chatbot exceptions"""
    logger.error(f"Chatbot exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected error occurred"
        }
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"{settings.APP_NAME} API",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else "disabled"
    }


@app.get("/api/info")
async def api_info():
    """API information"""
    return {
        "app_name": settings.APP_NAME,
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "endpoints": {
            "webhook": "/api/webhook",
            "health": "/health",
            "messages": "/api/messages",
            "stats": "/api/stats"
        }
    }


@app.get("/api/debug/config")
async def debug_config():
    """Debug: Show loaded configuration (without secrets)"""
    return {
        "qdrant_url": settings.QDRANT_URL,
        "qdrant_api_key_set": bool(settings.QDRANT_API_KEY),
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "waha_api_url": settings.WAHA_API_URL,
        "waha_api_key_set": bool(settings.WAHA_API_KEY),
        "database_url": settings.DATABASE_URL.split("@")[1] if "@" in settings.DATABASE_URL else "***",
        "redis_url": settings.REDIS_URL,
        "openai_key_set": bool(settings.OPENAI_API_KEY),
        "debug_mode": settings.DEBUG,
        "environment": settings.ENVIRONMENT
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
