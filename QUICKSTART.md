# Quick Start Guide

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- MySQL 8.0+ (or use Docker)
- Redis (or use Docker)

## 1. Clone & Setup

```bash
cd D:\Magang\rag\backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure Environment

```bash
# Copy environment template
copy .env.example .env

# Edit .env file with your settings:
# - OPENAI_API_KEY=sk-your-key-here
# - DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/whatsapp_chatbot
# - REDIS_URL=redis://localhost:6379/0
```

## 3. Start Services with Docker Compose

### Option A: Use Existing Docker Compose (Recommended)

```bash
# Go to root directory
cd D:\Magang\rag

# Start all services
docker-compose -f tasks\general\docker-compose.yml up -d

# Check status
docker-compose -f tasks\general\docker-compose.yml ps
```

### Option B: Start Services Manually

```bash
# Start MySQL
docker run -d --name mysql -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=password123 \
  -e MYSQL_DATABASE=whatsapp_chatbot \
  mysql:8.0

# Start Redis
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Start Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest
```

## 4. Initialize Database

```bash
# From backend directory
cd D:\Magang\rag\backend

# Activate venv
.\venv\Scripts\activate

# Create tables (auto-created on first run)
python -c "from app.database.session import engine; from app.database.base import Base; from app.models import *; Base.metadata.create_all(engine)"
```

## 5. Run the Application

### Development Mode (with auto-reload)

```bash
uvicorn app.main:app --reload --port 8000
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 6. Test the API

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-06T...",
  "dependencies": {
    "database": "connected",
    "redis": "connected",
    "qdrant": "connected",
    "waha": "not available"
  }
}
```

### API Info

```bash
curl http://localhost:8000/api/info
```

### Test Webhook

```bash
curl -X POST http://localhost:8000/api/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "event": "message.incoming",
    "data": {
      "messageId": "test_123",
      "from": "1234567890",
      "text": "Hello chatbot!"
    }
  }'
```

Expected response:
```json
{
  "status": "queued",
  "request_id": "webhook_...",
  "job_id": "job_...",
  "message_id": "test_123"
}
```

### Get Messages

```bash
curl http://localhost:8000/api/messages?limit=10
```

### Get Statistics

```bash
curl http://localhost:8000/api/stats
```

## 7. Access API Documentation

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Common Issues

### Database Connection Error

```bash
# Check if MySQL is running
docker ps | grep mysql

# Test connection
mysql -h localhost -u root -p
```

### Redis Connection Error

```bash
# Check if Redis is running
docker ps | grep redis

# Test connection
redis-cli ping
```

### Port Already in Use

```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

## Next Steps

1. **Configure WAHA**: Set up WhatsApp connection
2. **Implement RAG Pipeline**: Add vector database and LLM integration
3. **Set up Message Queue**: Configure RQ workers
4. **Add Tests**: Run `pytest` to verify functionality
5. **Deploy**: Use Docker Compose for production deployment

## Development Workflow

```bash
# 1. Make changes to code

# 2. Format code
black app/

# 3. Run tests
pytest

# 4. Check health
curl http://localhost:8000/health

# 5. Test your changes
curl -X POST http://localhost:8000/api/webhook -d '...'
```

## Monitoring

### View Logs

```bash
# Application logs (console)
# Check the terminal where uvicorn is running

# Docker logs
docker-compose logs -f fastapi-backend
```

### Check Queue

```bash
# Connect to Redis
redis-cli

# List queues
KEYS *

# Check queue length
LLEN rq:queue:default
```

## Production Deployment

```bash
# Build Docker image
docker build -t whatsapp-rag-backend .

# Run with docker-compose
docker-compose up -d

# Scale workers
docker-compose up -d --scale rq-worker=3
```

## Support

- Documentation: See README.md
- Architecture: See tasks/backend/BACKEND-ARCHITECTURE-DESIGN.md
- Issues: Create issue in repository
