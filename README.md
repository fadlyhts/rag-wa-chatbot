# WhatsApp RAG Chatbot Backend

FastAPI-based backend for intelligent WhatsApp chatbot with RAG capabilities.

## Features

- 🚀 FastAPI with async support
- 💬 WAHA (WhatsApp HTTP API) integration
- 🧠 RAG pipeline with LangChain + Qdrant
- 📊 MySQL database with SQLAlchemy ORM
- 🔄 Redis + Python-RQ for message queue
- 🔒 Rate limiting and webhook validation
- 📈 Monitoring and statistics endpoints
- 🐳 Docker support

## Project Structure

```
backend/
├── app/
│   ├── api/endpoints/       # API route handlers
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   ├── rag/                 # RAG pipeline
│   ├── database/            # Database config
│   ├── utils/               # Utilities
│   ├── jobs/                # Background jobs
│   ├── security/            # Security utilities
│   ├── config.py            # Configuration
│   ├── exceptions.py        # Custom exceptions
│   └── main.py              # FastAPI app
├── tests/                   # Test suite
├── migrations/              # Alembic migrations
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker image
└── .env.example             # Environment template
```

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# - OPENAI_API_KEY
# - DATABASE_URL
# - REDIS_URL
# - QDRANT_URL
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f fastapi-backend

# Stop services
docker-compose down
```

### 4. Run Locally

```bash
# Start FastAPI
uvicorn app.main:app --reload --port 8000

# Or use Python
python -m app.main
```

## API Endpoints

### Webhook
- `POST /api/webhook` - Receive WAHA webhooks

### Health
- `GET /health` - Health check (database, redis, qdrant, waha)

### Messages
- `GET /api/messages` - List messages with filters

### Stats
- `GET /api/stats` - System statistics

### Documentation
- `GET /docs` - Swagger UI (development only)
- `GET /redoc` - ReDoc (development only)

## Database

### Run Migrations

```bash
# Generate migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=app

# Specific test file
pytest tests/unit/test_services/test_message_service.py
```

## Development

### Code Quality

```bash
# Format code
black app/

# Lint
flake8 app/

# Type check
mypy app/
```

### Environment Variables

See `.env.example` for all available configuration options.

### Rate Limiting

Default: 10 messages/minute per user
Configure: `RATE_LIMIT_MESSAGES_PER_MINUTE` in `.env`

## Deployment

### Docker

```bash
# Build image
docker build -t whatsapp-rag-backend .

# Run container
docker run -p 8000:8000 --env-file .env whatsapp-rag-backend
```

### Production Checklist

- [ ] Set `DEBUG=False`
- [ ] Configure secure `WEBHOOK_SECRET`
- [ ] Set strong `DATABASE_URL` password
- [ ] Configure SSL/TLS
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Set up backup strategy
- [ ] Review rate limits

## Monitoring

Access metrics at `/metrics` (when enabled).

Key metrics:
- Request count and latency
- Database connection pool usage
- Queue depth
- LLM token usage
- Error rates

## Troubleshooting

### Database Connection Issues

```bash
# Test connection
python -c "from app.database.session import engine; engine.connect()"
```

### Redis Connection Issues

```bash
# Test Redis
redis-cli ping
```

### WAHA Issues

Check WAHA logs:
```bash
docker-compose logs waha
```

## License

MIT License

## Support

For issues and questions, please create an issue in the repository.
