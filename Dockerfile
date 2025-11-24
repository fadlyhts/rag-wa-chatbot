FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including OCR tools
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    default-libmysqlclient-dev \
    pkg-config \
    libmagic1 \
    default-mysql-client \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ind \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# Pin bcrypt to 4.x to ensure compatibility with passlib
RUN pip install --no-cache-dir "bcrypt>=4.0.1,<5.0.0" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory and copy entrypoint
RUN mkdir -p uploads

# Copy and set up entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create non-root user and fix permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown appuser:appuser /docker-entrypoint.sh

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Run application with entrypoint that handles migrations
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
