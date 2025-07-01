# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Create necessary directories
RUN mkdir -p /app/mlruns /app/data/processed && chmod -R 777 /app

# Copy source code, scripts and data
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080
ENV WORKERS=4
ENV TIMEOUT=120
ENV WORKER_CLASS="uvicorn.workers.UvicornWorker"

# Expose port
EXPOSE 8080

# Run the application with gunicorn
CMD ["sh", "-c", "gunicorn src.api.main:app --workers ${WORKERS} --timeout ${TIMEOUT} --bind 0.0.0.0:${PORT} --worker-class ${WORKER_CLASS} --preload"]
