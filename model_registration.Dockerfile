FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories and set permissions
RUN mkdir -p /app/mlruns && chmod -R 777 /app

# Copy source code and scripts
COPY src/ src/
COPY scripts/ scripts/

# Set environment variables
ENV PYTHONPATH=/app

# Run the model registration script
CMD ["python", "scripts/register_best_model.py"]
