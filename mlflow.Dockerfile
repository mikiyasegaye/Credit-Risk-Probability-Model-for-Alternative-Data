# Use the official Python image
FROM python:3.10-slim

# Install mlflow and gunicorn
RUN pip install --no-cache-dir mlflow[extras] gunicorn

# Create directories for artifacts and backend DB if needed
RUN mkdir -p /mlflow/artifacts /mlflow/db

# Set working directory
WORKDIR /mlflow

# Set environment variables
ENV BACKEND_STORE_URI=sqlite:///mlflow.db
ENV DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
ENV WORKERS=2
ENV TIMEOUT=120

# Expose MLflow UI port
EXPOSE 5050

# Start MLflow tracking server
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${DEFAULT_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5050 \
    --workers ${WORKERS}
