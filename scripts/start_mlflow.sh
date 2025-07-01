#!/bin/bash

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Get port from environment variable or use default
PORT=${MLFLOW_PORT:-5050}

# Start MLflow tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port ${PORT}
