#!/bin/bash

# Create mlruns directory if it doesn't exist
mkdir -p mlruns

# Start MLflow tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
