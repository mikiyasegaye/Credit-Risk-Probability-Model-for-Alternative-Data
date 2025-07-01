from fastapi import FastAPI, HTTPException
from .pydantic_models import PredictionRequest, PredictionResponse
import mlflow
import pandas as pd
import numpy as np
from typing import Dict
import logging
import os
import time
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using ML models",
    version="1.0.0",
)

# Global variables for model and preprocessing
model = None
model_version = None
feature_names = None

# Get MLflow tracking URI from environment variable or use default
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


@app.on_event("startup")
async def load_model():
    """Load the model from MLflow on startup."""
    global model, model_version, feature_names

    # Maximum number of retries
    max_retries = 5
    retry_delay = 10  # seconds

    for attempt in range(max_retries):
        try:
            # Set up MLflow tracking URI
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"Using MLflow tracking URI: {MLFLOW_TRACKING_URI}")

            # Load the production model
            model_name = "credit_risk_model"

            # Get the latest production version
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            production_versions = [
                v for v in versions if v.current_stage == "Production"]

            if not production_versions:
                raise Exception(
                    f"No production version found for model {model_name}")

            latest_version = production_versions[0].version
            model_version = latest_version

            logger.info(f"Loading model {model_name} version {model_version}")

            # Load the model using the model URI format
            model = mlflow.sklearn.load_model(
                model_uri=f"models:/{model_name}/{model_version}"
            )

            # Get feature names from the model
            feature_names = [
                "recency",
                "frequency",
                "monetary",
                "avg_transaction_amount",
                "transaction_frequency",
                "customer_age",
                "customer_tenure",
            ]

            logger.info(
                f"Model loaded successfully with features: {feature_names}")
            break

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Error loading model after {max_retries} attempts: {str(e)}")
                raise RuntimeError(f"Failed to load model: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Credit Risk Prediction API",
        "model_version": model_version,
        "status": "healthy",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": model_version}


def preprocess_features(data: Dict) -> pd.DataFrame:
    """
    Preprocess the input features to match model expectations.

    Args:
        data: Dictionary containing the raw features

    Returns:
        DataFrame with processed features
    """
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame([data])

        # Create all required features
        features = {
            # Transaction-based features
            'Total_Transaction_Amount': df['monetary'].iloc[0],
            'Avg_Transaction_Amount': df['avg_transaction_amount'].iloc[0],
            'Transaction_Count': df['frequency'].iloc[0],
            # Estimated variation
            'Std_Transaction_Amount': df['avg_transaction_amount'].iloc[0] * 0.1,
            # Estimated minimum
            'Min_Transaction_Amount': df['avg_transaction_amount'].iloc[0] * 0.8,
            # Estimated maximum
            'Max_Transaction_Amount': df['avg_transaction_amount'].iloc[0] * 1.2,
            'Total_Value': df['monetary'].iloc[0],
            'Avg_Value': df['monetary'].iloc[0] / df['frequency'].iloc[0],

            # Customer features
            'customer_tenure': df['customer_tenure'].iloc[0],
            'Subscription_Count': 1,  # Default for single transaction

            # Temporal features
            'Transaction_Hour': 12,  # Default to noon
            'Transaction_Day': 15,   # Default to mid-month
            'Transaction_Month': 6,  # Default to mid-year
            'Transaction_Year': 2024,
            'Transaction_DayOfWeek': 2,  # Default to Tuesday
            'Transaction_WeekOfYear': 26,  # Default to mid-year
            'Is_Weekend': 0,
            'Is_NightTime': 0,

            # Diversity features
            'Unique_Product_Categories': 1,
            'Unique_Channels': 1,
            'Unique_Providers': 1,
            'Unique_Currencies': 1
        }

        # Create final feature array with proper column order
        final_features = pd.DataFrame([features])

        # Keep the list of numerical columns handy in case we later load a
        # persisted scaler from disk / MLflow artefacts.
        numerical_features = [
            'Total_Transaction_Amount', 'Avg_Transaction_Amount', 'Transaction_Count',
            'Std_Transaction_Amount', 'Min_Transaction_Amount', 'Max_Transaction_Amount',
            'Total_Value', 'Avg_Value', 'customer_tenure', 'Subscription_Count',
            'Transaction_Hour', 'Transaction_Day', 'Transaction_Month', 'Transaction_Year',
            'Transaction_DayOfWeek', 'Transaction_WeekOfYear'
        ]

        # TODO: load a pre-fitted scaler and apply `transform` here once the
        # training pipeline persist it.  For now we simply pass the raw
        # numerical values to the model so that each request retains its
        # unique information.

        # Log preprocessed features for debugging
        logger.info(f"Input data: {data}")
        logger.info(
            f"Preprocessed features: {final_features.iloc[0].to_dict()}")
        logger.info(f"Number of features: {len(final_features.columns)}")

        return final_features

    except Exception as e:
        logger.error(f"Error preprocessing features: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error preprocessing features: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a credit risk prediction for the given customer data.

    Args:
        request: PredictionRequest object containing customer features

    Returns:
        PredictionResponse object containing risk prediction and metadata
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later."
        )

    try:
        # Convert request to dictionary
        data = request.model_dump()

        # Preprocess features
        features_df = preprocess_features(data)

        # Make prediction
        risk_prob = model.predict_proba(features_df)[0, 1]
        risk_label = "high" if risk_prob >= 0.5 else "low"

        # Get feature importance if available
        feature_importance = None
        if hasattr(model, "feature_importances_"):
            feature_importance = dict(
                zip(feature_names, model.feature_importances_.tolist())
            )

        # Prepare response
        response = PredictionResponse(
            risk_probability=float(risk_prob),
            risk_label=risk_label,
            model_version=model_version,
            features_used=feature_names,
            feature_importance=feature_importance,
        )

        # Log prediction
        logger.info(
            f"Prediction made: risk_probability={risk_prob:.3f}, risk_label={risk_label}"
        )

        return response

    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error making prediction: {str(e)}"
        )
