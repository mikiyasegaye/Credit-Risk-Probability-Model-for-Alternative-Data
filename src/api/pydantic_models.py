from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionRequest(BaseModel):
    """
    Pydantic model for prediction request data.
    This model expects customer transaction data to assess credit risk.

    Risk Factors:
    - Low customer_tenure (< 90 days) indicates higher risk
    - Low frequency (< 5 transactions) indicates higher risk
    - Low monetary value (< $100) indicates higher risk
    - Low avg_transaction_amount (< $50) indicates higher risk
    """

    monetary: float = Field(
        ...,
        description="Total monetary value of all transactions",
        gt=0,
        example=1500.0
    )
    frequency: int = Field(
        ...,
        description="Total number of transactions",
        gt=0,
        example=10
    )
    avg_transaction_amount: float = Field(
        ...,
        description="Average amount per transaction (monetary/frequency)",
        gt=0,
        example=150.0
    )
    customer_tenure: float = Field(
        ...,
        description="Number of days since first transaction",
        gt=0,
        example=365.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "monetary": 1500.0,
                "frequency": 10,
                "avg_transaction_amount": 150.0,
                "customer_tenure": 365.0
            }
        }


class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction response data.
    """

    risk_probability: float = Field(...,
                                    description="Probability of high risk")
    risk_label: str = Field(..., description="Risk label (high/low)")
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )
    features_used: List[str] = Field(
        ..., description="List of features used in prediction"
    )
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "risk_probability": 0.75,
                "risk_label": "high",
                "model_version": "1.0.0",
                "features_used": ["monetary", "frequency", "avg_transaction_amount", "customer_tenure"],
                "feature_importance": {
                    "monetary": 0.3,
                    "frequency": 0.4,
                    "avg_transaction_amount": 0.3,
                    "customer_tenure": 0.3,
                },
            }
        }
