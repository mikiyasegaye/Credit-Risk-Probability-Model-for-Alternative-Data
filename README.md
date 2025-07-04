---
# Credit Risk Probability Model for Alternative Data

## Overview

This project is part of the 10 Academy Artificial Intelligence Mastery challenge. In Week 1, our objective is to understand the concept of **Credit Risk** and establish the foundational structure required to build credit risk models in a regulatory-compliant, interpretable, and well-documented manner.
---

## Objectives

- Set up a clean, modular Python project with Git version control.
- Understand the regulatory and business context of credit risk modeling.
- Begin exploratory analysis of credit scoring techniques and modeling challenges.
- Document domain insights in the context of the Basel II Capital Accord.

---

## Folder Structure

```

├── .github/
│   └── workflows/           # CI/CD config (e.g., unit tests)
├── notebooks/               # Jupyter Notebooks for EDA and experiments
├── scripts/                 # Utility and data preparation scripts
├── src/                     # Main source code modules
├── tests/                   # Unit tests for the project
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation

```

---

## Environment Setup

1. **Clone the repository**

```bash
   git clone https://github.com/mikiyasegaye/Credit-Risk-Probability-Model-for-Alternative-Data.git
   cd Credit-Risk-Probability-Model-for-Alternative-Data
```

2. **Set up Python environment**

```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
```

3. **Install packages**
   Initial packages to install:

   - `pandas`, `numpy`, `matplotlib`, `seaborn`
   - `nltk`, `textblob`
   - `scikit-learn`
   - `jupyter`

4. **Launch Jupyter Notebook**

```bash
   jupyter notebook notebooks/
```

---

## Credit Scoring Business Understanding

### 1. How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord allows financial institutions to use internal models to estimate key credit risk parameters: Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). Under the Advanced Internal Ratings-Based (AIRB) approach, institutions gain flexibility but are also required to ensure transparency, traceability, and explainability.

This leads to a strong emphasis on interpretable and well-documented models because:

- Regulators must be able to understand and audit model outputs.
- Internal governance teams need clarity to validate assumptions and decisions.
- Business users (such as lending officers) must explain outcomes, especially in cases of credit rejection.

In this environment, model interpretability is not just helpful—it is essential for regulatory compliance and responsible risk management.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In many datasets, a direct indicator of default (such as legal default status or write-off confirmation) may not be available due to data limitations or reporting lags. As a workaround, organizations define a **proxy variable** to approximate default, such as "payment overdue by 90+ days" or "account closed due to non-payment."

While this is a practical necessity, using a proxy introduces risks:

- **Inaccuracy**: The proxy may not perfectly match real-world default behavior.
- **Bias**: It could systematically exclude certain default scenarios or overrepresent others.
- **Regulatory concerns**: Poorly justified proxies can undermine trust in the model and pose compliance issues.

Careful design and business validation of the proxy variable are critical to avoid misleading predictions and operational risks.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Aspect                          | Logistic Regression (with WoE)                        | Gradient Boosting (e.g., XGBoost, LightGBM)           |
| ------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| **Interpretability**            | High – coefficients and WoE are easily explainable    | Low – model behavior is complex and opaque            |
| **Predictive Power**            | Moderate – works well for linearly separable problems | High – captures non-linear interactions               |
| **Regulatory Acceptance**       | Strong – commonly accepted and auditable              | Weak – often requires justification and documentation |
| **Deployment Simplicity**       | Easy – scoring functions are light and portable       | More demanding – needs careful resource management    |
| **Auditability and Governance** | High – transparent decisions and low model risk       | Low – black-box models are harder to validate         |

In regulated financial settings, **simplicity and explainability are often prioritized over raw predictive performance**. Logistic regression with Weight of Evidence (WoE) is widely used due to its compliance with regulatory expectations. Complex models like Gradient Boosting may be used internally or as benchmark models but are less favored for production use without extensive documentation and model risk controls.

---

## References

- [Basel II Overview – Statistica Sinica](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [Alternative Credit Scoring – HKMA](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [Credit Scoring Guidelines – World Bank](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
- [Developing a Credit Risk Model – TDS](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Credit Risk Explained – CFI](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
- [Risk Officer on Credit Risk](https://www.risk-officer.com/Credit_Risk.htm)

---

### Model Development Process

1. **Data Preprocessing and Feature Engineering**

   - Transaction data cleaning and formatting
   - Feature engineering based on transaction patterns
   - RFM analysis for customer segmentation

2. **Model Training Pipeline**

   - Multiple model experiments tracked in MLflow
   - Hyperparameter optimization
   - Model performance evaluation
   - Best model selection and registration

3. **Model Deployment**
   - Containerized model serving
   - RESTful API endpoints
   - Prediction service with FastAPI

### Running the Project

1. **Start the Services**

```bash
# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps
```

2. **Access Points**

   - MLflow UI: `http://localhost:5050`
   - API Documentation:
     - Swagger UI: `http://localhost:8080/docs`
     - ReDoc: `http://localhost:8080/redoc`
   - Health Check: `http://localhost:8080/health`

3. **API Endpoints**

   - `POST /predict`: Make credit risk predictions
   - `GET /health`: Check API health status
   - `GET /`: Get API information and version

4. **Making Predictions**

```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "recency": 30,
           "frequency": 10,
           "monetary": 5000,
           "avg_transaction_amount": 500,
           "transaction_frequency": 0.33,
           "customer_age": 35,
           "customer_tenure": 365
         }'
```

### Service Architecture

The project uses a three-service architecture:

1. **MLflow Server (Port 5050)**

   - Experiment tracking
   - Model versioning
   - Model registry
   - Artifact storage

2. **Model Registration Service**

   - Trains models
   - Evaluates performance
   - Registers best model to MLflow
   - Automated via Docker

3. **API Service (Port 8080)**
   - FastAPI-based prediction service
   - Loads production model from MLflow
   - Handles real-time predictions
   - Health monitoring

### Resource Management

Each service has defined resource limits:

- MLflow: 1 CPU, 1GB RAM
- Model Registration: 2 CPU, 4GB RAM
- API Service: 2 CPU, 4GB RAM

### Monitoring and Maintenance

1. **Health Checks**

   - API health endpoint: `http://localhost:8080/health`
   - Docker health checks configured
   - Automatic container restart on failure

2. **Logging**

   - Centralized logging for all services
   - Performance metrics tracking
   - Error monitoring and reporting

3. **Model Updates**
   - Automated model retraining
   - Version control in MLflow
   - Zero-downtime model updates

### Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_data_processing.py
pytest tests/test_model_interaction.py
pytest tests/test_rfm_analysis.py
pytest tests/test_train.py
```

---

## License

This project is developed as part of 10 Academy. All rights belong to the respective contributors and institutions.

---
