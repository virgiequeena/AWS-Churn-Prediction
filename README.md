# AWS Churn Prediction

Customer churn classifier trained with a SageMaker pipeline and served two ways: as a **FastAPI** service with a Streamlit front end, and as a **SageMaker endpoint** called from Streamlit on EC2.

## Overview

| | |
|---|---|
| Task | Binary classification: will a customer churn? |
| Model | scikit-learn `Pipeline` (imputation + ordinal encoding + `RandomForestClassifier`) |
| Features | Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction |
| Test results | Accuracy 0.975 · Precision 0.995 · Recall 0.961 |
| Stack | Python, scikit-learn, FastAPI, Streamlit, AWS SageMaker, AWS EC2 |

## Repository structure

```
├── fastAPI.py               # FastAPI service: loads the model, exposes POST /predict
├── app_fastapi.py           # Streamlit front end that calls the FastAPI service
├── app_streamlit.py         # Streamlit front end that calls the SageMaker endpoint
├── artifacts/
│   └── model_churn.joblib   # trained pipeline (output of pipeline/train.py)
├── pipeline/                # SageMaker pipeline: ingest → preprocess → train → evaluate
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluation.py
│   └── pipeline.py
├── deploy/
│   ├── deploy_endpoint.ipynb    # package the model, upload to S3, deploy the SageMaker endpoint
│   ├── inference.py             # SageMaker inference handlers
│   ├── user-data.sh             # EC2 bootstrap: Streamlit + SageMaker endpoint
│   └── user-data-fastapi.sh     # EC2 bootstrap: FastAPI + Streamlit
└── requirements.txt
```

## Option 1: FastAPI + Streamlit

```bash
pip install -r requirements.txt
uvicorn fastAPI:app --host 0.0.0.0 --port 8000
streamlit run app_fastapi.py
```

Interactive API docs are available at `http://127.0.0.1:8000/docs`.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 30, "Gender": "Female", "Tenure": 5, "UsageFrequency": 10, "SupportCalls": 2, "PaymentDelay": 3, "SubscriptionType": "Standard", "ContractLength": "Annual", "TotalSpend": 500, "LastInteraction": 10}'
```

Response:

```json
{"prediction": 1, "label": "Churn", "probabilities": {"No Churn": 0.257, "Churn": 0.743}}
```

To host both on a single EC2 instance, paste `deploy/user-data-fastapi.sh` into the instance's user data and open ports 8000 and 8501 in the security group.

## Option 2: SageMaker endpoint + Streamlit on EC2

1. Run `pipeline/pipeline.py` on a SageMaker notebook instance to train and evaluate the model.
2. Run `deploy/deploy_endpoint.ipynb` to package `model_churn.joblib`, upload it to S3 and deploy `churn-endpoint` (replace `<account-id>` with your own bucket).
3. Launch an EC2 instance with `deploy/user-data.sh` as user data. It clones this repo and runs `app_streamlit.py`, which calls the endpoint through `boto3`.

## Data

The customer churn dataset (`customer_churn.csv`) is not redistributed in this repository. Place it next to the pipeline scripts before running `pipeline/pipeline.py`.
