from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
#Load the machine learning model (sklearn Pipeline: preprocessing + RandomForest)
model = joblib.load('artifacts/model_churn.joblib')

CLASS_NAMES = ["No Churn", "Churn"]

#field names must match the column names the pipeline was trained on
class CustomerInput(BaseModel):
    Age: int
    Gender: Literal["Male", "Female"]
    Tenure: float
    UsageFrequency: int
    SupportCalls: float
    PaymentDelay: int
    SubscriptionType: Literal["Basic", "Standard", "Premium"]
    ContractLength: Literal["Monthly", "Quarterly", "Annual"]
    TotalSpend: float
    LastInteraction: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Churn Predictor"}

@app.post('/predict')
def predict(data: CustomerInput):
    features = pd.DataFrame([data.dict()])
    probs = model.predict_proba(features)[0]
    prediction = int(np.argmax(probs))
    return {
        'prediction': prediction,
        'label': CLASS_NAMES[prediction],
        'probabilities': {'No Churn': float(probs[0]), 'Churn': float(probs[1])}
    }
