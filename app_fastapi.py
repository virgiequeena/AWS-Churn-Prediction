"""
Streamlit UI for the Customer Churn classifier served by FastAPI (fastAPI.py).

Start the API first:
    uvicorn fastAPI:app --host 0.0.0.0 --port 8000
then run this app:
    streamlit run app_fastapi.py
"""

import os

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


def make_prediction(features):
    response = requests.post(f"{API_URL}/predict", json=features)
    return response.json()


st.title("Customer Churn Predictor")
st.write("Enter customer details below to predict churn probability via FastAPI.")

age             = st.number_input("Age", min_value=18, max_value=100, value=30)
gender          = st.radio("Gender", ["Male", "Female"])
tenure          = st.number_input("Tenure (years)", min_value=0, max_value=60, value=5)
usage_freq      = st.number_input("Usage Frequency (times/month)", min_value=0, max_value=100, value=10)
support_calls   = st.number_input("Support Calls", min_value=0, max_value=20, value=2)
payment_delay   = st.number_input("Payment Delay (days)", min_value=0, max_value=30, value=3)
subs_type       = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
total_spend     = st.number_input("Total Spend ($)", min_value=0, max_value=10000, value=500)
last_interaction= st.number_input("Last Interaction (days ago)", min_value=0, max_value=60, value=10)

if st.button("Predict", type="primary"):
    # keys must match CustomerInput in fastAPI.py
    features = {
        "Age": int(age),
        "Gender": gender,
        "Tenure": float(tenure),
        "UsageFrequency": int(usage_freq),
        "SupportCalls": float(support_calls),
        "PaymentDelay": int(payment_delay),
        "SubscriptionType": subs_type,
        "ContractLength": contract_length,
        "TotalSpend": float(total_spend),
        "LastInteraction": int(last_interaction),
    }
    try:
        result = make_prediction(features)
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_URL}. Is uvicorn running?")
    else:
        label = result["label"]
        probs = result["probabilities"]

        if label == "Churn":
            st.error(f"⚠️ Prediction: **{label}**")
        else:
            st.success(f"✅ Prediction: **{label}**")

        st.write("Class probabilities:")
        st.bar_chart(probs)
