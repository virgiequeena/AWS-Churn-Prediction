"""
Streamlit UI for the Customer Churn classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os

import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "churn-endpoint")    # CHANGED: iris-endpoint → churn-endpoint
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list) -> dict:                          # CHANGED: list[float] → list (mixed types)
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


# CHANGED: title + description
st.title("Customer Churn Predictor")
st.write("Enter customer details below to predict churn probability via SageMaker.")

# CHANGED: 10 churn input widgets replacing 4 iris sliders
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
    # CHANGED: order must match FEATURE_NAMES in inference.py
    features = [
        int(age), gender, float(tenure), int(usage_freq),
        float(support_calls), int(payment_delay), subs_type,
        contract_length, float(total_spend), int(last_interaction)
    ]
    try:
        result = invoke_endpoint(features)
    except NoCredentialsError:
        st.error(
            "No AWS credentials found. If running on EC2, attach LabInstanceProfile. "
            "If running locally, configure ~/.aws/credentials."
        )
    except ClientError as e:
        st.error(f"AWS error: {e.response['Error'].get('Message', str(e))}")
    else:
        label = result["labels"][0]
        probs = result["probabilities"][0]

        # CHANGED: churn-specific result display
        if label == "Churn":
            st.error(f"⚠️ Prediction: **{label}**")
        else:
            st.success(f"✅ Prediction: **{label}**")

        st.write("Class probabilities:")
        st.bar_chart({"No Churn": probs[0], "Churn": probs[1]})       # CHANGED: named bars
