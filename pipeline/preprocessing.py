import pandas as pd
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # --- DYNAMIC PATH LOGIC ---
    if os.environ.get("SM_CHANNEL_INPUT") or os.path.exists("/opt/ml/processing"):
        input_dir = "/opt/ml/processing/ingested"
        out_train = "/opt/ml/processing/train"
        out_test = "/opt/ml/processing/test"
    else:
        input_dir = "/home/ec2-user/SageMaker/ingested"
        out_train = "/home/ec2-user/SageMaker/train"
        out_test = "/home/ec2-user/SageMaker/test"

    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)

    input_file = os.path.join(input_dir, "customer_churn.csv")
    
    if os.path.exists(input_file):
        df = pd.read_csv(input_file)

        df = df.rename(columns={
            "Usage Frequency":  "UsageFrequency",
            "Support Calls":    "SupportCalls",
            "Payment Delay":    "PaymentDelay",
            "Subscription Type":"SubscriptionType",
            "Contract Length":  "ContractLength",
            "Total Spend":      "TotalSpend",
            "Last Interaction": "LastInteraction"
        })

        df = df.drop(columns=["CustomerID"])

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        train_df.to_csv(os.path.join(out_train, "train.csv"), index=False)
        test_df.to_csv(os.path.join(out_test, "test.csv"), index=False)
        print("✅ Preprocessing complete.")
    else:
        print(f"❌ Error: {input_file} not found!")
