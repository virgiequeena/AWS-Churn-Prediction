import os
import pandas as pd

def ingest_data():
    # 1. Inside a SageMaker container, use these paths:
    container_input = "/opt/ml/processing/input"
    container_output = "/opt/ml/processing/ingested"
    
    # 2. Logic to check if we are in a container or testing locally on notebook
    if os.path.exists(container_input):
        input_dir = container_input
        output_dir = container_output
    else:
        # Fallback for when you run '!python data_ingestion.py' manually
        input_dir = "/home/ec2-user/SageMaker"
        output_dir = "/home/ec2-user/SageMaker/ingested"
    
    os.makedirs(output_dir, exist_ok=True)

    # 3. Look for the file in the input_dir
    input_file = os.path.join(input_dir, "customer_churn.csv")
    
    print(f"Looking for file at: {input_file}")
    
    if os.path.exists(input_file):
        df = pd.read_csv(input_file, sep=";")
        output_file = os.path.join(output_dir, "customer_churn.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ Success: Ingested data saved to {output_file}")
    else:
        print(f"❌ Error: {input_file} not found!")
        # Debug: list what the container actually sees
        print(f"Container sees these files in {input_dir}: {os.listdir(input_dir)}")

if __name__ == "__main__":
    ingest_data()
