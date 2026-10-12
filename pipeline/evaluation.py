import joblib
import pandas as pd
import os
import json
import tarfile
from sklearn.metrics import accuracy_score, precision_score, recall_score

if __name__ == "__main__":
    if os.environ.get("SM_CHANNEL_TEST") or os.path.exists("/opt/ml/processing"):
        model_dir  = "/opt/ml/processing/model"
        test_path  = "/opt/ml/processing/test/test.csv"
        output_dir = "/opt/ml/processing/evaluation"
    else:
        model_dir  = "/home/ec2-user/SageMaker/model"
        test_path  = "/home/ec2-user/SageMaker/test/test.csv"
        output_dir = "/home/ec2-user/SageMaker/eval"

    os.makedirs(output_dir, exist_ok=True)

    model_tar  = os.path.join(model_dir, "model.tar.gz")
    model_path = os.path.join(model_dir, "model_churn.joblib")

    if os.path.exists(model_tar) and not os.path.exists(model_path):
        print(f"Extracting model from {model_tar}...")
        with tarfile.open(model_tar, "r:gz") as tar:
            tar.extractall(model_dir)
        print(f"✅ Extracted. Contents: {os.listdir(model_dir)}")

    if os.path.exists(model_path) and os.path.exists(test_path):
        model   = joblib.load(model_path)
        test_df = pd.read_csv(test_path)

        X_test = test_df.drop("Churn", axis=1)
        y_test = test_df["Churn"]

        predictions = model.predict(X_test)
        acc  = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average="binary")
        rec  = recall_score(y_test, predictions, average="binary")

        report_dict = {
            "binary_classification_metrics": {
                "accuracy":  {"value": acc,  "standard_deviation": "NaN"},
                "precision": {"value": prec, "standard_deviation": "NaN"},
                "recall":    {"value": rec,  "standard_deviation": "NaN"}
            }
        }

        output_file_path = os.path.join(output_dir, "evaluation.json")
        with open(output_file_path, "w") as f:
            json.dump(report_dict, f)

        print(f"✅ Evaluation complete. Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"Report saved to: {output_file_path}")
    else:
        print("❌ Error: Missing files for evaluation.")
        if not os.path.exists(model_path): print(f"Missing model at: {model_path}")
        if not os.path.exists(test_path):  print(f"Missing test data at: {test_path}")
