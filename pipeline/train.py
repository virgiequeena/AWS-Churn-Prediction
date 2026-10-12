import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == "__main__":
    # --- DYNAMIC PATH LOGIC ---
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/home/ec2-user/SageMaker/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/home/ec2-user/SageMaker/model")
    
    os.makedirs(model_dir, exist_ok=True)
    train_file = os.path.join(train_dir, "train.csv")
    
    if os.path.exists(train_file):
        df = pd.read_csv(train_file)
        X_train = df.drop("Churn", axis=1)
        y_train = df["Churn"]

        num_feat = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_feat = ["Gender", "SubscriptionType", "ContractLength"]

        numeric_preprocess = Pipeline([
            ("num_imputer", SimpleImputer(strategy="mean"))
        ])
        categorical_preprocess = Pipeline([
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OrdinalEncoder(categories=[
                ["Male", "Female"],
                ["Basic", "Standard", "Premium"],
                ["Monthly", "Quarterly", "Annual"]
            ]))
        ])

        preprocess = ColumnTransformer(transformers=[
            ("numPreprocess", numeric_preprocess, num_feat),
            ("catPreprocess", categorical_preprocess, cat_feat)
        ], remainder="drop")

        model = Pipeline([
            ("preprocessing", preprocess),
            ("classifier", RandomForestClassifier(criterion="gini", max_depth=4))
        ])

        model.fit(X_train, y_train)
        
        joblib.dump(model, os.path.join(model_dir, "model_churn.joblib"))
        print("✅ Training complete.")
    else:
        print(f"❌ Error: {train_file} not found!")
