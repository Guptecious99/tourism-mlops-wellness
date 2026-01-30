import os
import json
import joblib
import pandas as pd

from datasets import load_dataset
from huggingface_hub import HfApi

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def main():
    # ----------------------------
    # Pull config from env (set by GitHub Actions secrets)
    # ----------------------------
    hf_token = os.environ.get("HF_TOKEN")
    hf_dataset_id = os.environ.get("HF_DATASET_ID")
    hf_model_id = os.environ.get("HF_MODEL_ID")

    if not hf_token:
        raise ValueError("HF_TOKEN is missing")
    if not hf_dataset_id:
        raise ValueError("HF_DATASET_ID is missing")
    if not hf_model_id:
        raise ValueError("HF_MODEL_ID is missing")

    # ----------------------------
    # Load train/test from Hugging Face Dataset Hub
    # ----------------------------
    print("Loading train/test from:", hf_dataset_id)

    ds = load_dataset(
        hf_dataset_id,
        data_files={"train": "train.csv", "test": "test.csv"}
    )

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    # ----------------------------
    # Split into X/y
    # ----------------------------
    if "ProdTaken" not in train_df.columns:
        raise ValueError("ProdTaken missing from train dataset")

    X_train = train_df.drop(columns=["ProdTaken"])
    y_train = train_df["ProdTaken"].astype(int)

    X_test = test_df.drop(columns=["ProdTaken"])
    y_test = test_df["ProdTaken"].astype(int)

    # ----------------------------
    # Build preprocessing + model pipeline
    # ----------------------------
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )

    # Using your tuned "best params" directly for a fast CI run
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    # ----------------------------
    # Train
    # ----------------------------
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # ----------------------------
    # Evaluate
    # ----------------------------
    print("Evaluating model...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_threshold": 0.5,
        "best_params_used": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42
        }
    }

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    metrics_path = "reports/metrics_ci.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = "models/model.joblib"
    joblib.dump(pipeline, model_path)

    print("Saved:", metrics_path)
    print("Saved:", model_path)

    # ----------------------------
    # Upload model + metrics to Hugging Face Model Hub
    # ----------------------------
    api = HfApi(token=hf_token)

    print("Uploading model to:", hf_model_id)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.joblib",
        repo_id=hf_model_id,
        repo_type="model"
    )

    print("Uploading CI metrics to:", hf_model_id)
    api.upload_file(
        path_or_fileobj=metrics_path,
        path_in_repo="metrics_ci.json",
        repo_id=hf_model_id,
        repo_type="model"
    )

    print("✅ CI training + registration complete.")
    print("ROC-AUC:", roc_auc, "| PR-AUC:", pr_auc)


if __name__ == "__main__":
    main()
