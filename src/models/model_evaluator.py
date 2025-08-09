import os
import pandas as pd
import joblib
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_on_test_data(model_name, test_data_path, target="sentiment_weighted_engagement", model_path=None, feature_columns_path=None, logger=None):
    """
    Evaluate a trained model on a user-supplied test dataset.
    Args:
        model_name: Name of the model ("Random Forest", "XGBoost", etc.)
        test_data_path: Path to the test CSV file (e.g., outputs/engineered_data_test.csv)
        target: Target variable name (default: sentiment_weighted_engagement)
        model_path: Optional path to the trained model file
        feature_columns_path: Optional path to feature columns file
        logger: Optional logger for logging messages
    Returns:
        Dictionary of evaluation metrics
    """
    log = logger.info if logger else print
    log(f"Evaluating {model_name} on test data: {test_data_path}")
    # Load test data
    if not os.path.exists(test_data_path):
        msg = f"Test data file not found: {test_data_path}"
        if logger: logger.error(msg)
        else: print(msg)
        return {"error": "Test data file not found."}
    df = pd.read_csv(test_data_path)
    # Load feature columns
    if feature_columns_path is None:
        feature_columns_path = "outputs/feature_columns.json"
    if os.path.exists(feature_columns_path):
        with open(feature_columns_path, "r") as f:
            feature_columns = json.load(f)
    else:
        feature_columns = [col for col in df.columns if col not in {"post_id", target}]
    # Prepare X, y
    X = df[feature_columns].fillna(0)
    if target in df.columns:
        y_raw = df[target].fillna(0)
        if y_raw.nunique() > 1:
            median_val = y_raw.median()
            y = (y_raw > median_val).astype(int)
        else:
            msg = f"{target} has no variance in test set, using likes as fallback target"
            if logger: logger.warning(msg)
            else: print(msg)
            if "likes" in df.columns:
                likes_median = df["likes"].median()
                y = (df["likes"] > likes_median).astype(int)
            else:
                return {"error": "No suitable target in test set."}
    else:
        msg = f"Target column {target} not found in test set."
        if logger: logger.error(msg)
        else: print(msg)
        return {"error": f"Target column {target} not found in test set."}
    # Load model
    if model_path is None:
        model_path = f"outputs/models/"
        if model_name.lower() == "random forest":
            model_path += "rf_model.pkl"
        elif model_name.lower() == "xgboost":
            model_path += "xgb_model.pkl"
        elif model_name.lower() == "lightgbm":
            model_path += "lgb_model.pkl"
        else:
            msg = f"Model loading not implemented for {model_name}"
            if logger: logger.error(msg)
            else: print(msg)
            return {"error": f"Model loading not implemented for {model_name}"}
    if not os.path.exists(model_path):
        msg = f"Model file not found: {model_path}"
        if logger: logger.error(msg)
        else: print(msg)
        return {"error": "Model file not found."}
    model = joblib.load(model_path)
    # Predict and evaluate
    y_pred = model.predict(X)
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        if proba.shape[1] > 1:
            y_pred_proba = proba[:, 1]
        else:
            y_pred_proba = proba[:, 0]
    else:
        y_pred_proba = y_pred
    results = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 and y_pred_proba is not None else 0.5
    }
    log(f"Test set evaluation for {model_name}: {results}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name (Random Forest, XGBoost, LightGBM)')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--target', type=str, default='sentiment_weighted_engagement', help='Target column name')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model file')
    parser.add_argument('--feature_columns', type=str, default=None, help='Path to feature columns JSON file')
    args = parser.parse_args()

    results = evaluate_on_test_data(
        model_name=args.model_name,
        test_data_path=args.test_data,
        target=args.target,
        model_path=args.model_path,
        feature_columns_path=args.feature_columns
    )
    print("Test set evaluation results:")
    print(results)
