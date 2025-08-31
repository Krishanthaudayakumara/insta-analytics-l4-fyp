import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Optionally: from pytorch_tabnet.tab_model import TabNetClassifier

def load_high_value_followers(owner_id):
    hv_path = f"outputs/high_value_followers_{owner_id}.json"
    if os.path.exists(hv_path):
        with open(hv_path, "r") as f:
            hv_followers = json.load(f)["high_value_followers"]
        return set(hv_followers.keys())
    return set()

def train_sentiment_weighted_engagement_model(
    owner_id=None,
    preprocessed_path="outputs/preprocessed_data.csv",
    engineered_path="outputs/engineered_data_filtered.csv",
    model_output_dir="outputs/models/",
    feature_columns_path="outputs/feature_columns.json",
    test_size=0.2,
    random_state=42,
    models=("Random Forest", "XGBoost", "LightGBM")
):
    os.makedirs(model_output_dir, exist_ok=True)
    # Load engineered filtered data
    df_eng = pd.read_csv(engineered_path)
    df_pre = pd.read_csv(preprocessed_path)
    # Merge on post_id (inner join to keep only posts present in both)
    if 'post_id' in df_eng.columns and 'post_id' in df_pre.columns:
        df = pd.merge(df_eng, df_pre, on='post_id', suffixes=('', '_pre'))
    else:
        df = df_eng.copy()
    if owner_id:
        hv_usernames = load_high_value_followers(owner_id)
        if "comment_owner_username" in df.columns:
            df = df[df["comment_owner_username"].isin(hv_usernames)]
    # Remove posts with zero comments
    if "num_comments" in df.columns:
        df = df[df["num_comments"] > 0]
    # Only keep numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"post_id", "sentiment_weighted_engagement", "owner_id", "timestamp", "likes_pre", "comments_count", "location_id", "user_id_encoded", "engagement_binary", "engagement_probability"}
    feature_columns = [col for col in numeric_cols if col not in exclude_cols]
    X = df[feature_columns].fillna(0)
    y = df["sentiment_weighted_engagement"].fillna(0)
    print("Feature columns used for training:", feature_columns)
    # Check for high correlation with target
    corr = df[feature_columns + ["sentiment_weighted_engagement"]].corr()["sentiment_weighted_engagement"].sort_values(ascending=False)
    print("Top 10 feature correlations with target:")
    print(corr.head(11))
    # Print a few rows of X and y
    print("Sample features (X):\n", X.head())
    print("Sample target (y):\n", y.head())
    # Save feature columns for evaluation
    with open(feature_columns_path, "w") as f:
        json.dump(feature_columns, f, indent=2)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = {}
    if "Random Forest" in models:
        model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        model_file = os.path.join(model_output_dir, f"rf_model_{owner_id}.pkl" if owner_id else "rf_model.pkl")
        joblib.dump(model, model_file)
        y_pred = model.predict(X_test)
        results['Random Forest'] = {
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    if "XGBoost" in models:
        model = XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        model_file = os.path.join(model_output_dir, f"xgb_model_{owner_id}.pkl" if owner_id else "xgb_model.pkl")
        joblib.dump(model, model_file)
        y_pred = model.predict(X_test)
        results['XGBoost'] = {
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    if "LightGBM" in models:
        model = LGBMRegressor(n_estimators=200, max_depth=7, learning_rate=0.05, num_leaves=31, random_state=random_state)
        model.fit(X_train, y_train)
        model_file = os.path.join(model_output_dir, f"lgb_model_{owner_id}.pkl" if owner_id else "lgb_model.pkl")
        joblib.dump(model, model_file)
        y_pred = model.predict(X_test)
        results['LightGBM'] = {
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
    print("Training complete. Regression Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train models on sentiment_weighted_engagement.")
    parser.add_argument('--owner_id', type=str, default=None, help='Instagram account owner_id (for high-value follower filtering)')
    parser.add_argument('--preprocessed', type=str, default='outputs/preprocessed_data.csv', help='Path to preprocessed data')
    parser.add_argument('--engineered', type=str, default='outputs/engineered_data_filtered.csv', help='Path to engineered filtered data')
    parser.add_argument('--model_dir', type=str, default='outputs/models/', help='Directory to save model')
    parser.add_argument('--feature_columns', type=str, default='outputs/feature_columns.json', help='Path to save feature columns')
    parser.add_argument('--models', type=str, nargs='+', default=["Random Forest", "XGBoost", "LightGBM"], help='Models to train (Random Forest, XGBoost, LightGBM)')
    args = parser.parse_args()
    train_sentiment_weighted_engagement_model(
        owner_id=args.owner_id,
        preprocessed_path=args.preprocessed,
        engineered_path=args.engineered,
        model_output_dir=args.model_dir,
        feature_columns_path=args.feature_columns,
        models=args.models
    )
