from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import logging
import numpy as np
import pandas as pd
import joblib

def run(df):
    df = df.fillna(0)
    # --- Feature Engineering: Use all relevant features ---
    X_cols = []
    # Fixed: close the list and check all relevant columns
    for col in [
        'likes', 'shares', 'caption_length', 'num_hashtags', 'engagement_rate',
        'follower_adjusted_likes', 'follower_adjusted_comments',
        'caption_sentiment_vader', 'caption_sentiment', 'hour_of_day', 'day_of_week',
        '#Followers', '#Followees', '#Posts', 'user_cluster_k', 'user_cluster_agglom'
    ]:
        if col in df.columns:
            X_cols.append(col)
    if not X_cols:
        logging.error("No valid features for engagement prediction.")
        raise ValueError("No valid features for engagement prediction.")
    X = df[X_cols]
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Comments'] if 'Comments' in df.columns else df['comments_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    # Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    pred_lr = model_lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, pred_lr)
    results['LinearRegression'] = mse_lr
    # Ridge Regression
    model_ridge = Ridge()
    model_ridge.fit(X_train, y_train)
    pred_ridge = model_ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_test, pred_ridge)
    results['Ridge'] = mse_ridge
    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, pred_rf)
    results['RandomForest'] = mse_rf
    logging.info(f"Engagement Prediction MSEs: {results}")
    print("Model MSEs:", results)
    # Save trained models for download
    joblib.dump(model_lr, 'outputs/model_linear_regression.joblib')
    joblib.dump(model_ridge, 'outputs/model_ridge.joblib')
    joblib.dump(model_rf, 'outputs/model_random_forest.joblib')
    # Optionally, save feature columns for later use
    joblib.dump(X_cols, 'outputs/model_features.joblib')
    return df

def train_and_save_like_comment_models(df):
    df = df.fillna(0)
    # --- Feature selection ---
    feature_cols = [
        'likes', 'shares', 'caption_length', 'num_hashtags', 'engagement_rate',
        'follower_adjusted_likes', 'follower_adjusted_comments',
        'caption_sentiment_vader', 'caption_sentiment', 'hour_of_day', 'day_of_week',
        '#Followers', '#Followees', '#Posts', 'user_cluster_k', 'user_cluster_agglom'
    ]
    # Likes models: exclude 'likes', 'comments', 'comments_count' from X
    X_cols_likes = [col for col in feature_cols if col in df.columns and col.lower() not in ['likes', 'comments', 'comments_count']]
    X_likes = df[X_cols_likes].apply(pd.to_numeric, errors='coerce').fillna(0)
    # Comments models: exclude 'comments', 'comments_count', 'likes' from X
    X_cols_comments = [col for col in feature_cols if col in df.columns and col.lower() not in ['comments', 'comments_count', 'likes']]
    X_comments = df[X_cols_comments].apply(pd.to_numeric, errors='coerce').fillna(0)
    # --- Likes ---
    if 'Likes' in df.columns:
        y_likes = df['Likes']
    elif 'likes' in df.columns:
        y_likes = df['likes']
    else:
        y_likes = None
    # --- Comments ---
    if 'Comments' in df.columns:
        y_comments = df['Comments']
    elif 'comments_count' in df.columns:
        y_comments = df['comments_count']
    else:
        y_comments = None
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    # Likes models
    if y_likes is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_likes, y_likes, test_size=0.2, random_state=42)
        model_lr_likes = LinearRegression().fit(X_train, y_train)
        model_ridge_likes = Ridge().fit(X_train, y_train)
        model_rf_likes = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        joblib.dump(model_lr_likes, 'outputs/model_linear_regression_likes.joblib')
        joblib.dump(model_ridge_likes, 'outputs/model_ridge_likes.joblib')
        joblib.dump(model_rf_likes, 'outputs/model_random_forest_likes.joblib')
        joblib.dump(X_cols_likes, 'outputs/model_features_likes.joblib')
    # Comments models
    if y_comments is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_comments, y_comments, test_size=0.2, random_state=42)
        model_lr_comments = LinearRegression().fit(X_train, y_train)
        model_ridge_comments = Ridge().fit(X_train, y_train)
        model_rf_comments = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        joblib.dump(model_lr_comments, 'outputs/model_linear_regression_comments.joblib')
        joblib.dump(model_ridge_comments, 'outputs/model_ridge_comments.joblib')
        joblib.dump(model_rf_comments, 'outputs/model_random_forest_comments.joblib')
        joblib.dump(X_cols_comments, 'outputs/model_features_comments.joblib')
    return True

# Add CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and save like/comment ML models.")
    parser.add_argument('--data', type=str, default='data/processed_data/cleaned_merged_user_post_data.csv', help='Path to input data CSV')
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    train_and_save_like_comment_models(df)
    print("Like/comment models trained and saved.")