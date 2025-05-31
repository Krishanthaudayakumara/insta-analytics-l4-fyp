from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging
import numpy as np

def run(df):
    df = df.fillna(0)
    X_cols = []
    if 'likes' in df.columns:
        X_cols.append('likes')
    if 'shares' in df.columns:
        X_cols.append('shares')
    if 'caption_length' in df.columns:
        X_cols.append('caption_length')
    if 'num_hashtags' in df.columns:
        X_cols.append('num_hashtags')
    if 'engagement_rate' in df.columns:
        X_cols.append('engagement_rate')
    if not X_cols:
        logging.error("No valid features for engagement prediction.")
        raise ValueError("No valid features for engagement prediction.")
    X = df[X_cols]
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