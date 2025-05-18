from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def run(df):
    df = df.fillna(0)
    # Use available columns: 'likes', 'comments_count' (no 'Shares' in your data)
    X_cols = []
    if 'likes' in df.columns:
        X_cols.append('likes')
    if 'shares' in df.columns:
        X_cols.append('shares')
    # Fallback: if no 'shares', just use 'likes'
    if not X_cols:
        raise ValueError("No valid features for engagement prediction.")
    X = df[X_cols]
    # Use 'comments_count' as target if 'Comments' not present
    y = df['Comments'] if 'Comments' in df.columns else df['comments_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"Engagement Prediction MSE: {mse:.2f}")