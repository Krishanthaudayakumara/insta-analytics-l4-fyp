from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run(df):
    # Use lowercase column names if available, fallback to original if not
    col_map = {
        'Likes': 'likes',
        'Comments': 'comments_count',
        'Shares': None  # No shares column in your data
    }
    features = []
    for col in ['Likes', 'Comments', 'Shares']:
        if col in df.columns:
            features.append(col)
        elif col_map[col] and col_map[col] in df.columns:
            features.append(col_map[col])
    if not features:
        raise ValueError("No valid engagement columns found for clustering.")
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=0)
    df['user_cluster'] = kmeans.fit_predict(X_scaled)
    return df
