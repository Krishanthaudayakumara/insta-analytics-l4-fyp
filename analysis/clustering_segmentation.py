from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging
import numpy as np

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
        logging.error("No valid engagement columns found for clustering.")
        raise ValueError("No valid engagement columns found for clustering.")
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Use a sample for silhouette analysis and AgglomerativeClustering to avoid OOM
    sample_size = min(2000, len(df))
    X_sample = X_scaled[:sample_size]
    # Silhouette analysis for optimal k (on sample)
    best_k = 2
    best_score = -1
    for k in range(2, 6):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, labels)
        if score > best_score:
            best_score = score
            best_k = k
    # KMeans on full data
    kmeans = KMeans(n_clusters=best_k, random_state=0)
    df['user_cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    # Agglomerative only on sample (optional, for research)
    try:
        agg = AgglomerativeClustering(n_clusters=best_k)
        sample_labels = agg.fit_predict(X_sample)
        # Assign sample labels to a new column (rest as -1)
        df['user_cluster_hier'] = -1
        df.loc[df.index[:sample_size], 'user_cluster_hier'] = sample_labels
    except Exception as e:
        logging.warning(f"AgglomerativeClustering failed or skipped: {e}")
    logging.info(f"Clustering complete. KMeans clusters: {best_k}, Silhouette: {best_score:.2f}")
    return df
