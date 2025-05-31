import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import seaborn as sns

def run(df):
    # Use 'timestamp' column if 'Date' is not present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        date_col = 'Date'
    elif 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        date_col = 'Date'
    else:
        logging.error("No 'Date' or 'timestamp' column found in DataFrame.")
        raise KeyError("No 'Date' or 'timestamp' column found in DataFrame.")
    df = df.dropna(subset=['Date'])
    # Use available engagement columns
    value_cols = [col for col in ['Likes', 'Comments', 'Shares', 'likes', 'comments_count', 'engagement_rate'] if col in df.columns]
    if not value_cols:
        logging.error("No engagement columns found for plotting.")
        raise KeyError("No engagement columns found for plotting.")
    os.makedirs('outputs', exist_ok=True)
    # Daily engagement trends
    daily_engagement = df.groupby(df['Date'].dt.date)[value_cols].mean()
    daily_engagement.plot(figsize=(10,6))
    plt.title("Daily Engagement Trends")
    plt.xlabel("Date")
    plt.ylabel("Average Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/engagement_trends.png")
    plt.close()
    logging.info("Saved engagement trends plot to outputs/engagement_trends.png")
    # Engagement by hour
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
        plt.figure(figsize=(10,6))
        sns.boxplot(x='hour', y='likes', data=df)
        plt.title('Likes by Hour of Day')
        plt.savefig('outputs/likes_by_hour.png')
        plt.close()
        logging.info("Saved likes by hour plot to outputs/likes_by_hour.png")
    # Engagement by cluster
    if 'user_cluster_kmeans' in df.columns:
        plt.figure(figsize=(10,6))
        sns.boxplot(x='user_cluster_kmeans', y='likes', data=df)
        plt.title('Likes by User Cluster (KMeans)')
        plt.savefig('outputs/likes_by_cluster.png')
        plt.close()
        logging.info("Saved likes by cluster plot to outputs/likes_by_cluster.png")
