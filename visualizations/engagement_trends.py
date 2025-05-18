import pandas as pd
import matplotlib.pyplot as plt

def run(df):
    # Use 'timestamp' column if 'Date' is not present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        date_col = 'Date'
    elif 'timestamp' in df.columns:
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        date_col = 'Date'
    else:
        raise KeyError("No 'Date' or 'timestamp' column found in DataFrame.")
    df = df.dropna(subset=['Date'])
    # Use available engagement columns
    value_cols = [col for col in ['Likes', 'Comments', 'Shares', 'likes', 'comments_count'] if col in df.columns]
    if not value_cols:
        raise KeyError("No engagement columns found for plotting.")
    daily_engagement = df.groupby(df['Date'].dt.date)[value_cols].mean()

    daily_engagement.plot(figsize=(10,6))
    plt.title("Daily Engagement Trends")
    plt.xlabel("Date")
    plt.ylabel("Average Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/engagement_trends.png")
    plt.close()
