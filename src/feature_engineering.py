"""
Feature Engineering Pipeline for Instagram Engagement Modeling
Aggregates comment-level data to post-level, merges sentiment, and computes features for modeling.
"""
import pandas as pd
import numpy as np
import json
import os

def run_feature_engineering(
    preprocessed_path="outputs/preprocessed_data.csv",
    sentiment_path="outputs/sentiment_scores.json",
    output_path="outputs/engineered_data.csv"
):
    # Load preprocessed data
    df = pd.read_csv(preprocessed_path)
    # Load sentiment scores
    if os.path.exists(sentiment_path):
        with open(sentiment_path, "r") as f:
            sentiment_scores = json.load(f)
    else:
        sentiment_scores = {}
    # Merge sentiment scores into df
    df['sentiment_positive'] = 0.33
    df['sentiment_negative'] = 0.33
    df['sentiment_neutral'] = 0.34
    df['avg_comment_sentiment'] = 0.0
    for idx, row in df.iterrows():
        comment_key = f"comment_{idx}_{row['comment_owner_username']}"
        if comment_key in sentiment_scores:
            s = sentiment_scores[comment_key]
            df.at[idx, 'sentiment_positive'] = s.get('positive', 0.33)
            df.at[idx, 'sentiment_negative'] = s.get('negative', 0.33)
            df.at[idx, 'sentiment_neutral'] = s.get('neutral', 0.34)
            df.at[idx, 'avg_comment_sentiment'] = (
                s.get('positive', 0.33) - s.get('negative', 0.33)
            )
    # Aggregate to post level
    agg_dict = {
        'likes': 'first',
        'comments_count': 'first',
        'comment_likes': 'sum',
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'sentiment_neutral': 'mean',
        'avg_comment_sentiment': 'mean',
        'comment_owner_username': 'count',
    }
    if 'post_id' in df.columns:
        post_df = df.groupby('post_id').agg(agg_dict).reset_index()
    else:
        post_df = df.copy()
    post_df = post_df.rename(columns={'comment_owner_username': 'num_comments', 'comment_likes': 'comment_likes_sum'})
    # Compute sentiment_weighted_engagement
    post_df['sentiment_weighted_engagement'] = post_df['comment_likes_sum'].fillna(0) * post_df['avg_comment_sentiment'].fillna(0)
    # Save engineered data
    post_df.to_csv(output_path, index=False)
    print(f"[INFO] Feature engineering complete. Output: {output_path}")
    return post_df

if __name__ == "__main__":
    run_feature_engineering()
