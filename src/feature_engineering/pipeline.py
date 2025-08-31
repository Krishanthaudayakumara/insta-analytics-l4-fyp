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
            sentiment_data = json.load(f)
        sentiment_scores = sentiment_data["sentiment_scores"] if "sentiment_scores" in sentiment_data else sentiment_data
    else:
        sentiment_scores = {}
    # Convert sentiment_scores to DataFrame for merging
    sentiment_df = pd.DataFrame.from_dict(sentiment_scores, orient='index')
    # Try to extract a unique key for merging
    merge_cols = []
    if 'comment_id' in df.columns and 'comment_id' in sentiment_df.columns:
        sentiment_df = sentiment_df.reset_index(drop=True)
        merge_cols = ['comment_id']
    elif all(col in df.columns for col in ['post_id', 'comment_owner_username', 'comment_text']) and \
         all(col in sentiment_df.columns for col in ['post_id', 'comment_owner_username', 'comment_text']):
        merge_cols = ['post_id', 'comment_owner_username', 'comment_text']
    else:
        # Fallback: try to merge by index (row order)
        sentiment_df = sentiment_df.reset_index().rename(columns={'index': 'row_idx'})
        df = df.reset_index().rename(columns={'index': 'row_idx'})
        merge_cols = ['row_idx']
    # Merge
    df_merged = df.merge(sentiment_df, on=merge_cols, how='left', suffixes=('', '_sent'))
    # Use sentiment columns from sentiment_df if available, else fallback
    df_merged['sentiment_positive'] = df_merged.get('positive', 0.33).fillna(0.33)
    df_merged['sentiment_negative'] = df_merged.get('negative', 0.33).fillna(0.33)
    df_merged['sentiment_neutral'] = df_merged.get('neutral', 0.34).fillna(0.34)
    df_merged['avg_comment_sentiment'] = (
        df_merged['sentiment_positive'] - df_merged['sentiment_negative']
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
    if 'post_id' in df_merged.columns:
        post_df = df_merged.groupby('post_id').agg(agg_dict).reset_index()
    else:
        post_df = df_merged.copy()
    post_df = post_df.rename(columns={'comment_owner_username': 'num_comments', 'comment_likes': 'comment_likes_sum'})
    # Remove any duplicate or leftover comment_owner_username or comments_count columns
    for col in ['comment_owner_username', 'comments_count']:
        if col in post_df.columns and col != 'num_comments':
            post_df = post_df.drop(columns=[col])
    # Compute sentiment_weighted_engagement using avg_comment_sentiment * num_comments (not comment_likes_sum)
    post_df['sentiment_weighted_engagement'] = post_df['num_comments'].fillna(0) * post_df['avg_comment_sentiment'].fillna(0)
    # Save engineered data
    post_df.to_csv(output_path, index=False)
    print(f"[INFO] Feature engineering complete. Output: {output_path}")
    return post_df

def filter_posts_with_comments(input_path="outputs/engineered_data.csv", output_path="outputs/engineered_data_filtered.csv"):
    """Filter out posts with zero comments from engineered data."""
    df = pd.read_csv(input_path)
    if 'num_comments' in df.columns:
        filtered_df = df[df['num_comments'] > 0].copy()
    else:
        filtered_df = df.copy()
    filtered_df.to_csv(output_path, index=False)
    print(f"[INFO] Filtered data saved to: {output_path}")
    return filtered_df

if __name__ == "__main__":
    run_feature_engineering()
