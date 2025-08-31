import pandas as pd
import json
import argparse
import os
import sys

def load_high_value_followers(owner_id):
    """Load high-value follower usernames for a given owner_id."""
    hv_path = f"outputs/high_value_followers_{owner_id}.json"
    if os.path.exists(hv_path):
        with open(hv_path, "r") as f:
            hv_followers = json.load(f)["high_value_followers"]
        return set(hv_followers.keys())
    return set()

def filter_comments_by_high_value(df_comments, owner_id):
    """Filter comments to only those made by high-value followers."""
    hv_usernames = load_high_value_followers(owner_id)
    if "comment_owner_username" in df_comments.columns:
        return df_comments[df_comments["comment_owner_username"].isin(hv_usernames)]
    return df_comments

def aggregate_high_value_features(df_comments, post_id_col="post_id"):
    """Aggregate high-value follower comment features at the post level."""
    agg_funcs = {
        "comment_id": "count",
        "comment_likes": "sum",
        "comment_sentiment": "mean",
        "comment_length": "mean"
    }
    missing_cols = [k for k in agg_funcs if k not in df_comments.columns]
    if missing_cols:
        print(f"Warning: Missing columns for aggregation: {missing_cols}", file=sys.stderr)
    agg_df = df_comments.groupby(post_id_col).agg({
        k: v for k, v in agg_funcs.items() if k in df_comments.columns
    }).reset_index()
    agg_df = agg_df.rename(columns={
        "comment_id": "num_comments_high_value",
        "comment_likes": "comment_likes_sum_high_value",
        "comment_sentiment": "avg_comment_sentiment_high_value",
        "comment_length": "avg_comment_length_high_value"
    })
    return agg_df

def main(post_level_path, comment_level_path, owner_id, output_path):
    """Main pipeline to create post-level features for high-value follower approach."""
    # Check input files exist
    for path in [post_level_path, comment_level_path]:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)
    # Load post-level features
    df_post = pd.read_csv(post_level_path)
    # Load comment-level features
    df_comments = pd.read_csv(comment_level_path)
    # Ensure post_id type consistency
    if 'post_id' in df_post.columns and 'post_id' in df_comments.columns:
        df_post['post_id'] = df_post['post_id'].astype(str)
        df_comments['post_id'] = df_comments['post_id'].astype(str)
    # Filter comments to only high-value followers
    df_hv_comments = filter_comments_by_high_value(df_comments, owner_id)
    # Aggregate high-value follower features at post level
    agg_hv = aggregate_high_value_features(df_hv_comments)
    # Merge with post-level features
    df_merged = pd.merge(df_post, agg_hv, on="post_id", how="left")
    # Ensure all expected columns exist, fill NaNs for posts with no high-value comments
    expected_cols = [
        "num_comments_high_value",
        "comment_likes_sum_high_value",
        "avg_comment_sentiment_high_value",
        "avg_comment_length_high_value"
    ]
    for col in expected_cols:
        if col not in df_merged.columns:
            df_merged[col] = 0
        else:
            df_merged[col] = df_merged[col].fillna(0)
    # Add binary indicator
    df_merged["has_high_value_follower_comment"] = (df_merged["num_comments_high_value"] > 0).astype(int)
    # Optional: reorder columns for clarity
    hv_cols = expected_cols + ["has_high_value_follower_comment"]
    other_cols = [c for c in df_merged.columns if c not in hv_cols]
    df_merged = df_merged[other_cols + hv_cols]
    # Save to output
    df_merged.to_csv(output_path, index=False)
    print(f"High-value follower post-level features saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create post-level features for high-value follower approach.")
    parser.add_argument('--post_level', type=str, required=True, help='Path to post-level features CSV')
    parser.add_argument('--comment_level', type=str, required=True, help='Path to comment-level features CSV')
    parser.add_argument('--owner_id', type=str, required=True, help='Instagram account owner_id')
    parser.add_argument('--output', type=str, required=True, help='Path to save output CSV')
    args = parser.parse_args()
    main(args.post_level, args.comment_level, args.owner_id, args.output)
