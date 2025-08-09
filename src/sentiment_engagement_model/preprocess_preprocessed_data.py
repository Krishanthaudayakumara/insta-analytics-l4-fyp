import pandas as pd
import argparse
import numpy as np

def aggregate_preprocessed_to_post_level(input_path, output_path):
    """
    Aggregates comment-level preprocessed data to post-level, keeping only post-level features.
    """
    df = pd.read_csv(input_path)
    # Define post-level columns to keep (add more as needed)
    post_level_cols = [
        'post_id', 'owner_id', 'timestamp', 'likes', 'comments_count', 'caption', 'hashtags',
        'location_id', 'media_type', 'username', 'shortcode', 'location', 'is_private',
        'is_verified', 'mentions', 'Category', '#Followers', '#Followees', '#Posts',
        'caption_length', 'num_hashtags', 'has_mention', 'has_url', 'follower_adjusted_likes',
        'follower_adjusted_comments', 'engagement_rate', 'hashtags_agg', 'engagement_frequency',
        'influence_score', 'content_interaction', 'comment_engagement_ratio', 'comment_length',
        'has_emoji', 'category_beauty', 'category_family', 'category_fashion', 'category_fitness',
        'category_food', 'category_pet', 'category_travel', 'user_id_encoded', 'engagement_binary',
        'engagement_probability'
    ]
    # Only keep columns that exist in the input
    post_level_cols = [col for col in post_level_cols if col in df.columns]
    # Drop duplicates, keeping the first occurrence for each post_id
    df_post = df[post_level_cols].drop_duplicates(subset=['post_id'])

    # Do NOT add BERT embeddings here; only aggregate to post-level.
    # BERT embedding extraction is handled in the next step (feature_engineer_post_level_numeric.py)

    df_post.to_csv(output_path, index=False)
    print(f"Aggregated post-level data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate preprocessed comment-level data to post-level.")
    parser.add_argument('--input', type=str, required=True, help='Path to preprocessed_data.csv')
    parser.add_argument('--output', type=str, required=True, help='Path to save post-level CSV')
    args = parser.parse_args()
    aggregate_preprocessed_to_post_level(args.input, args.output)
