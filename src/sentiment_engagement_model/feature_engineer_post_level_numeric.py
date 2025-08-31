import pandas as pd
import argparse
from collections import Counter
import re
from sentence_transformers import SentenceTransformer
import numpy as np

def extract_top_hashtags(df, col='hashtags', top_n=10):
    all_tags = df[col].dropna().str.replace('#', '').str.lower().str.replace(' ', '')
    tags_split = all_tags.str.split(',')
    tag_counts = Counter(tag for tags in tags_split for tag in tags if tag)
    top_tags = [tag for tag, _ in tag_counts.most_common(top_n)]
    for tag in top_tags:
        df[f'hashtag_{tag}'] = df[col].str.contains(tag, case=False, na=False).astype(int)
    return df, top_tags

def count_emojis(text):
    emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)
    return len(emoji_pattern.findall(str(text)))

def add_bert_embeddings(df, text_col='caption', model_name='all-MiniLM-L6-v2', prefix='caption_bert'):
    """Add BERT embeddings for a text column as new columns."""
    model = SentenceTransformer(model_name)
    texts = df[text_col].fillna("").astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    emb_dim = embeddings.shape[1]
    for i in range(emb_dim):
        df[f'{prefix}_emb_{i}'] = embeddings[:, i]
    return df

def process_post_level_data(input_path, output_path, top_n_hashtags=10, add_bert_caption=True, bert_model_name='all-MiniLM-L6-v2', add_bert_hashtags=False):
    df = pd.read_csv(input_path)

    # Drop unnecessary columns
    drop_cols = [
        'shortcode', 'username', 'location_id', 'location', 'is_private', 'is_verified', 'mentions'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Hashtag features: top-N one-hot
    if 'hashtags' in df.columns:
        df, top_tags = extract_top_hashtags(df, 'hashtags', top_n=top_n_hashtags)
        print(f"Top {top_n_hashtags} hashtags: {top_tags}")

    # One-hot encode Category and media_type
    for col in ['Category', 'media_type']:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

    # Caption features
    if 'caption' in df.columns:
        df['caption_length'] = df['caption'].fillna('').apply(len)
        df['caption_num_emojis'] = df['caption'].apply(count_emojis)
        # Add BERT embeddings for caption text if enabled
        if add_bert_caption:
            df = add_bert_embeddings(df, text_col='caption', model_name=bert_model_name, prefix='caption_bert')
        df = df.drop(columns=['caption'])

    # Hashtag BERT embeddings (optional)
    if add_bert_hashtags and 'hashtags' in df.columns:
        df = add_bert_embeddings(df, text_col='hashtags', model_name=bert_model_name, prefix='hashtags_bert')

    # Drop raw hashtags column (after feature extraction)
    if 'hashtags' in df.columns:
        df = df.drop(columns=['hashtags'])

    df.to_csv(output_path, index=False)
    print(f"Processed post-level data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineer post-level data for ML.")
    parser.add_argument('--input', type=str, required=True, help='Path to post-level CSV')
    parser.add_argument('--output', type=str, required=True, help='Path to save processed CSV')
    parser.add_argument('--top_n_hashtags', type=int, default=10, help='Number of top hashtags to one-hot encode')
    parser.add_argument('--add_bert_caption', action='store_true', help='Add BERT embeddings for captions')
    parser.add_argument('--add_bert_hashtags', action='store_true', help='Add BERT embeddings for hashtags')
    parser.add_argument('--bert_model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    args = parser.parse_args()
    process_post_level_data(
        args.input, args.output,
        top_n_hashtags=args.top_n_hashtags,
        add_bert_caption=args.add_bert_caption,
        bert_model_name=args.bert_model_name,
        add_bert_hashtags=args.add_bert_hashtags
    )