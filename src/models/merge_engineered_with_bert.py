import pandas as pd

# Paths to your files
engineered_path = "outputs/engineered_data_filtered.csv"
features_path = "outputs/preprocessed_post_level_features.csv"
output_path = "outputs/engineered_data_filtered_with_bert.csv"

# Load both files
engineered = pd.read_csv(engineered_path)
features = pd.read_csv(features_path)

# Drop duplicate columns from features (except post_id and BERT columns)
bert_cols = [col for col in features.columns if col.startswith('caption_bert_emb_') or col.startswith('hashtags_bert_emb_')]
# Only keep post_id and BERT columns (exclude any non-numeric columns accidentally included)
features_bert = features[['post_id'] + bert_cols]

# Merge on post_id
merged = pd.merge(engineered, features_bert, on='post_id', how='inner')

# Remove any non-numeric columns except post_id and target columns
non_numeric_cols = merged.select_dtypes(include=['object']).columns.difference(['post_id', 'sentiment_weighted_engagement'])
merged = merged.drop(columns=non_numeric_cols)

# Double-check: ensure all columns except post_id and sentiment_weighted_engagement are numeric
for col in merged.columns:
    if col not in ['post_id', 'sentiment_weighted_engagement']:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Save merged file
merged.to_csv(output_path, index=False)
print(f"Merged file with BERT features and sentiment_weighted_engagement saved to {output_path}")
