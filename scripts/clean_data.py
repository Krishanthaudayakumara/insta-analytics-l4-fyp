import pandas as pd
import logging
import os
import re

# Configure logging
logging.basicConfig(filename='clean_data.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def clean_and_deduplicate(input_csv, output_csv):
    """Cleans, deduplicates, and preprocesses the merged data."""

    logging.info("Starting data cleaning and deduplication")

    try:
        df = pd.read_csv(input_csv, encoding='utf-8', engine='python')
        logging.info(f"Loaded data: {df.shape[0]} rows")

    except FileNotFoundError as e:
        logging.error(f"Could not find input CSV file: {e}")
        return

    # --- Filter out rows with 0 comments_count ---
    df = df[df['comments_count'] > 0]
    logging.info(f"Removed rows with 0 comments_count. Remaining rows: {df.shape[0]}")

    # --- Filter out rows with empty caption or hashtags ---
    df = df.dropna(subset=['caption', 'hashtags'])
    logging.info(f"Removed rows with empty caption or hashtags. Remaining rows: {df.shape[0]}")


    # --- Clean captions (remove hashtags and extra whitespace) ---
    def clean_caption(caption):
        caption_without_hashtags = re.sub(r"#[\w-]+", "", caption) # remove hastags
        cleaned_caption = ' '.join(caption_without_hashtags.split()) # remove extra whitespace
        return cleaned_caption
    df['caption'] = df['caption'].astype(str).apply(clean_caption) # apply the function



    # --- Drop Duplicates ---
    df.drop_duplicates(inplace=True)
    logging.info(f"Removed duplicate rows. Remaining rows: {df.shape[0]}")


    # --- Remove 'Username' column ---
    if 'Username' in df.columns:
        df.drop(columns=['Username'], inplace=True)
        logging.info("Removed 'Username' column")

    # --- Feature engineering: caption_length (number of words in original caption) ---
    if 'caption' in df.columns:
        df['caption_length'] = df['caption'].astype(str).apply(lambda x: len(x.split()))
        logging.info("Added caption_length feature.")

    # --- Feature engineering: ensure hashtags is a string and not NaN or empty ---
    if 'hashtags' in df.columns:
        df['hashtags'] = df['hashtags'].fillna('').astype(str)
        # If any are 'nan' or empty, set to ''
        df['hashtags'] = df['hashtags'].replace('nan', '').replace('None', '')
        logging.info("Ensured hashtags column is string and not NaN or None.")
    else:
        df['hashtags'] = ''
        logging.info("Created empty hashtags column.")

    # --- Feature engineering: engagement rate ---
    if 'likes' in df.columns and '#Followers' in df.columns:
        df['engagement_rate'] = (df['likes'] + df['comments_count']) / (df['#Followers'] + 1)
        logging.info("Added engagement_rate feature.")

    # --- Aggregate hashtags per post into a comma-separated string ---
    if 'post_id' in df.columns and 'hashtags' in df.columns:
        hashtags_agg = df.groupby('post_id')['hashtags'].apply(lambda x: ','.join(sorted(set([h for h in x if pd.notnull(h) and str(h).strip()])))).reset_index()
        hashtags_agg.rename(columns={'hashtags': 'hashtags_agg'}, inplace=True)
        df = df.merge(hashtags_agg, on='post_id', how='left')
        logging.info("Aggregated hashtags per post into 'hashtags_agg' column.")

    # --- Save Cleaned Data ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    logging.info(f"Saved cleaned data to {output_csv}")


# --- Main execution ---
input_csv_path = "/home/krishantha/Github/fyp-l4/data/processed_data/merged_user_post_data.csv"
output_cleaned_csv_path = "/home/krishantha/Github/fyp-l4/data/processed_data/cleaned_merged_user_post_data.csv"


clean_and_deduplicate(input_csv_path, output_cleaned_csv_path)