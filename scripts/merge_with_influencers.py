import pandas as pd
import logging
from tqdm import tqdm  # For progress bar (optional, but recommended)

# Configure logging
logging.basicConfig(filename='merge_with_influencers.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def merge_with_influencers(posts_csv_path, influencers_csv_path, output_csv_path):
    """Merges post data with influencer data based on username."""

    logging.info("Starting merge process")

    try:
        posts_df = pd.read_csv(posts_csv_path, encoding='utf-8', engine='python')
        influencers_df = pd.read_csv(influencers_csv_path, sep=",", encoding='utf-8', engine='python')
        influencers_df.dropna(inplace=True)  # Drop rows with missing values in influencers_df


        logging.info(f"Loaded posts data: {posts_df.shape[0]} rows")
        logging.info(f"Loaded and cleaned influencers data: {influencers_df.shape[0]} rows")

    except FileNotFoundError as e:
        logging.error(f"Could not find CSV file: {e}")
        return

    # --- Enhanced Cleaning and Conversion for Merge ---
    influencers_df['Username'] = influencers_df['Username'].astype(str).str.strip().str.lower()
    posts_df['username'] = posts_df['username'].astype(str).str.strip().str.lower()
    

    # --- Merge ---
    try:
        merged_df = pd.merge(posts_df, influencers_df, left_on='username', right_on='Username', how='left')
        # Feature engineering: follower-adjusted engagement
        if '#Followers' in merged_df.columns:
            merged_df['follower_adjusted_likes'] = merged_df['likes'] / (merged_df['#Followers'] + 1)
            merged_df['follower_adjusted_comments'] = merged_df['comments_count'] / (merged_df['#Followers'] + 1)
        logging.info(f"Merged data successfully. Merged DataFrame shape: {merged_df.shape}")




    except KeyError as e:
      logging.error(f"Key error during merge: {e}") #Log if key is not found



    # --- Save Merged Data ---
    try:
        merged_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        logging.info(f"Saved merged data to {output_csv_path}")
    except Exception as e: # Catch any issues during save
        logging.error(f"Error saving merged data: {e}")




# --- Main execution ---
posts_csv_path = "/home/krishantha/Github/fyp-l4/data/processed_data/user_post_data.csv"
influencers_csv_path = "/home/krishantha/Github/fyp-l4/data/clustered_data/influencers.csv"
output_merged_csv_path = "/home/krishantha/Github/fyp-l4/data/processed_data/merged_user_post_data.csv"

merge_with_influencers(posts_csv_path, influencers_csv_path, output_merged_csv_path)