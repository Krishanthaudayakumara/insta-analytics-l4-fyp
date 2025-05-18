import json
import os
import pandas as pd
import re
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(filename='process_data_posts.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def extract_post_data(file_path):
    """Extracts data from a .info file, handling missing data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            post_data = json.load(f)

        extracted = {
            'post_id': post_data['id'],
            'owner_id': post_data['owner']['id'],
            'timestamp': post_data['taken_at_timestamp'],
            'likes': post_data['edge_media_preview_like']['count'],
            'comments_count': post_data.get('edge_media_to_parent_comment', {}).get('count', 0),
            'comments': [],
            'caption': "",
            'hashtags': [],
            'location_id': post_data['location']['id'] if post_data.get('location') else None
        }

        if post_data.get('edge_media_to_caption') and post_data['edge_media_to_caption']['edges']:
            extracted['caption'] = post_data['edge_media_to_caption']['edges'][0]['node']['text']
            extracted['hashtags'] = [match.group(0)[1:] for match in re.finditer(r"#[\w-]+", extracted['caption'])]

        if post_data.get('edge_media_to_parent_comment') and post_data['edge_media_to_parent_comment']['edges']:
            for comment in post_data['edge_media_to_parent_comment']['edges']:
                extracted['comments'].append({
                    'comment_text': comment['node']['text'],
                    'comment_created_at': comment['node']['created_at'],
                    'comment_owner_username': comment['node']['owner']['username'],
                    'comment_likes': comment['node']['edge_liked_by']['count']
                })

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

    return extracted

def process_all_data(clustered_data_path, max_users=None, chunk_size=1000):  # Removed influencers_csv_path
    """Processes post data, handles mismatches, with user limit, chunking, logging, progress bars, and empty DataFrame checks."""
    logging.info("Starting data processing")

    all_posts = []
    user_count = 0
    for root, dirs, files in os.walk(clustered_data_path):
        if max_users and user_count >= max_users:
            break

        for dir in dirs:
            if max_users and user_count >= max_users:
                break

            username = dir
            user_count += 1

            for filename in tqdm(os.listdir(os.path.join(root, dir)), desc=f"Processing posts from user {username}"):
                if filename.endswith(".info"):
                    file_path = os.path.join(root, dir, filename)
                    post_data = extract_post_data(file_path)
                    if post_data:
                        post_data['username'] = username
                        all_posts.append(post_data)

    logging.info(f"Extracted data from {len(all_posts)} posts from {user_count} users")

    posts_df = pd.DataFrame(all_posts)
    posts_df['username'] = posts_df['username'].astype(str).str.strip().str.lower()


    # --- Chunking (No merge with influencers_df)---
    final_df = pd.DataFrame() # Initialize outside loop
    for i in tqdm(range(0, len(all_posts), chunk_size), desc="Processing data in chunks"):
        chunk = all_posts[i:i + chunk_size]
        posts_chunk_df = pd.DataFrame(chunk)

        if not posts_chunk_df.empty:
            posts_chunk_df = posts_chunk_df.explode('hashtags', ignore_index=True)
            posts_chunk_df = posts_chunk_df.explode('comments', ignore_index=True) # Explode comments as well


            if not posts_chunk_df['comments'].isnull().all():
                 posts_chunk_df = pd.concat([posts_chunk_df.drop(['comments'], axis=1), posts_chunk_df['comments'].apply(pd.Series)], axis=1)

        final_df = pd.concat([final_df, posts_chunk_df], ignore_index=True)

    if final_df.empty:
        logging.warning("Final DataFrame is empty. No data processed.")


    logging.info(f"Finished processing. Final DataFrame shape: {final_df.shape}")
    return final_df

# --- Main execution ---
clustered_data_path = "/home/krishantha/Github/fyp-l4/data/clustered_data/followers_10000_to_50000"
# influencers_csv_path = "/home/krishantha/Github/fyp-l4/data/clustered_data/influencers.csv"  # No longer needed in this version
max_users_to_process = 20  # Set your desired limit

final_df = process_all_data(clustered_data_path, max_users=max_users_to_process)  # Removed influencers_csv_path argument

if final_df is not None:
    print(final_df.head())
    final_df.to_csv("/home/krishantha/Github/fyp-l4/data/processed_data/user_post_data.csv", index=False)