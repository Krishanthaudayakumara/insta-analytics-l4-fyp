import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def run(df):
    # Use lowercase column names if uppercase not present
    likes_col = 'Likes' if 'Likes' in df.columns else 'likes'
    comments_col = 'Comments' if 'Comments' in df.columns else 'comments_count'
    post_id_col = 'Post ID' if 'Post ID' in df.columns else 'post_id'
    caption_col = 'Caption' if 'Caption' in df.columns else 'caption'
    # Top posts by engagement
    top_posts = df.sort_values(by=[likes_col, comments_col], ascending=False).head(5)
    logging.info("Top 5 Recommended Posts (by engagement):")
    logging.info(top_posts[[post_id_col, caption_col, likes_col, comments_col]].to_string(index=False))
    print("Top 5 Recommended Posts (by engagement):")
    print(top_posts[[post_id_col, caption_col, likes_col, comments_col]])
    # Content-based recommendation: recommend similar posts to the top post
    if len(df) > 5:
        try:
            valid_captions = df[caption_col].dropna().astype(str).str.strip().replace('', float('nan')).dropna()
            if valid_captions.empty:
                print("No valid captions for content-based recommendations.")
            else:
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(df[caption_col].astype(str))
                sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
                similar_indices = sim_scores.argsort()[-6:][::-1][1:]
                similar_posts = df.iloc[similar_indices][[post_id_col, caption_col, likes_col, comments_col]]
                print("\nContent-based Recommendations (similar to top post):")
                print(similar_posts)
                logging.info("Content-based Recommendations (similar to top post):\n" + similar_posts.to_string(index=False))
        except Exception as e:
            print(f"Content-based recommendation failed: {e}")
            logging.warning(f"Content-based recommendation failed: {e}")
    # Collaborative filtering (simple): recommend posts with similar hashtag usage
    if 'hashtags' in df.columns:
        try:
            valid_hashtags = df['hashtags'].dropna().astype(str).str.strip().replace('', float('nan')).dropna()
            if valid_hashtags.empty:
                print("No valid hashtags for collaborative filtering.")
            else:
                df['hashtags_str'] = df['hashtags'].astype(str)
                tfidf_hash = TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b')
                tfidf_matrix_hash = tfidf_hash.fit_transform(df['hashtags_str'])
                sim_scores_hash = cosine_similarity(tfidf_matrix_hash[0:1], tfidf_matrix_hash).flatten()
                similar_indices_hash = sim_scores_hash.argsort()[-6:][::-1][1:]
                similar_posts_hash = df.iloc[similar_indices_hash][[post_id_col, caption_col, likes_col, comments_col]]
                print("\nCollaborative Filtering Recommendations (by hashtags):")
                print(similar_posts_hash)
                logging.info("Collaborative Filtering Recommendations (by hashtags):\n" + similar_posts_hash.to_string(index=False))
        except Exception as e:
            print(f"Collaborative filtering failed: {e}")
            logging.warning(f"Collaborative filtering failed: {e}")
