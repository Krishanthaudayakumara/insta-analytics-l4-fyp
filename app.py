import streamlit as st
import pandas as pd
import os
from analysis import sentiment_analysis, clustering_segmentation, engagement_prediction
from recommendations import post_recommender
from visualizations import engagement_trends

st.set_page_config(page_title="Instagram User Behavior Analysis", layout="wide")
st.title("Instagram User Behavior Analysis Dashboard")

# Sidebar for file selection
st.sidebar.header("Data Selection & Actions")
data_file = st.sidebar.file_uploader("Upload cleaned_merged_user_post_data.csv", type=["csv"])
preprocess_data = st.sidebar.button("Preprocess Raw Data (Full Pipeline)")
run_analysis = st.sidebar.button("Run Analysis Pipeline")

# Data Preprocessing Pipeline
if preprocess_data:
    with st.spinner("Processing raw Instagram data (extract, merge, clean)..."):
        import subprocess
        # Use python3 for Linux compatibility
        subprocess.run(["python3", "scripts/process_data_posts.py"])
        subprocess.run(["python3", "scripts/merge_with_influencers.py"])
        subprocess.run(["python3", "scripts/clean_data.py"])
    st.success("Data preprocessing complete! Cleaned data available in data/processed_data/cleaned_merged_user_post_data.csv.")
    st.info("You can now upload or use the cleaned data for analysis.")

if data_file:
    df = pd.read_csv(data_file)
    st.success("Data loaded successfully!")
else:
    # Try to load default data if exists
    default_path = "data/final_with_all_outputs.csv"
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.info("Loaded default data from data/processed_data/cleaned_merged_user_post_data.csv")
    else:
        st.warning("Please upload a data file to proceed.")
        st.stop()

if run_analysis:
    with st.spinner("Running Sentiment Analysis..."):
        df = sentiment_analysis.run(df)
    with st.spinner("Running Clustering/User Segmentation..."):
        df = clustering_segmentation.run(df)
    with st.spinner("Predicting Engagement..."):
        engagement_prediction.run(df)
    with st.spinner("Generating Engagement Visualizations..."):
        engagement_trends.run(df)
    with st.spinner("Generating Recommendations..."):
        post_recommender.run(df)
    st.success("Analysis pipeline complete! See results below.")

    # Save processed data for download
    st.download_button(
        label="Download Results CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="final_with_all_outputs.csv",
        mime="text/csv"
    )

# Show data preview
st.subheader("Data Preview")
st.dataframe(df.head(50))

# Show visualizations if available
st.subheader("Visualizations")
visualization_dir = "visualizations"
if os.path.exists(visualization_dir):
    for img in os.listdir(visualization_dir):
        if img.endswith(".png"):
            st.image(os.path.join(visualization_dir, img), caption=img)
outputs_dir = "outputs"
if os.path.exists(outputs_dir):
    for img in os.listdir(outputs_dir):
        if img.endswith(".png"):
            st.image(os.path.join(outputs_dir, img), caption=img)

# Show recommendations (top posts)
st.subheader("Top 5 Recommended Posts (by engagement)")
likes_col = 'Likes' if 'Likes' in df.columns else 'likes'
comments_col = 'Comments' if 'Comments' in df.columns else 'comments_count'
post_id_col = 'Post ID' if 'Post ID' in df.columns else 'post_id'
caption_col = 'Caption' if 'Caption' in df.columns else 'caption'
top_posts = df.sort_values(by=[likes_col, comments_col], ascending=False).head(5)
st.table(top_posts[[post_id_col, caption_col, likes_col, comments_col]])

# Content-based recommendations
st.subheader("Content-based Recommendations (similar to top post)")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
if len(df) > 5:
    try:
        if df[caption_col].dropna().astype(str).str.strip().replace('', float('nan')).dropna().empty:
            st.warning("No valid captions for content-based recommendations.")
        else:
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df[caption_col].astype(str))
            sim_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
            similar_indices = sim_scores.argsort()[-6:][::-1][1:]
            similar_posts = df.iloc[similar_indices][[post_id_col, caption_col, likes_col, comments_col]]
            st.table(similar_posts)
    except Exception as e:
        st.warning(f"Content-based recommendation failed: {e}")

# Collaborative filtering (by hashtags)
st.subheader("Collaborative Filtering Recommendations (by hashtags)")
if 'hashtags' in df.columns:
    try:
        if df['hashtags'].dropna().astype(str).str.strip().replace('', float('nan')).dropna().empty:
            st.warning("No valid hashtags for collaborative filtering.")
        else:
            df['hashtags_str'] = df['hashtags'].astype(str)
            tfidf_hash = TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b')
            tfidf_matrix_hash = tfidf_hash.fit_transform(df['hashtags_str'])
            sim_scores_hash = cosine_similarity(tfidf_matrix_hash[0:1], tfidf_matrix_hash).flatten()
            similar_indices_hash = sim_scores_hash.argsort()[-6:][::-1][1:]
            similar_posts_hash = df.iloc[similar_indices_hash][[post_id_col, caption_col, likes_col, comments_col]]
            st.table(similar_posts_hash)
    except Exception as e:
        st.warning(f"Collaborative filtering failed: {e}")

st.info("See logs/project.log for detailed logs and errors.")
