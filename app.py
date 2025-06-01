import streamlit as st
import pandas as pd
import os
from analysis import sentiment_analysis, clustering_segmentation, engagement_prediction
from recommendations import post_recommender
from visualizations import engagement_trends

st.set_page_config(page_title="Instagram User Behavior Analysis", layout="wide")
st.title("Instagram User Behavior Analysis Dashboard")

# Add a reload button at the top of the app
if st.button("Reload App"):
    st.experimental_rerun()

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
        if df is None:
            st.error("Sentiment analysis failed. DataFrame is None.")
            st.stop()
    with st.spinner("Running Clustering/User Segmentation..."):
        df = clustering_segmentation.run(df)
        if df is None:
            st.error("Clustering/segmentation failed. DataFrame is None.")
            st.stop()
    with st.spinner("Predicting Engagement (Advanced ML Models)..."):
        df = engagement_prediction.run(df)
        if df is None:
            st.error("Engagement prediction failed. DataFrame is None.")
            st.stop()
    with st.spinner("Generating Engagement Visualizations..."):
        engagement_trends.run(df)
    with st.spinner("Generating Recommendations..."):
        post_recommender.run(df)
    st.success("Analysis pipeline complete! See results below.")

    # Save processed data for download
    st.download_button(
        label="Download Results CSV",
        data=df.to_csv(index=False),
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

# --- Download trained models section ---
import streamlit as st
import os

st.subheader("Download Trained ML Models")
model_files = [
    ("Linear Regression", "outputs/model_linear_regression.joblib"),
    ("Ridge Regression", "outputs/model_ridge.joblib"),
    ("Random Forest", "outputs/model_random_forest.joblib"),
    ("Feature Columns", "outputs/model_features.joblib")
]
for label, path in model_files:
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(f"Download {label} Model", f, file_name=os.path.basename(path))
    else:
        st.warning(f"{label} model not found. Run the analysis pipeline first.")

# --- ML Model Evaluation UI ---
st.subheader("Predict Engagement for Custom Input")
import joblib
import numpy as np

# Load models and features if available
model_paths = {
    'LinearRegression': 'outputs/model_linear_regression.joblib',
    'Ridge': 'outputs/model_ridge.joblib',
    'RandomForest': 'outputs/model_random_forest.joblib',
    'Features': 'outputs/model_features.joblib'
}
models = {}
for k, v in model_paths.items():
    if os.path.exists(v):
        models[k] = joblib.load(v)

# Only show prediction form if models and features are loaded
if 'Features' in models and len(models['Features']) > 0:
    st.markdown("Enter post/user details to predict engagement:")
    user_input = {}
    for feat in models['Features']:
        # Provide reasonable defaults and input types
        if feat in ['caption_sentiment', 'caption_sentiment_vader']:
            user_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.3f")
        elif feat in ['hour_of_day', 'day_of_week', 'user_cluster_k', 'user_cluster_agglom']:
            user_input[feat] = st.number_input(f"{feat}", value=0, step=1)
        else:
            user_input[feat] = st.number_input(f"{feat}", value=0.0)
    if st.button("Predict Engagement"):
        X_pred = np.array([[user_input.get(f, 0) for f in models['Features']]])
        st.write("Predictions:")
        for k in ['LinearRegression', 'Ridge', 'RandomForest']:
            if k in models:
                pred = models[k].predict(X_pred)[0]
                st.write(f"{k}: {pred:.2f}")
else:
    st.info("Trained models not found or features missing. Run the analysis pipeline first.")

# --- Predict Likes and Comments for Custom Input ---
st.subheader("Predict Likes and Comments for Custom Input (Separate Models)")
import joblib
import numpy as np

# Load models and features for likes and comments if available
model_paths_likes = {
    'LinearRegression_likes': 'outputs/model_linear_regression_likes.joblib',
    'Ridge_likes': 'outputs/model_ridge_likes.joblib',
    'RandomForest_likes': 'outputs/model_random_forest_likes.joblib',
    'Features_likes': 'outputs/model_features_likes.joblib'
}
model_paths_comments = {
    'LinearRegression_comments': 'outputs/model_linear_regression_comments.joblib',
    'Ridge_comments': 'outputs/model_ridge_comments.joblib',
    'RandomForest_comments': 'outputs/model_random_forest_comments.joblib',
    'Features_comments': 'outputs/model_features_comments.joblib'
}
models_likes = {}
for k, v in model_paths_likes.items():
    if os.path.exists(v):
        models_likes[k] = joblib.load(v)
models_comments = {}
for k, v in model_paths_comments.items():
    if os.path.exists(v):
        models_comments[k] = joblib.load(v)

# Only show prediction form if models and features are loaded for both
if (
    'Features_likes' in models_likes and len(models_likes['Features_likes']) > 0 and
    'Features_comments' in models_comments and len(models_comments['Features_comments']) > 0
):
    st.markdown("Enter post/user details to predict likes and comments:")
    # Exclude 'likes' from likes input, and 'comments'/'comments_count' from comments input
    features_likes = [f for f in models_likes['Features_likes'] if f.lower() not in ['likes', 'comments', 'comments_count']]
    features_comments = [f for f in models_comments['Features_comments'] if f.lower() not in ['comments', 'comments_count', 'likes']]
    all_features = sorted(set(features_likes) | set(features_comments))
    user_input = {}
    for feat in all_features:
        if feat in ['caption_sentiment', 'caption_sentiment_vader']:
            user_input[feat] = st.number_input(f"{feat}", value=0.0, format="%.3f", key=f"likecom_{feat}")
        elif feat in ['hour_of_day', 'day_of_week', 'user_cluster_k', 'user_cluster_agglom']:
            user_input[feat] = st.number_input(f"{feat}", value=0, step=1, key=f"likecom_{feat}")
        else:
            user_input[feat] = st.number_input(f"{feat}", value=0.0, key=f"likecom_{feat}")
    if st.button("Predict Likes and Comments", key="predict_likes_comments"):
        # Use the exact feature order and count for each model
        X_pred_likes = np.array([[user_input.get(f, 0) for f in models_likes['Features_likes'] if f.lower() not in ['likes', 'comments', 'comments_count']]])
        X_pred_comments = np.array([[user_input.get(f, 0) for f in models_comments['Features_comments'] if f.lower() not in ['comments', 'comments_count', 'likes']]])
        # If the number of features does not match, show a warning and skip prediction
        if X_pred_likes.shape[1] != len([f for f in models_likes['Features_likes'] if f.lower() not in ['likes', 'comments', 'comments_count']]):
            st.error(f"Input for likes prediction has {X_pred_likes.shape[1]} features, but model expects {len([f for f in models_likes['Features_likes'] if f.lower() not in ['likes', 'comments', 'comments_count']])}.")
        else:
            st.write("Predicted Likes:")
            for k in ['LinearRegression_likes', 'Ridge_likes', 'RandomForest_likes']:
                if k in models_likes:
                    pred = models_likes[k].predict(X_pred_likes)[0]
                    st.write(f"{k.replace('_likes','')}: {pred:.2f}")
        if X_pred_comments.shape[1] != len([f for f in models_comments['Features_comments'] if f.lower() not in ['comments', 'comments_count', 'likes']]):
            st.error(f"Input for comments prediction has {X_pred_comments.shape[1]} features, but model expects {len([f for f in models_comments['Features_comments'] if f.lower() not in ['comments', 'comments_count', 'likes']])}.")
        else:
            st.write("Predicted Comments:")
            for k in ['LinearRegression_comments', 'Ridge_comments', 'RandomForest_comments']:
                if k in models_comments:
                    pred = models_comments[k].predict(X_pred_comments)[0]
                    st.write(f"{k.replace('_comments','')}: {pred:.2f}")
else:
    st.info("Trained like/comment models or features not found. Please run the training pipeline for both targets.")

# Add a button to train like/comment models as a pipeline
st.subheader("Train Like/Comment ML Models Pipeline")
if st.button("Train Like/Comment Models (Pipeline)"):
    with st.spinner("Training like/comment ML models..."):
        from analysis.engagement_prediction import train_and_save_like_comment_models
        train_and_save_like_comment_models(df)
    st.success("Like/comment models trained and saved! You can now use the prediction UI below.")

st.info("See logs/project.log for detailed logs and errors.")
