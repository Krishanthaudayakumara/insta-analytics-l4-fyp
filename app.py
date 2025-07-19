import streamlit as st
import pandas as pd
import os
import subprocess
import joblib
import numpy as np
import re
from analysis import sentiment_analysis, clustering_segmentation, engagement_prediction
from recommendations import post_recommender
from visualizations import engagement_trends

# Import advanced ML components
try:
    from ui.advanced_components import AdvancedMLIntegration, AdvancedMLComponents
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    st.warning("⚠️ Advanced ML components not available. Install required packages for full functionality.")

st.set_page_config(
    page_title="Instagram User Behavior Analysis", 
    layout="wide",
    page_icon="📊"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #E91E63;
    text-align: center;
    margin-bottom: 1rem;
    background: linear-gradient(45deg, #E91E63, #9C27B0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: bold;
}
.section-header {
    font-size: 1.8rem;
    color: #1976D2;
    margin: 1rem 0;
    border-left: 4px solid #2196F3;
    padding-left: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 Instagram User Behavior Analysis Dashboard</h1>', 
           unsafe_allow_html=True)

# Add a reload button at the top of the app
if st.button("Reload App"):
    st.rerun()

# Sidebar for file selection
st.sidebar.header("Data Selection & Actions")
data_file = st.sidebar.file_uploader("Upload cleaned_merged_user_post_data.csv", type=["csv"])
preprocess_data = st.sidebar.button("Preprocess Raw Data (Full Pipeline)")
run_analysis = st.sidebar.button("Run Analysis Pipeline")

# Advanced ML Integration
advanced_features = []
if ADVANCED_ML_AVAILABLE:
    advanced_ml = AdvancedMLIntegration()
    advanced_features = advanced_ml.add_advanced_sidebar()

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
    default_path = "data/processed_data/cleaned_merged_user_post_data.csv"
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

# Advanced ML Analysis Section
if ADVANCED_ML_AVAILABLE and advanced_features:
    st.markdown('<h2 class="section-header">🧠 Advanced Machine Learning Analysis</h2>', 
               unsafe_allow_html=True)
    
    try:
        # Run advanced ML analysis
        advanced_results = advanced_ml.integrate_with_existing_analysis(df, advanced_features)
        
        # Store results in session state for persistence
        if advanced_results:
            st.session_state.advanced_results = advanced_results
            
    except Exception as e:
        st.error(f"❌ Advanced ML analysis failed: {str(e)}")
        st.info("💡 This might be due to missing dependencies or incompatible data format.")

# Display Advanced ML Results if available
if 'advanced_results' in st.session_state:
    st.markdown("### 🎯 Advanced Analysis Summary")
    
    advanced_results = st.session_state.advanced_results
    
    # Create summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'boosting' in advanced_results:
            best_accuracy = advanced_results['boosting'].get('metrics', {}).get('accuracy', 0)
            st.metric("Best Model Accuracy", f"{best_accuracy:.3f}")
    
    with col2:
        if 'nlp' in advanced_results:
            avg_sentiment = advanced_results['nlp'].get('sentiment_score', 0)
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    
    with col3:
        if 'gnn' in advanced_results:
            communities = advanced_results['gnn'].get('communities_detected', 0)
            st.metric("Communities Found", communities)
    
    with col4:
        if 'multimodal' in advanced_results:
            quality_score = advanced_results['multimodal'].get('content_quality', 0)
            st.metric("Content Quality", f"{quality_score}/10")
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
# Check for hashtags_agg first, then hashtags
hashtag_col = 'hashtags_agg' if 'hashtags_agg' in df.columns else 'hashtags'
if hashtag_col in df.columns:
    try:
        # Only keep rows with non-empty, non-null hashtags
        valid_hashtags = df[hashtag_col].dropna().astype(str).str.strip()
        valid_hashtags = valid_hashtags[(valid_hashtags != '') & (valid_hashtags != 'nan')]
        if valid_hashtags.empty:
            st.warning("No valid hashtags for collaborative filtering.")
        else:
            # For hashtags_agg (comma-separated) or single hashtags
            def clean_hashtags(h):
                if hashtag_col == 'hashtags_agg' and ',' in h:
                    # hashtags_agg: comma-separated
                    return ' '.join([tag.strip() for tag in h.split(',') if tag.strip()])
                elif ' ' in h:
                    # Space-separated hashtags
                    return ' '.join([tag.strip() for tag in h.split() if tag.strip()])
                else:
                    # Single hashtag
                    return h.strip()
            
            # Apply cleaning to valid hashtags only
            df_filtered = df[df[hashtag_col].isin(valid_hashtags)].copy()
            df_filtered['hashtags_str'] = df_filtered[hashtag_col].apply(clean_hashtags)
            
            # Remove empty strings after cleaning
            df_filtered = df_filtered[df_filtered['hashtags_str'].str.strip() != '']
            
            if df_filtered.empty or df_filtered['hashtags_str'].str.strip().replace('', float('nan')).dropna().empty:
                st.warning("No valid hashtags for collaborative filtering after cleaning.")
            else:
                # Use min_df=1 to avoid empty vocabulary
                tfidf_hash = TfidfVectorizer(token_pattern=r'(?u)\\b\\w+\\b', min_df=1, max_df=0.95)
                tfidf_matrix_hash = tfidf_hash.fit_transform(df_filtered['hashtags_str'])
                if tfidf_matrix_hash.shape[0] > 1:
                    sim_scores_hash = cosine_similarity(tfidf_matrix_hash[0:1], tfidf_matrix_hash).flatten()
                    similar_indices_hash = sim_scores_hash.argsort()[-6:][::-1][1:]
                    similar_posts_hash = df_filtered.iloc[similar_indices_hash][[post_id_col, caption_col, likes_col, comments_col]]
                    st.table(similar_posts_hash)
                else:
                    st.warning("Not enough posts with hashtags for collaborative filtering.")
    except Exception as e:
        st.warning(f"Collaborative filtering failed: {e}")
else:
    st.warning("No hashtag columns found in the data.")

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

# --- Sidebar Model Training Buttons ---
st.sidebar.subheader("Model Training & Pipelines")
if 'df' in locals() or 'df' in globals():
    if st.sidebar.button("Train Like/Comment Models (Pipeline)"):
        with st.spinner("Training like/comment ML models..."):
            from analysis.engagement_prediction import train_and_save_like_comment_models
            train_and_save_like_comment_models(df)
        st.success("Like/comment models trained and saved! You can now use the prediction UI below.")
    if st.sidebar.button("Train Personalized Post Recommendation Model"):
        with st.spinner("Training personalized post recommendation model..."):
            from analysis.engagement_prediction import train_and_save_post_recommendation_model
            train_and_save_post_recommendation_model(df)
        st.success("Personalized post recommendation model trained and saved!")
else:
    st.sidebar.info("Please load data before training models.")

# --- Model Performance Evaluation Dashboard ---
st.header("🎯 Model Performance Evaluation")
st.markdown("---")

# Store dataframe in session state for dashboard access
if 'df' in locals() and df is not None:
    st.session_state.df = df

# Import the evaluation dashboard
from model_evaluation_dashboard import show_model_evaluation_dashboard

# Display the comprehensive model evaluation dashboard
show_model_evaluation_dashboard()

st.markdown("---")

# --- Advanced Personalized Post Recommendations ---
st.subheader("Advanced Personalized Post Recommendations")
user_col = 'username' if 'username' in df.columns else 'user'
if user_col not in df.columns:
    st.info("No user column found in data.")
else:
    # Filter out invalid usernames (True, False, 1, 0, empty, numeric, floats, whitespace)
    import re
    def is_valid_username(u):
        if not isinstance(u, str):
            return False
        u_strip = u.strip()
        if u_strip.lower() in ('true', 'false', 'username', '', 'none'):
            return False
        if u_strip.isnumeric():
            return False
        # Exclude floats
        try:
            float(u_strip)
            return False
        except ValueError:
            pass
        # Exclude if only whitespace
        if not u_strip:
            return False
        # Exclude if looks like a sentence or is too long
        if len(u_strip) > 30:
            return False
        # Exclude if contains spaces and is not a typical username
        if ' ' in u_strip and not re.match(r'^[a-zA-Z0-9_.-]+$', u_strip.replace(' ', '')):
            return False
        return True
    required_cols = ['caption_length', 'num_hashtags', 'engagement_rate']
    user_list = []
    debug_info = []
    for u in df[user_col].unique().tolist():
        valid_flag = is_valid_username(u)
        debug_info.append((u, valid_flag))
        if valid_flag and u not in user_list:
            user_list.append(u)
    if not user_list:
        st.warning("No users with valid data found in the dataset.")
        st.write("#### Debug: Sample of usernames and filtering status")
        st.write(pd.DataFrame(debug_info, columns=["username", "is_valid_username"]).head(30))
        st.stop()
    selected_user = st.selectbox("Select a user for advanced personalized recommendations:", user_list)
    if selected_user:
        user_df = df[df[user_col] == selected_user]
        if user_df.empty:
            st.warning(f"No data found for user '{selected_user}'. Please select another user.")
            st.stop()
        # 1. Show user profile summary
        st.markdown(f"**Profile for {selected_user}:**")
        st.write({
            "Followers": int(user_df['#Followers'].iloc[0]) if '#Followers' in user_df else "N/A",
            "Cluster": int(user_df['user_cluster_k'].iloc[0]) if 'user_cluster_k' in user_df else "N/A",
            "Recent Sentiment": user_df['caption_sentiment'].value_counts().idxmax() if 'caption_sentiment' in user_df else "N/A",
            "Avg Engagement Rate": round(user_df['engagement_rate'].mean(), 3) if 'engagement_rate' in user_df else "N/A"
        })
        # 2. Identify high-value followers (top commenters)
        st.markdown("**High-Value Followers (Top Engagers):**")
        if 'comment_owner_username' in user_df.columns:
            top_engagers = user_df['comment_owner_username'].value_counts().head(5)
            st.write(top_engagers)
            high_value_followers = top_engagers.index.tolist()
        else:
            st.info("No per-commenter data available for this user.")
            high_value_followers = []
        # 3. Personalized Recommendation for Next Post (for the user)
        st.markdown("**AI-Recommended Next Post (for Higher Engagement):**")
        import joblib
        rec_model_path = 'outputs/model_post_recommendation.joblib'
        feature_cols_path = 'outputs/model_post_recommendation_features.joblib'
        if os.path.exists(rec_model_path) and os.path.exists(feature_cols_path):
            rec_model = joblib.load(rec_model_path)
            feature_cols = joblib.load(feature_cols_path)
            rec_features = {}
            # Map sentiment strings to numeric values for model input
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            for feat in feature_cols:
                if feat == 'caption_sentiment':
                    val = user_df[feat].iloc[0] if feat in user_df and not user_df[feat].dropna().empty else 0
                    if isinstance(val, str):
                        rec_features[feat] = sentiment_map.get(val.lower(), 0)
                    else:
                        rec_features[feat] = val
                elif feat in user_df and not user_df[feat].dropna().empty:
                    rec_features[feat] = user_df[feat].iloc[0]
                else:
                    rec_features[feat] = 0
            # Handle category encoding for recommendation
            category_encoder_path = 'outputs/model_post_recommendation_category_encoder.joblib'
            if 'Category_encoded' in feature_cols and os.path.exists(category_encoder_path):
                le = joblib.load(category_encoder_path)
                if 'Category' in user_df and not user_df['Category'].dropna().empty:
                    cat_val = user_df['Category'].iloc[0]
                    rec_features['Category_encoded'] = le.transform([str(cat_val)])[0] if cat_val in le.classes_ else 0
                else:
                    rec_features['Category_encoded'] = 0
            # --- Clean up debug output and refine recommendation display ---
            rec = rec_model.recommend(rec_features, user_df=user_df)
            # Decode recommended category if possible
            recommended_category = None
            if 'Category_encoded' in feature_cols and os.path.exists(category_encoder_path):
                le = joblib.load(category_encoder_path)
                if isinstance(rec.get('category', None), (int, float)) and rec.get('category', None) != 'N/A':
                    try:
                        recommended_category = le.inverse_transform([int(rec['category'])])[0]
                    except Exception:
                        recommended_category = None
                elif isinstance(rec.get('category', None), str) and rec.get('category', None) not in (None, '', 'N/A'):
                    recommended_category = rec.get('category')
            elif rec.get('category', None) not in (None, '', 'N/A'):
                recommended_category = rec.get('category')

            # Only show fields if a valid recommendation is available
            st.write("**Recommended Caption Sentiment:**", rec.get('caption_sentiment', 'N/A'))
            st.write("**Recommended Caption Length (words):**", rec.get('caption_length', 'N/A'))
            if recommended_category:
                st.write("**Recommended Post Category:**", recommended_category)
            # Show hashtags or fallback to most common from user history
            hashtags = rec.get('hashtags', [])
            if hashtags:
                st.write("**Recommended Hashtags:**", ', '.join(hashtags))
            else:
                # Fallback: suggest most common hashtags from user's history
                if 'hashtags_agg' in user_df and not user_df['hashtags_agg'].dropna().empty:
                    from collections import Counter
                    all_hashtags = []
                    for h in user_df['hashtags_agg'].dropna().astype(str):
                        # Handle comma-separated hashtags in hashtags_agg
                        if ',' in h:
                            all_hashtags.extend([tag.strip() for tag in h.split(',') if tag.strip()])
                        else:
                            all_hashtags.extend([tag.strip() for tag in h.split() if tag.strip()])
                    top_hashtags = [h for h, _ in Counter(all_hashtags).most_common(5)]
                    if top_hashtags:
                        st.write("**Suggested Hashtags (from history):**", ', '.join(top_hashtags))
            if rec.get('theme', None) not in (None, '', 'N/A'):
                st.write("**Recommended Content Theme:**", rec.get('theme'))
            
            # Show keywords from the recommendation model or analyze user's posts
            model_keywords = rec.get('keywords', [])
            if model_keywords:
                st.write("**Recommended Caption Keywords:**", ', '.join(model_keywords))
            else:
                # Fallback to manual keyword analysis
                st.write("**Recommended Caption Keywords:**")
                try:
                    from collections import Counter
                    import re
                    import nltk
                    
                    # Ensure NLTK resources are available
                    try:
                        nltk.data.find('tokenizers/punkt')
                    except LookupError:
                        nltk.download('punkt')
                    try:
                        nltk.data.find('corpora/stopwords')
                    except LookupError:
                        nltk.download('stopwords')
                    
                    from nltk.corpus import stopwords
                    from nltk.tokenize import word_tokenize
                    
                    # Get user's high-engagement posts (top 25% by engagement rate)
                    if 'engagement_rate' in user_df and not user_df['engagement_rate'].dropna().empty:
                        engagement_threshold = user_df['engagement_rate'].quantile(0.75)
                        high_engagement_posts = user_df[user_df['engagement_rate'] >= engagement_threshold]
                    else:
                        # Fallback to top posts by likes if no engagement rate
                        if 'likes' in user_df and len(user_df) > 1:
                            likes_threshold = user_df['likes'].quantile(0.75)
                            high_engagement_posts = user_df[user_df['likes'] >= likes_threshold]
                        else:
                            high_engagement_posts = user_df.head(5)  # Use recent posts
                    
                    # Extract keywords from captions of high-engagement posts
                    caption_col = 'caption' if 'caption' in high_engagement_posts else 'Caption'
                    if caption_col in high_engagement_posts and not high_engagement_posts[caption_col].dropna().empty:
                        all_captions = ' '.join(high_engagement_posts[caption_col].dropna().astype(str))
                        
                        # Clean and tokenize text
                        # Remove hashtags, mentions, URLs, and special characters
                        cleaned_text = re.sub(r'#\w+|@\w+|http\S+|[^a-zA-Z\s]', ' ', all_captions.lower())
                        
                        # Tokenize and remove stopwords
                        stop_words = set(stopwords.words('english'))
                        words = word_tokenize(cleaned_text)
                        keywords = [word for word in words if word.isalpha() and len(word) > 2 and word not in stop_words]
                        
                        # Get most common keywords
                        keyword_counts = Counter(keywords)
                        top_keywords = [word for word, _ in keyword_counts.most_common(8)]
                        
                        if top_keywords:
                            st.write(f"💡 **Based on your top-performing posts:** {', '.join(top_keywords)}")
                            
                            # Additional category-specific keywords
                            category_keywords = {
                                'travel': ['adventure', 'journey', 'explore', 'destination', 'wanderlust', 'vacation', 'trip'],
                                'fashion': ['style', 'outfit', 'trendy', 'chic', 'fashionable', 'look', 'design'],
                                'food': ['delicious', 'tasty', 'recipe', 'yummy', 'flavor', 'cooking', 'fresh'],
                                'fitness': ['workout', 'healthy', 'strong', 'training', 'motivation', 'goals', 'fit'],
                                'beauty': ['skincare', 'makeup', 'glow', 'natural', 'beautiful', 'radiant', 'care'],
                                'lifestyle': ['inspiration', 'motivation', 'happiness', 'positivity', 'mindful', 'grateful'],
                                'business': ['success', 'growth', 'innovation', 'professional', 'strategy', 'leadership']
                            }
                            
                            user_category = user_df['Category'].iloc[0] if 'Category' in user_df and not user_df['Category'].dropna().empty else None
                            if user_category and str(user_category).lower() in category_keywords:
                                category_words = category_keywords[str(user_category).lower()]
                                # Remove already suggested keywords
                                new_category_words = [w for w in category_words if w not in top_keywords][:4]
                                if new_category_words:
                                    st.write(f"🎯 **Category-specific suggestions ({user_category}):** {', '.join(new_category_words)}")
                        else:
                            st.write("No specific keywords identified from your posts.")
                    else:
                        st.write("No caption data available for keyword analysis.")
                except Exception as e:
                    st.write("Keyword analysis unavailable.")
            
            # Add explanatory info box about engagement rate
            with st.expander("ℹ️ What is Engagement Rate?", expanded=False):
                st.markdown("""
                **Engagement Rate** = (Total Engagements ÷ Total Followers) × 100
                
                - **Total Engagements** = Likes + Comments + Shares + Saves
                - **Industry Benchmarks:**
                  - 🔥 **Excellent**: 6%+ (top-tier influencers)
                  - ✅ **Good**: 3-6% (above average performance)
                  - 📊 **Average**: 1-3% (typical for most accounts)
                  - 📉 **Below Average**: <1% (needs optimization)
                
                **Example**: If you have 10,000 followers and get 500 total engagements, your rate is 5% (good performance).
                """)
            
            # Enhanced Engagement Rate Display with Context
            engagement_rate = rec.get('expected_engagement_rate', 'N/A')
            if engagement_rate != 'N/A':
                engagement_percent = round(float(engagement_rate) * 100, 2)
                st.write(f"**Expected Engagement Rate:** {engagement_percent}%")
                
                # Add context about what this means
                if engagement_percent >= 6:
                    st.success(f"🔥 Excellent engagement rate! ({engagement_percent}% is above 6% - top-tier performance)")
                elif engagement_percent >= 3:
                    st.info(f"✅ Good engagement rate! ({engagement_percent}% is above average - expect strong audience interaction)")
                elif engagement_percent >= 1:
                    st.warning(f"📊 Average engagement rate ({engagement_percent}% - typical for most accounts)")
                else:
                    st.error(f"📉 Below average engagement rate ({engagement_percent}% - consider optimizing content)")
                
                # Calculate expected interactions based on follower count
                follower_count = user_df['#Followers'].iloc[0] if '#Followers' in user_df.columns and not user_df['#Followers'].empty else None
                if follower_count and follower_count > 0:
                    expected_engagements = int(follower_count * float(engagement_rate))
                    st.write(f"**Expected Total Engagements:** ~{expected_engagements:,} interactions (likes + comments)")
                    st.caption(f"Based on your {follower_count:,} followers × {engagement_percent}% engagement rate")
            else:
                st.write("**Expected Engagement Rate:**", engagement_rate)
            
            # Add Like Count Prediction
            st.markdown("---")
            st.subheader("📈 Predicted Performance Metrics")
            
            # Try to load like prediction models
            like_models = {
                'Linear Regression': 'outputs/model_linear_regression_likes.joblib',
                'Random Forest': 'outputs/model_random_forest_likes.joblib',
                'Ridge Regression': 'outputs/model_ridge_likes.joblib'
            }
            
            like_features_path = 'outputs/model_features_likes.joblib'
            
            if os.path.exists(like_features_path):
                like_feature_cols = joblib.load(like_features_path)
                
                # Prepare features for like prediction
                like_features = {}
                for feat in like_feature_cols:
                    if feat == 'caption_sentiment':
                        val = user_df[feat].iloc[0] if feat in user_df and not user_df[feat].dropna().empty else 0
                        if isinstance(val, str):
                            like_features[feat] = sentiment_map.get(val.lower(), 0)
                        else:
                            like_features[feat] = val
                    elif feat in user_df and not user_df[feat].dropna().empty:
                        like_features[feat] = user_df[feat].iloc[0]
                    else:
                        like_features[feat] = 0
                
                # Handle category encoding for likes
                if 'Category_encoded' in like_feature_cols and os.path.exists(category_encoder_path):
                    le = joblib.load(category_encoder_path)
                    if 'Category' in user_df and not user_df['Category'].dropna().empty:
                        cat_val = user_df['Category'].iloc[0]
                        like_features['Category_encoded'] = le.transform([str(cat_val)])[0] if cat_val in le.classes_ else 0
                    else:
                        like_features['Category_encoded'] = 0
                
                # Predict likes with different models
                like_predictions = {}
                for model_name, model_path in like_models.items():
                    if os.path.exists(model_path):
                        try:
                            like_model = joblib.load(model_path)
                            X_input = np.array([[like_features.get(f, 0) for f in like_feature_cols]])
                            pred_likes = max(0, int(like_model.predict(X_input)[0]))  # Ensure non-negative
                            like_predictions[model_name] = pred_likes
                        except Exception as e:
                            st.warning(f"Could not load {model_name} likes model: {e}")
                
                if like_predictions:
                    # Display predictions
                    avg_likes = int(np.mean(list(like_predictions.values())))
                    st.write(f"**Predicted Likes:** ~{avg_likes:,}")
                    
                    # Show range
                    min_likes = min(like_predictions.values())
                    max_likes = max(like_predictions.values())
                    if min_likes != max_likes:
                        st.caption(f"Model range: {min_likes:,} - {max_likes:,} likes")
                    
                    # Compare with user's average
                    if 'likes' in user_df.columns:
                        user_avg_likes = user_df['likes'].mean()
                        if avg_likes > user_avg_likes:
                            improvement = ((avg_likes - user_avg_likes) / user_avg_likes) * 100
                            st.success(f"🚀 {improvement:.1f}% better than your average ({user_avg_likes:.0f} likes)")
                        elif avg_likes < user_avg_likes * 0.9:
                            st.info(f"📊 Below your average ({user_avg_likes:.0f} likes) - consider refining strategy")
                    
                    # Show individual model predictions in expander
                    with st.expander("View detailed model predictions"):
                        for model_name, pred in like_predictions.items():
                            st.write(f"- {model_name}: {pred:,} likes")
                else:
                    st.info("Like prediction models not available")
                    
                # Add Comment Count Prediction
                comment_models = {
                    'Linear Regression': 'outputs/model_linear_regression_comments.joblib',
                    'Random Forest': 'outputs/model_random_forest_comments.joblib',
                    'Ridge Regression': 'outputs/model_ridge_comments.joblib'
                }
                
                comment_features_path = 'outputs/model_features_comments.joblib'
                
                if os.path.exists(comment_features_path):
                    comment_feature_cols = joblib.load(comment_features_path)
                    
                    # Prepare features for comment prediction
                    comment_features = {}
                    for feat in comment_feature_cols:
                        if feat == 'caption_sentiment':
                            val = user_df[feat].iloc[0] if feat in user_df and not user_df[feat].dropna().empty else 0
                            if isinstance(val, str):
                                comment_features[feat] = sentiment_map.get(val.lower(), 0)
                            else:
                                comment_features[feat] = val
                        elif feat in user_df and not user_df[feat].dropna().empty:
                            comment_features[feat] = user_df[feat].iloc[0]
                        else:
                            comment_features[feat] = 0
                    
                    # Handle category encoding for comments
                    if 'Category_encoded' in comment_feature_cols and os.path.exists(category_encoder_path):
                        le = joblib.load(category_encoder_path)
                        if 'Category' in user_df and not user_df['Category'].dropna().empty:
                            cat_val = user_df['Category'].iloc[0]
                            comment_features['Category_encoded'] = le.transform([str(cat_val)])[0] if cat_val in le.classes_ else 0
                        else:
                            comment_features['Category_encoded'] = 0
                    
                    # Predict comments with different models
                    comment_predictions = {}
                    for model_name, model_path in comment_models.items():
                        if os.path.exists(model_path):
                            try:
                                comment_model = joblib.load(model_path)
                                X_input = np.array([[comment_features.get(f, 0) for f in comment_feature_cols]])
                                pred_comments = max(0, int(comment_model.predict(X_input)[0]))  # Ensure non-negative
                                comment_predictions[model_name] = pred_comments
                            except Exception as e:
                                st.warning(f"Could not load {model_name} comments model: {e}")
                    
                    if comment_predictions:
                        # Display comment predictions
                        avg_comments = int(np.mean(list(comment_predictions.values())))
                        st.write(f"**Predicted Comments:** ~{avg_comments:,}")
                        
                        # Show range
                        min_comments = min(comment_predictions.values())
                        max_comments = max(comment_predictions.values())
                        if min_comments != max_comments:
                            st.caption(f"Model range: {min_comments:,} - {max_comments:,} comments")
                        
                        # Compare with user's average
                        if 'comments_count' in user_df.columns:
                            user_avg_comments = user_df['comments_count'].mean()
                            if avg_comments > user_avg_comments:
                                improvement = ((avg_comments - user_avg_comments) / user_avg_comments) * 100
                                st.success(f"💬 {improvement:.1f}% more comments than your average ({user_avg_comments:.0f})")
                            elif avg_comments < user_avg_comments * 0.9:
                                st.info(f"💬 Below your average ({user_avg_comments:.0f} comments)")
                        
                        # Add comment predictions to detailed view
                        with st.expander("View detailed comment predictions"):
                            for model_name, pred in comment_predictions.items():
                                st.write(f"- {model_name}: {pred:,} comments")
                    
                    # Calculate total predicted engagement
                    if like_predictions and comment_predictions:
                        total_predicted_engagement = avg_likes + avg_comments
                        st.markdown("---")
                        st.metric(
                            label="**Total Predicted Engagement**",
                            value=f"{total_predicted_engagement:,}",
                            help="Combined likes + comments prediction"
                        )
                        
                        # Calculate predicted engagement rate
                        if follower_count and follower_count > 0:
                            predicted_engagement_rate = (total_predicted_engagement / follower_count) * 100
                            st.metric(
                                label="**Predicted Engagement Rate**",
                                value=f"{predicted_engagement_rate:.2f}%",
                                delta=f"{abs(predicted_engagement_rate - engagement_percent):.2f}% vs expected",
                                help="Based on likes + comments predictions"
                            )
            else:
                st.info("Like prediction features not found. Train models first to get like predictions.")
        else:
            st.info("Personalized post recommendation model not found. Please train it from the sidebar.")
        # 4. Personalized Recommendations for High-Value Followers
        if high_value_followers:
            st.markdown("**Personalized Recommendations for High-Value Followers:**")
            # --- Combined Overall Recommendation ---
            all_recs = [rec]
            for follower in high_value_followers:
                follower_comments = user_df[user_df['comment_owner_username'] == follower]
                follower_sentiment = follower_comments['caption_sentiment'].value_counts().idxmax() if 'caption_sentiment' in follower_comments and not follower_comments.empty else 'neutral'
                follower_features = rec_features.copy()
                if isinstance(follower_sentiment, str):
                    follower_features['caption_sentiment'] = sentiment_map.get(follower_sentiment.lower(), 0)
                else:
                    follower_features['caption_sentiment'] = follower_sentiment
                follower_rec = rec_model.recommend(follower_features, user_df=user_df)
                all_recs.append(follower_rec)
            # Aggregate overall recommendation
            from collections import Counter
            def most_common(lst):
                return Counter(lst).most_common(1)[0][0] if lst else 'N/A'
            sentiments = [r.get('caption_sentiment', None) for r in all_recs if r.get('caption_sentiment', None) is not None]
            caption_lengths = [r.get('caption_length', None) for r in all_recs if r.get('caption_length', None) is not None]
            hashtags = sum([r.get('hashtags', []) for r in all_recs if r.get('hashtags', [])], [])
            engagement_rates = [r.get('expected_engagement_rate', None) for r in all_recs if r.get('expected_engagement_rate', None) is not None]
            st.markdown("**Combined Overall Recommendation:**")
            st.write({
                "Most Common Caption Sentiment": most_common(sentiments),
                "Average Caption Length (words)": round(sum(caption_lengths)/len(caption_lengths), 2) if caption_lengths else 'N/A',
                "Most Common Hashtags": ', '.join([h for h, _ in Counter(hashtags).most_common(3)]) if hashtags else 'N/A',
                "Average Expected Engagement Rate": round(sum(engagement_rates)/len(engagement_rates), 3) if engagement_rates else 'N/A'
            })
            st.markdown("---")
            for follower in high_value_followers:
                follower_comments = user_df[user_df['comment_owner_username'] == follower]
                follower_sentiment = follower_comments['caption_sentiment'].value_counts().idxmax() if 'caption_sentiment' in follower_comments and not follower_comments.empty else 'neutral'
                follower_features = rec_features.copy()
                # Map follower sentiment to numeric
                if isinstance(follower_sentiment, str):
                    follower_features['caption_sentiment'] = sentiment_map.get(follower_sentiment.lower(), 0)
                else:
                    follower_features['caption_sentiment'] = follower_sentiment
                follower_rec = rec_model.recommend(follower_features, user_df=user_df)
                st.write(f"**Follower:** {follower}")
                st.write("Recommended Caption Sentiment:", follower_rec.get('caption_sentiment', 'N/A'))
                st.write("Recommended Caption Length (words):", follower_rec.get('caption_length', 'N/A'))
                st.write("Recommended Hashtags:", ', '.join(follower_rec.get('hashtags', [])))
                st.write("Expected Engagement Rate:", follower_rec.get('expected_engagement_rate', 'N/A'))
                st.markdown("---")

# --- Trending Keywords Analysis ---
st.subheader("📈 Trending Keywords in Dataset")
try:
    from collections import Counter
    import re
    import nltk
    
    # Get captions from high-engagement posts across the dataset
    caption_col = 'caption' if 'caption' in df else 'Caption'
    if caption_col in df and not df[caption_col].dropna().empty:
        # Filter for high-engagement posts (top 20% by engagement rate)
        if 'engagement_rate' in df and not df['engagement_rate'].dropna().empty:
            engagement_threshold = df['engagement_rate'].quantile(0.8)
            trending_posts = df[df['engagement_rate'] >= engagement_threshold]
        else:
            # Fallback to top posts by likes
            if 'likes' in df and len(df) > 100:
                likes_threshold = df['likes'].quantile(0.8)
                trending_posts = df[df['likes'] >= likes_threshold]
            else:
                trending_posts = df.head(100)  # Use sample
        
        if not trending_posts.empty:
            # Sample for performance if dataset is large
            if len(trending_posts) > 1000:
                trending_posts = trending_posts.sample(n=1000)
            
            # Extract keywords from trending captions
            all_trending_captions = ' '.join(trending_posts[caption_col].dropna().astype(str))
            
            # Clean and tokenize
            cleaned_text = re.sub(r'#\w+|@\w+|http\S+|[^a-zA-Z\s]', ' ', all_trending_captions.lower())
            
            # Ensure NLTK resources
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
            except LookupError:
                st.info("Downloading language resources...")
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(cleaned_text)
            keywords = [word for word in words if word.isalpha() and len(word) > 2 and word not in stop_words]
            
            # Get trending keywords
            keyword_counts = Counter(keywords)
            trending_keywords = [word for word, count in keyword_counts.most_common(15)]
            
            if trending_keywords:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔥 Most Used Words in High-Engagement Posts:**")
                    st.write(", ".join(trending_keywords[:8]))
                
                with col2:
                    st.markdown("**💡 Content Strategy Insights:**")
                    # Analyze keyword patterns
                    emotion_words = [w for w in trending_keywords if w in ['love', 'happy', 'amazing', 'beautiful', 'awesome', 'incredible', 'perfect', 'wonderful', 'excited', 'grateful']]
                    action_words = [w for w in trending_keywords if w in ['explore', 'discover', 'create', 'share', 'enjoy', 'experience', 'celebrate', 'achieve', 'inspire', 'transform']]
                    
                    if emotion_words:
                        st.write(f"😊 **Emotion keywords:** {', '.join(emotion_words[:3])}")
                    if action_words:
                        st.write(f"⚡ **Action keywords:** {', '.join(action_words[:3])}")
                
                # Category-specific trending analysis
                if 'Category' in df and not df['Category'].dropna().empty:
                    st.markdown("**📊 Trending by Category:**")
                    categories = df['Category'].value_counts().head(3).index.tolist()
                    
                    category_cols = st.columns(len(categories))
                    for i, category in enumerate(categories):
                        with category_cols[i]:
                            cat_posts = trending_posts[trending_posts['Category'] == category]
                            if not cat_posts.empty and len(cat_posts) >= 5:
                                cat_captions = ' '.join(cat_posts[caption_col].dropna().astype(str))
                                cat_cleaned = re.sub(r'#\w+|@\w+|http\S+|[^a-zA-Z\s]', ' ', cat_captions.lower())
                                cat_words = word_tokenize(cat_cleaned)
                                cat_keywords = [word for word in cat_words if word.isalpha() and len(word) > 2 and word not in stop_words]
                                cat_top = [word for word, _ in Counter(cat_keywords).most_common(4)]
                                
                                st.write(f"**{category.title()}:**")
                                st.caption(", ".join(cat_top))
            else:
                st.info("No trending keywords found.")
    else:
        st.info("No caption data available for trending analysis.")
        
except Exception as e:
    st.warning("Trending keywords analysis unavailable.")

st.info("See logs/project.log for detailed logs and errors.")
