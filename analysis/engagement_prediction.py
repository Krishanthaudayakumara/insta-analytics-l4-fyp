from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import logging
import numpy as np
import pandas as pd
import joblib

class PostRecommendationModel:
    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols
    def recommend(self, features, user_df=None):
        import numpy as np
        import pandas as pd
        from collections import Counter
        X_input = np.array([[features.get(f, 0) for f in self.feature_cols]])
        pred_engagement = self.model.predict(X_input)[0]
        # Defaults
        best_caption_length = features.get('caption_length', 15)
        best_category = features.get('Category_encoded', 'N/A')
        best_theme = 'N/A'
        best_hashtags = []
        best_sentiment = features.get('caption_sentiment', 'neutral')
        best_engagement_rate = round(pred_engagement, 3)
        debug_caption_lengths = []
        # If user_df is provided, recommend based on user's top engagement posts
        if user_df is not None and not user_df.empty:
            if 'engagement_rate' in user_df.columns:
                top_posts = user_df.sort_values(by='engagement_rate', ascending=False).head(5)
                # Sentiment: most common among top posts
                if 'caption_sentiment' in top_posts.columns:
                    sentiments = top_posts['caption_sentiment'].dropna().astype(str).str.lower()
                    if not sentiments.empty:
                        best_sentiment = Counter(sentiments).most_common(1)[0][0]
                # Caption length: average among top posts, skip zeros/empty
                if 'caption_length' in top_posts.columns:
                    lengths = top_posts['caption_length'].dropna().astype(float)
                    lengths = lengths[lengths > 0]
                    debug_caption_lengths = lengths.tolist()
                    if not lengths.empty:
                        best_caption_length = int(round(lengths.mean()))
                # Fallback: if still zero, use user's overall average
                if (not debug_caption_lengths or best_caption_length == 0) and 'caption_length' in user_df.columns:
                    all_lengths = user_df['caption_length'].dropna().astype(float)
                    all_lengths = all_lengths[all_lengths > 0]
                    if not all_lengths.empty:
                        best_caption_length = int(round(all_lengths.mean()))
                # Category: most common among top posts
                if 'Category' in top_posts.columns:
                    cats = top_posts['Category'].dropna().astype(str)
                    if not cats.empty:
                        best_category = Counter(cats).most_common(1)[0][0]
                # Theme: most common among top posts
                if 'theme' in top_posts.columns:
                    themes = top_posts['theme'].dropna().astype(str)
                    if not themes.empty:
                        best_theme = Counter(themes).most_common(1)[0][0]
                # Hashtags: use hashtags_agg if present, else hashtags
                hashtags_col = 'hashtags_agg' if 'hashtags_agg' in top_posts.columns else 'hashtags'
                if hashtags_col in top_posts.columns:
                    hashtags_series = top_posts[hashtags_col].dropna().astype(str)
                    hashtags_flat = []
                    for hstr in hashtags_series:
                        hstr = hstr.strip()
                        if not hstr or hstr.lower() in ('', 'nan', 'none', '[]'):
                            continue
                        # Support both comma and space delimiters
                        if ',' in hstr:
                            hashtags_flat.extend([t.strip() for t in hstr.split(',') if t.strip()])
                        else:
                            hashtags_flat.extend([t.strip() for t in hstr.split() if t.strip()])
                    hashtags_flat = [h for h in hashtags_flat if h]
                    if hashtags_flat:
                        best_hashtags = [h for h, _ in Counter(hashtags_flat).most_common(3)]
                # Fallback: if no hashtags found, use user's most common hashtags
                if not best_hashtags and 'hashtags_agg' in user_df.columns:
                    all_hashtags = sum([
                        [t.strip() for t in (h.split(',') if ',' in h else h.split()) if t.strip()]
                        for h in user_df['hashtags_agg'].dropna().astype(str)
                        if h.strip() and h.lower() not in ('', 'nan', 'none', '[]')
                    ], [])
                    if all_hashtags:
                        best_hashtags = [h for h, _ in Counter(all_hashtags).most_common(3)]
                # Engagement rate: average among top posts
                if 'engagement_rate' in top_posts.columns:
                    rates = top_posts['engagement_rate'].dropna().astype(float)
                    if not rates.empty:
                        best_engagement_rate = round(rates.mean(), 3)
        
        # Generate keyword recommendations based on user's high-performing posts
        best_keywords = []
        if user_df is not None and not user_df.empty:
            try:
                import re
                import nltk
                from collections import Counter
                
                # Ensure NLTK resources are available
                try:
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    try:
                        nltk.download('punkt', quiet=True)
                        nltk.download('stopwords', quiet=True)
                    except:
                        pass  # Skip if download fails
                
                from nltk.corpus import stopwords
                from nltk.tokenize import word_tokenize
                
                # Get top-performing posts for keyword analysis
                if 'engagement_rate' in user_df.columns:
                    engagement_threshold = user_df['engagement_rate'].quantile(0.75)
                    top_keyword_posts = user_df[user_df['engagement_rate'] >= engagement_threshold]
                else:
                    # Fallback to top posts by likes
                    if 'likes' in user_df.columns and len(user_df) > 1:
                        likes_threshold = user_df['likes'].quantile(0.75)
                        top_keyword_posts = user_df[user_df['likes'] >= likes_threshold]
                    else:
                        top_keyword_posts = user_df.head(5)
                
                # Extract keywords from captions
                caption_col = 'caption' if 'caption' in top_keyword_posts.columns else 'Caption'
                if caption_col in top_keyword_posts.columns and not top_keyword_posts[caption_col].dropna().empty:
                    all_captions = ' '.join(top_keyword_posts[caption_col].dropna().astype(str))
                    
                    # Clean text: remove hashtags, mentions, URLs, and special characters
                    cleaned_text = re.sub(r'#\w+|@\w+|http\S+|[^a-zA-Z\s]', ' ', all_captions.lower())
                    
                    # Tokenize and filter
                    stop_words = set(stopwords.words('english'))
                    words = word_tokenize(cleaned_text)
                    keywords = [word for word in words if word.isalpha() and len(word) > 2 and word not in stop_words]
                    
                    # Get top keywords
                    keyword_counts = Counter(keywords)
                    best_keywords = [word for word, _ in keyword_counts.most_common(6)]
                    
            except Exception:
                # If keyword extraction fails, provide category-based defaults
                category_keywords = {
                    'travel': ['adventure', 'explore', 'journey', 'destination'],
                    'fashion': ['style', 'outfit', 'trendy', 'look'],
                    'food': ['delicious', 'tasty', 'recipe', 'fresh'],
                    'fitness': ['workout', 'healthy', 'strong', 'motivation'],
                    'beauty': ['skincare', 'natural', 'glow', 'beautiful'],
                    'lifestyle': ['inspiration', 'happiness', 'mindful', 'grateful']
                }
                
                if best_category != 'N/A' and str(best_category).lower() in category_keywords:
                    best_keywords = category_keywords[str(best_category).lower()]
        
        # If category is encoded, decode it
        if best_category != 'N/A':
            try:
                le_path = 'outputs/model_post_recommendation_category_encoder.joblib'
                import joblib
                le = joblib.load(le_path)
                if isinstance(best_category, (int, float, np.integer)):
                    best_category = le.inverse_transform([int(best_category)])[0]
            except Exception:
                pass
        
        rec = {
            'caption_sentiment': best_sentiment,
            'caption_length': best_caption_length,
            'category': best_category,
            'hashtags': best_hashtags,
            'theme': best_theme,
            'expected_engagement_rate': best_engagement_rate,
            'keywords': best_keywords
        }
        return rec

def run(df):
    df = df.fillna(0)
    # --- Feature Engineering: Use all relevant features ---
    X_cols = []
    # Fixed: close the list and check all relevant columns
    for col in [
        'likes', 'shares', 'caption_length', 'num_hashtags', 'engagement_rate',
        'follower_adjusted_likes', 'follower_adjusted_comments',
        'caption_sentiment_vader', 'caption_sentiment', 'hour_of_day', 'day_of_week',
        '#Followers', '#Followees', '#Posts', 'user_cluster_k', 'user_cluster_agglom'
    ]:
        if col in df.columns:
            X_cols.append(col)
    if not X_cols:
        logging.error("No valid features for engagement prediction.")
        raise ValueError("No valid features for engagement prediction.")
    X = df[X_cols]
    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['Comments'] if 'Comments' in df.columns else df['comments_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    # Linear Regression
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    pred_lr = model_lr.predict(X_test)
    mse_lr = mean_squared_error(y_test, pred_lr)
    results['LinearRegression'] = mse_lr
    # Ridge Regression
    model_ridge = Ridge()
    model_ridge.fit(X_train, y_train)
    pred_ridge = model_ridge.predict(X_test)
    mse_ridge = mean_squared_error(y_test, pred_ridge)
    results['Ridge'] = mse_ridge
    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, pred_rf)
    results['RandomForest'] = mse_rf
    logging.info(f"Engagement Prediction MSEs: {results}")
    print("Model MSEs:", results)
    # Save trained models for download
    joblib.dump(model_lr, 'outputs/model_linear_regression.joblib')
    joblib.dump(model_ridge, 'outputs/model_ridge.joblib')
    joblib.dump(model_rf, 'outputs/model_random_forest.joblib')
    # Optionally, save feature columns for later use
    joblib.dump(X_cols, 'outputs/model_features.joblib')
    return df

def train_and_save_like_comment_models(df):
    df = df.fillna(0)
    # --- Feature selection ---
    feature_cols = [
        'likes', 'shares', 'caption_length', 'num_hashtags', 'engagement_rate',
        'follower_adjusted_likes', 'follower_adjusted_comments',
        'caption_sentiment_vader', 'caption_sentiment', 'hour_of_day', 'day_of_week',
        '#Followers', '#Followees', '#Posts', 'user_cluster_k', 'user_cluster_agglom'
    ]
    # Likes models: exclude 'likes', 'comments', 'comments_count' from X
    X_cols_likes = [col for col in feature_cols if col in df.columns and col.lower() not in ['likes', 'comments', 'comments_count']]
    X_likes = df[X_cols_likes].apply(pd.to_numeric, errors='coerce').fillna(0)
    # Comments models: exclude 'comments', 'comments_count', 'likes' from X
    X_cols_comments = [col for col in feature_cols if col in df.columns and col.lower() not in ['comments', 'comments_count', 'likes']]
    X_comments = df[X_cols_comments].apply(pd.to_numeric, errors='coerce').fillna(0)
    # --- Likes ---
    if 'Likes' in df.columns:
        y_likes = df['Likes']
    elif 'likes' in df.columns:
        y_likes = df['likes']
    else:
        y_likes = None
    # --- Comments ---
    if 'Comments' in df.columns:
        y_comments = df['Comments']
    elif 'comments_count' in df.columns:
        y_comments = df['comments_count']
    else:
        y_comments = None
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    # Likes models
    if y_likes is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_likes, y_likes, test_size=0.2, random_state=42)
        model_lr_likes = LinearRegression().fit(X_train, y_train)
        model_ridge_likes = Ridge().fit(X_train, y_train)
        model_rf_likes = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        joblib.dump(model_lr_likes, 'outputs/model_linear_regression_likes.joblib')
        joblib.dump(model_ridge_likes, 'outputs/model_ridge_likes.joblib')
        joblib.dump(model_rf_likes, 'outputs/model_random_forest_likes.joblib')
        joblib.dump(X_cols_likes, 'outputs/model_features_likes.joblib')
    # Comments models
    if y_comments is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_comments, y_comments, test_size=0.2, random_state=42)
        model_lr_comments = LinearRegression().fit(X_train, y_train)
        model_ridge_comments = Ridge().fit(X_train, y_train)
        model_rf_comments = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        joblib.dump(model_lr_comments, 'outputs/model_linear_regression_comments.joblib')
        joblib.dump(model_ridge_comments, 'outputs/model_ridge_comments.joblib')
        joblib.dump(model_rf_comments, 'outputs/model_random_forest_comments.joblib')
        joblib.dump(X_cols_comments, 'outputs/model_features_comments.joblib')
    return True

def train_and_save_post_recommendation_model(df):
    """
    Train and save a personalized post recommendation model for suggesting optimal next post attributes.
    Saves model to outputs/model_post_recommendation.joblib and features to outputs/model_post_recommendation_features.joblib.
    Now includes Category as a label-encoded feature.
    """
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    # Define features and targets for recommendation
    feature_cols = [
        'caption_length', 'num_hashtags', 'engagement_rate',
        'caption_sentiment', 'caption_sentiment_vader',
        'hour_of_day', 'day_of_week', 'user_cluster_k', 'user_cluster_agglom',
        '#Followers', '#Followees', '#Posts',
        'Category'  # Add category as a feature
    ]
    # Only keep features present in df
    feature_cols = [col for col in feature_cols if col in df.columns]
    # Encode Category if present
    le = None
    if 'Category' in feature_cols:
        le = LabelEncoder()
        df['Category_encoded'] = le.fit_transform(df['Category'].astype(str).fillna('unknown'))
        feature_cols = [c if c != 'Category' else 'Category_encoded' for c in feature_cols]
    # Target: engagement_rate (or likes/comments if preferred)
    target_col = 'engagement_rate' if 'engagement_rate' in df.columns else (
        'likes' if 'likes' in df.columns else 'comments_count'
    )
    if not feature_cols or target_col not in df.columns:
        raise ValueError("Required features or target not found in data for post recommendation model.")
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df[target_col].apply(pd.to_numeric, errors='coerce').fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # Save model and features
    joblib.dump(model, 'outputs/model_post_recommendation.joblib')
    joblib.dump(feature_cols, 'outputs/model_post_recommendation_features.joblib')
    # Save label encoder if used
    if le is not None:
        joblib.dump(le, 'outputs/model_post_recommendation_category_encoder.joblib')
    # Save wrapped model for UI
    wrapped_model = PostRecommendationModel(model, feature_cols)
    joblib.dump(wrapped_model, 'outputs/model_post_recommendation.joblib')
    return True

# Add CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and save like/comment ML models.")
    parser.add_argument('--data', type=str, default='data/processed_data/cleaned_merged_user_post_data.csv', help='Path to input data CSV')
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    train_and_save_like_comment_models(df)
    print("Like/comment models trained and saved.")