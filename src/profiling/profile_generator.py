"""
Profile Generation Module
Generates individual engagement profiles and content strategy guidelines
"""

import pandas as pd
import numpy as np
import json
import logging
import joblib
from collections import Counter

class ProfileGenerator:
    def __init__(self):
        self.logger = self._setup_logger()
        self.best_model = None
        self.best_model_name = None
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def generate_profiles(self, include_sentiment=True, include_content_prefs=True,
                         min_confidence=0.7, max_profiles=50):
        """
        Generate individual engagement profiles for high-value followers
        
        Args:
            include_sentiment: Include sentiment analysis in profiles
            include_content_prefs: Include content preferences
            min_confidence: Minimum prediction confidence
            max_profiles: Maximum number of profiles to generate
            
        Returns:
            Dictionary of user profiles
        """
        self.logger.info("Generating user engagement profiles...")
        
        # Load required data
        high_value_followers = self._load_high_value_followers()
        sentiment_scores = self._load_sentiment_scores() if include_sentiment else {}
        model_metrics = self._load_model_metrics()
        
        # Select best model
        self._select_best_model(model_metrics)
        
        # Load preprocessed data
        df = self._load_preprocessed_data()
        
        # Generate profiles for each high-value follower
        profiles = {}
        processed_count = 0
        
        for username, follower_data in high_value_followers.items():
            if processed_count >= max_profiles:
                break
                
            try:
                profile = self._generate_single_profile(
                    username, follower_data, df, sentiment_scores,
                    include_sentiment, include_content_prefs, min_confidence
                )
                
                if profile:
                    profiles[username] = profile
                    processed_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error generating profile for {username}: {str(e)}")
                continue
        
        self.logger.info(f"Generated {len(profiles)} user profiles")
        return profiles
    
    def _load_high_value_followers(self):
        """Load high-value followers data"""
        from src.utils.high_value_utils import get_consolidated_high_value_followers
        return get_consolidated_high_value_followers()
    
    def _load_sentiment_scores(self):
        """Load sentiment analysis results"""
        try:
            with open("outputs/sentiment_scores.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("Sentiment scores not found")
            return {}
    
    def _load_model_metrics(self):
        """Load model evaluation metrics"""
        try:
            with open("outputs/metrics.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("Model metrics not found")
            return {}
    
    def _load_preprocessed_data(self):
        """Load preprocessed data"""
        return pd.read_csv("outputs/preprocessed_data.csv")
    
    def _select_best_model(self, model_metrics):
        """Select the best performing model based on F1-Score"""
        if not model_metrics or 'model_results' not in model_metrics:
            self.logger.warning("No model metrics available, using Random Forest as default")
            self.best_model_name = "Random Forest"
            return
        
        results = model_metrics['model_results']
        best_f1 = 0
        best_model_name = None
        
        for model_name, metrics in results.items():
            if 'error' in metrics:
                continue
            
            f1_score = metrics.get('F1-Score', 0)
            if f1_score > best_f1:
                best_f1 = f1_score
                best_model_name = model_name
        
        self.best_model_name = best_model_name or "Random Forest"
        self.logger.info(f"Selected best model: {self.best_model_name} (F1-Score: {best_f1:.3f})")
        
        # Load the best model
        self._load_best_model()
    
    def _load_best_model(self):
        """Load the best performing model"""
        try:
            model_files = {
                "Random Forest": "outputs/rf_model.pkl",
                "XGBoost": "outputs/xgb_model.pkl", 
                "LightGBM": "outputs/lgb_model.pkl"
            }
            
            if self.best_model_name in model_files:
                self.best_model = joblib.load(model_files[self.best_model_name])
                self.logger.info(f"Loaded {self.best_model_name} model")
            else:
                self.logger.warning(f"Model file not found for {self.best_model_name}")
                
        except Exception as e:
            self.logger.error(f"Error loading best model: {str(e)}")
    
    def _generate_single_profile(self, username, follower_data, df, sentiment_scores,
                                include_sentiment, include_content_prefs, min_confidence):
        """Generate profile for a single user"""
        # Filter user data
        user_data = df[df['comment_owner_username'] == username]
        
        if user_data.empty:
            return None
        
        profile = {
            "username": username,
            "follower_metrics": {
                "engagement_score": follower_data.get('engagement_score', 0),
                "influence_score": follower_data.get('influence_score', 0),
                "total_score": follower_data.get('total_score', 0),
                "followers_count": follower_data.get('followers', 0)
            }
        }
        
        # Generate engagement predictions
        engagement_predictions = self._predict_engagement(user_data)
        if engagement_predictions:
            profile["engagement_predictions"] = engagement_predictions
            
            # Check confidence threshold
            avg_confidence = np.mean([pred.get('confidence', 0) for pred in engagement_predictions.values()])
            if avg_confidence < min_confidence:
                return None
        
        # Add sentiment analysis
        if include_sentiment:
            sentiment_profile = self._generate_sentiment_profile(username, sentiment_scores)
            if sentiment_profile:
                profile["sentiment_profile"] = sentiment_profile
        
        # Add content preferences
        if include_content_prefs:
            content_prefs = self._analyze_content_preferences(user_data)
            if content_prefs:
                profile["content_preferences"] = content_prefs
        
        # Add behavioral insights
        behavioral_insights = self._generate_behavioral_insights(user_data, follower_data)
        profile["behavioral_insights"] = behavioral_insights
        
        # Add engagement probability
        overall_engagement_prob = self._calculate_overall_engagement_probability(user_data)
        profile["overall_engagement_probability"] = overall_engagement_prob
        
        return profile
    
    def _predict_engagement(self, user_data):
        """Predict engagement for different content types"""
        if not self.best_model:
            return None
        
        try:
            # Prepare features (same as in training)
            feature_columns = [
                'likes', 'comments_count', 'comment_likes', '#Followers',
                'engagement_frequency', 'influence_score', 'content_interaction',
                'comment_engagement_ratio', 'comment_length', 'has_emoji',
                'sentiment_positive', 'sentiment_negative', 'sentiment_neutral'
            ]
            
            # Add dummy encoded features
            dummy_columns = [col for col in user_data.columns if col.startswith(('media_', 'category_'))]
            feature_columns.extend(dummy_columns)
            
            # Keep only existing columns
            feature_columns = [col for col in feature_columns if col in user_data.columns]
            
            # Prepare feature matrix
            X = user_data[feature_columns].fillna(0)
            
            if X.empty:
                return None
            
            # Make predictions
            if hasattr(self.best_model, 'predict_proba'):
                predictions = self.best_model.predict_proba(X)[:, 1]
            else:
                predictions = self.best_model.predict(X)
            
            # Analyze predictions by content type
            engagement_by_content = {}
            
            # Group by media type
            for media_type in ['photo', 'video', 'album']:
                media_col = f'media_{media_type}'
                if media_col in user_data.columns:
                    media_mask = user_data[media_col] == 1
                    if media_mask.any():
                        media_predictions = predictions[media_mask]
                        engagement_by_content[f"{media_type}_engagement"] = {
                            "probability": float(np.mean(media_predictions)),
                            "confidence": float(np.std(media_predictions)),
                            "sample_count": int(media_mask.sum())
                        }
            
            # Group by category
            for category in ['fashion', 'travel', 'food', 'lifestyle', 'tech']:
                cat_col = f'category_{category}'
                if cat_col in user_data.columns:
                    cat_mask = user_data[cat_col] == 1
                    if cat_mask.any():
                        cat_predictions = predictions[cat_mask]
                        engagement_by_content[f"{category}_engagement"] = {
                            "probability": float(np.mean(cat_predictions)),
                            "confidence": float(np.std(cat_predictions)),
                            "sample_count": int(cat_mask.sum())
                        }
            
            return engagement_by_content
            
        except Exception as e:
            self.logger.error(f"Error predicting engagement: {str(e)}")
            return None
    
    def _generate_sentiment_profile(self, username, sentiment_scores):
        """Generate sentiment profile for user"""
        user_sentiments = []
        
        # Find user's sentiment scores
        for comment_key, sentiment_data in sentiment_scores.items():
            if username in comment_key:
                user_sentiments.append(sentiment_data)
        
        if not user_sentiments:
            return None
        
        # Aggregate sentiment data
        sentiments = [s['sentiment'] for s in user_sentiments]
        positive_scores = [s['positive'] for s in user_sentiments]
        negative_scores = [s['negative'] for s in user_sentiments]
        neutral_scores = [s['neutral'] for s in user_sentiments]
        confidences = [s['confidence'] for s in user_sentiments]
        
        sentiment_counts = Counter(sentiments)
        dominant_sentiment = sentiment_counts.most_common(1)[0][0]
        
        return {
            "dominant_sentiment": dominant_sentiment,
            "sentiment_distribution": dict(sentiment_counts),
            "average_scores": {
                "positive": float(np.mean(positive_scores)),
                "negative": float(np.mean(negative_scores)),
                "neutral": float(np.mean(neutral_scores))
            },
            "average_confidence": float(np.mean(confidences)),
            "sentiment_variability": float(np.std(positive_scores)),
            "total_comments_analyzed": len(user_sentiments)
        }
    
    def _analyze_content_preferences(self, user_data):
        """Analyze user's content preferences"""
        preferences = {}
        
        # Media type preferences
        media_preferences = {}
        for media_type in ['photo', 'video', 'album']:
            media_col = f'media_{media_type}'
            if media_col in user_data.columns:
                interaction_score = user_data[user_data[media_col] == 1]['content_interaction'].mean()
                if not pd.isna(interaction_score):
                    media_preferences[media_type] = float(interaction_score)
        
        if media_preferences:
            # Normalize preferences
            max_score = max(media_preferences.values())
            if max_score > 0:
                media_preferences = {k: v/max_score for k, v in media_preferences.items()}
            
            preferences["media_type_preferences"] = media_preferences
            preferences["preferred_media_type"] = max(media_preferences, key=media_preferences.get)
        
        # Category preferences
        category_preferences = {}
        for category in ['fashion', 'travel', 'food', 'lifestyle', 'tech']:
            cat_col = f'category_{category}'
            if cat_col in user_data.columns:
                interaction_score = user_data[user_data[cat_col] == 1]['content_interaction'].mean()
                if not pd.isna(interaction_score):
                    category_preferences[category] = float(interaction_score)
        
        if category_preferences:
            # Normalize preferences
            max_score = max(category_preferences.values())
            if max_score > 0:
                category_preferences = {k: v/max_score for k, v in category_preferences.items()}
            
            preferences["category_preferences"] = category_preferences
            preferences["preferred_category"] = max(category_preferences, key=category_preferences.get)
        
        return preferences
    
    def _generate_behavioral_insights(self, user_data, follower_data):
        """Generate behavioral insights"""
        insights = {}
        
        # Engagement patterns
        insights["engagement_frequency"] = float(user_data['engagement_frequency'].iloc[0]) if len(user_data) > 0 else 0
        insights["average_comment_likes"] = float(user_data['comment_likes'].mean())
        insights["comment_engagement_ratio"] = float(user_data['comment_engagement_ratio'].mean())
        
        # Activity level
        total_interactions = len(user_data)
        if total_interactions > 20:
            insights["activity_level"] = "high"
        elif total_interactions > 10:
            insights["activity_level"] = "medium"
        else:
            insights["activity_level"] = "low"
        
        # Engagement consistency
        engagement_std = float(user_data['content_interaction'].std())
        if engagement_std < 0.2:
            insights["engagement_consistency"] = "consistent"
        elif engagement_std < 0.5:
            insights["engagement_consistency"] = "moderate"
        else:
            insights["engagement_consistency"] = "variable"
        
        # Influence factor
        influence_score = follower_data.get('influence_score', 0)
        if influence_score > 0.8:
            insights["influence_level"] = "high"
        elif influence_score > 0.5:
            insights["influence_level"] = "medium"
        else:
            insights["influence_level"] = "low"
        
        return insights
    
    def _calculate_overall_engagement_probability(self, user_data):
        """Calculate overall engagement probability"""
        if not self.best_model or user_data.empty:
            return 0.5
        
        try:
            # Use average user features for prediction
            feature_columns = [
                'likes', 'comments_count', 'comment_likes', '#Followers',
                'engagement_frequency', 'influence_score', 'content_interaction',
                'comment_engagement_ratio', 'comment_length', 'has_emoji',
                'sentiment_positive', 'sentiment_negative', 'sentiment_neutral'
            ]
            
            # Add dummy columns
            dummy_columns = [col for col in user_data.columns if col.startswith(('media_', 'category_'))]
            feature_columns.extend(dummy_columns)
            
            # Keep only existing columns
            feature_columns = [col for col in feature_columns if col in user_data.columns]
            
            # Calculate average features
            avg_features = user_data[feature_columns].mean().fillna(0)
            X = avg_features.values.reshape(1, -1)
            
            # Make prediction
            if hasattr(self.best_model, 'predict_proba'):
                prob = self.best_model.predict_proba(X)[0, 1]
            else:
                prob = self.best_model.predict(X)[0]
            
            return float(prob)
            
        except Exception as e:
            self.logger.error(f"Error calculating engagement probability: {str(e)}")
            return 0.5
    
    def generate_guidelines(self, profiles):
        """Generate content strategy guidelines based on profiles"""
        self.logger.info("Generating content strategy guidelines...")
        
        guidelines = []
        
        # Analyze overall patterns
        all_media_prefs = {}
        all_category_prefs = {}
        all_sentiments = []
        
        for username, profile in profiles.items():
            # Collect media preferences
            if 'content_preferences' in profile and 'media_type_preferences' in profile['content_preferences']:
                for media, score in profile['content_preferences']['media_type_preferences'].items():
                    if media not in all_media_prefs:
                        all_media_prefs[media] = []
                    all_media_prefs[media].append(score)
            
            # Collect category preferences
            if 'content_preferences' in profile and 'category_preferences' in profile['content_preferences']:
                for category, score in profile['content_preferences']['category_preferences'].items():
                    if category not in all_category_prefs:
                        all_category_prefs[category] = []
                    all_category_prefs[category].append(score)
            
            # Collect sentiment preferences
            if 'sentiment_profile' in profile:
                all_sentiments.append(profile['sentiment_profile']['dominant_sentiment'])
        
        # Generate media type guidelines
        if all_media_prefs:
            avg_media_prefs = {media: np.mean(scores) for media, scores in all_media_prefs.items()}
            best_media = max(avg_media_prefs, key=avg_media_prefs.get)
            guidelines.append(f"Focus on {best_media} content - shows highest engagement among high-value followers")
        
        # Generate category guidelines
        if all_category_prefs:
            avg_category_prefs = {cat: np.mean(scores) for cat, scores in all_category_prefs.items()}
            best_categories = sorted(avg_category_prefs.items(), key=lambda x: x[1], reverse=True)[:3]
            top_categories = [cat for cat, _ in best_categories]
            guidelines.append(f"Prioritize content in these categories: {', '.join(top_categories)}")
        
        # Generate sentiment guidelines
        if all_sentiments:
            sentiment_counts = Counter(all_sentiments)
            dominant_sentiment = sentiment_counts.most_common(1)[0][0]
            guidelines.append(f"Maintain {dominant_sentiment} tone in content to align with audience preferences")
        
        # Generate individual user guidelines
        high_engagement_users = sorted(profiles.items(), 
                                     key=lambda x: x[1].get('overall_engagement_probability', 0), 
                                     reverse=True)[:5]
        
        for username, profile in high_engagement_users:
            user_guidelines = self._generate_user_specific_guidelines(username, profile)
            guidelines.extend(user_guidelines)
        
        # Add timing and posting guidelines
        guidelines.extend(self._generate_posting_guidelines(profiles))
        
        return guidelines
    
    def _generate_user_specific_guidelines(self, username, profile):
        """Generate guidelines for specific user"""
        guidelines = []
        
        # High-value user specific recommendations
        if profile.get('overall_engagement_probability', 0) > 0.7:
            guidelines.append(f"Target user @{username} with personalized content - high engagement probability")
            
            # Content preference specific
            if 'content_preferences' in profile:
                if 'preferred_media_type' in profile['content_preferences']:
                    preferred_media = profile['content_preferences']['preferred_media_type']
                    guidelines.append(f"Create {preferred_media} content for @{username}")
                
                if 'preferred_category' in profile['content_preferences']:
                    preferred_category = profile['content_preferences']['preferred_category']
                    guidelines.append(f"Share {preferred_category} content to engage @{username}")
            
            # Sentiment-based recommendations
            if 'sentiment_profile' in profile:
                dominant_sentiment = profile['sentiment_profile']['dominant_sentiment']
                if dominant_sentiment == 'positive':
                    guidelines.append(f"Use uplifting, positive messaging when engaging @{username}")
                elif dominant_sentiment == 'negative':
                    guidelines.append(f"Address concerns or provide solutions when posting for @{username}")
        
        return guidelines
    
    def _generate_posting_guidelines(self, profiles):
        """Generate general posting guidelines"""
        guidelines = [
            "Post consistently to maintain engagement with high-value followers",
            "Monitor engagement patterns and adjust content strategy accordingly",
            "Engage directly with high-value followers through comments and responses",
            "Use data-driven insights to optimize posting times and content types"
        ]
        
        # Analyze activity levels
        activity_levels = [profile['behavioral_insights']['activity_level'] 
                          for profile in profiles.values() 
                          if 'behavioral_insights' in profile]
        
        if activity_levels:
            high_activity_count = activity_levels.count('high')
            if high_activity_count > len(activity_levels) * 0.5:
                guidelines.append("Increase posting frequency - many high-value followers are highly active")
        
        return guidelines
    
    def save_profiles_and_guidelines(self, profiles, guidelines, 
                                   profiles_path="outputs/profiles.json",
                                   guidelines_path="outputs/guidelines.json"):
        """Save profiles and guidelines to files"""
        # Save profiles
        with open(profiles_path, 'w') as f:
            json.dump(profiles, f, indent=2)
        
        # Save guidelines
        guidelines_data = {
            "guidelines": guidelines,
            "generated_for": len(profiles),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        with open(guidelines_path, 'w') as f:
            json.dump(guidelines_data, f, indent=2)
        
        self.logger.info(f"Profiles saved to {profiles_path}")
        self.logger.info(f"Guidelines saved to {guidelines_path}")
    
    def export_profiles_summary(self, profiles, output_path="outputs/profiles_summary.csv"):
        """Export profiles summary as CSV"""
        summary_data = []
        
        for username, profile in profiles.items():
            row = {
                'username': username,
                'engagement_probability': profile.get('overall_engagement_probability', 0),
                'engagement_score': profile['follower_metrics']['engagement_score'],
                'influence_score': profile['follower_metrics']['influence_score'],
                'followers_count': profile['follower_metrics']['followers_count']
            }
            
            # Add sentiment info
            if 'sentiment_profile' in profile:
                row['dominant_sentiment'] = profile['sentiment_profile']['dominant_sentiment']
                row['sentiment_confidence'] = profile['sentiment_profile']['average_confidence']
            
            # Add content preferences
            if 'content_preferences' in profile:
                if 'preferred_media_type' in profile['content_preferences']:
                    row['preferred_media'] = profile['content_preferences']['preferred_media_type']
                if 'preferred_category' in profile['content_preferences']:
                    row['preferred_category'] = profile['content_preferences']['preferred_category']
            
            # Add behavioral insights
            if 'behavioral_insights' in profile:
                row['activity_level'] = profile['behavioral_insights']['activity_level']
                row['influence_level'] = profile['behavioral_insights']['influence_level']
            
            summary_data.append(row)
        
        # Create DataFrame and save
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(output_path, index=False)
        
        self.logger.info(f"Profiles summary exported to {output_path}")
        return df_summary
