"""
Live Prediction UI Component
Handles real-time engagement prediction interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import joblib
import time
from .base import BaseUIComponent


class LivePredictionComponent(BaseUIComponent):
    """Live prediction component for real-time engagement prediction"""
    
    def show(self):
        """Live prediction interface"""
        st.markdown("### 🔮 Live Engagement Predictions")
        st.markdown("Get real-time engagement predictions for followers and content strategies")
        
        # Check if models are available
        if not self._check_model_availability():
            return
        
        # Prediction mode selection
        prediction_mode = st.selectbox(
            "Select Prediction Mode:",
            ["Single Follower Prediction", "Batch Prediction", "Content Strategy Optimizer"]
        )
        
        if prediction_mode == "Single Follower Prediction":
            self._show_single_prediction()
        elif prediction_mode == "Batch Prediction":
            self._show_batch_prediction()
        else:
            self._show_content_optimizer()
    
    def _check_model_availability(self):
        """Check if trained models are available"""
        model_files = ["outputs/rf_model.pkl", "outputs/xgb_model.pkl", "outputs/lgb_model.pkl"]
        available_models = [f for f in model_files if os.path.exists(f)]
        
        if not available_models:
            st.warning("⚠️ No trained models found! Please train models first.")
            st.info("Go to '🤖 Train Models' to train your models before using live predictions.")
            return False
        
        # Check for other required files
        required_files = [
            "outputs/metrics.json",
            "outputs/sentiment_scores.json"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.warning(f"⚠️ Missing required files: {', '.join(missing_files)}")
            return False
        
        return True
    
    def _show_single_prediction(self):
        """Show single follower prediction interface"""
        st.markdown("#### 👤 Single Follower Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Follower Information**")
            follower_count = st.number_input("Follower Count (#Followers)", min_value=0, value=1000, step=100)
            avg_likes = st.number_input("Average Likes per Post", min_value=0, value=50, step=5)
            avg_comments = st.number_input("Average Comments per Post", min_value=0, value=5, step=1)
            comment_likes = st.number_input("Average Comment Likes", min_value=0, value=2, step=1)
        
        with col2:
            st.markdown("**Content Preferences**")
            media_type = st.selectbox("Preferred Media Type", ["photo", "video", "album"])
            category = st.selectbox("Preferred Category", ["fashion", "travel", "food", "lifestyle", "tech"])
            sentiment_text = st.text_area("Sample Comment (for sentiment analysis)", 
                                        placeholder="Enter a sample comment to analyze sentiment...")
        
        # Prediction button
        if st.button("🔮 Predict Engagement", type="primary"):
            with st.spinner("Analyzing follower and generating predictions..."):
                try:
                    # Create feature vector
                    features = self._create_feature_vector(
                        follower_count, avg_likes, avg_comments, comment_likes,
                        media_type, category
                    )
                    
                    # Get sentiment if text provided
                    sentiment_score = 0.5  # Default neutral
                    sentiment_label = "neutral"
                    
                    if sentiment_text.strip():
                        sentiment_result = self._analyze_text_sentiment(sentiment_text)
                        sentiment_score = sentiment_result.get('confidence', 0.5)
                        sentiment_label = sentiment_result.get('sentiment', 'neutral')
                    
                    # Make predictions
                    predictions = self._make_predictions(features, sentiment_score)
                    
                    # Show results
                    self._show_prediction_results(predictions, sentiment_label, sentiment_score)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
    
    def _show_batch_prediction(self):
        """Show batch prediction interface"""
        st.markdown("#### 📊 Batch Follower Analysis")
        
        # File upload option
        uploaded_file = st.file_uploader("Upload CSV with follower data", type="csv")
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head())
                
                if st.button("🔮 Predict Batch Engagement", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        results = self._process_batch_predictions(df)
                        self._show_batch_results(results)
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        else:
            # Show sample data format
            st.markdown("**Expected CSV Format:**")
            sample_data = pd.DataFrame({
                'username': ['user1', 'user2', 'user3'],
                'followers': [1000, 2500, 800],
                'avg_likes': [50, 120, 30],
                'avg_comments': [5, 15, 3],
                'comment_likes': [2, 8, 1],
                'preferred_media': ['photo', 'video', 'photo'],
                'preferred_category': ['fashion', 'travel', 'lifestyle']
            })
            st.dataframe(sample_data)
            
            # Generate sample data button
            if st.button("📝 Generate Sample Data"):
                sample_df = self._generate_sample_batch_data()
                csv = sample_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Sample CSV",
                    data=csv,
                    file_name="sample_followers.csv",
                    mime="text/csv"
                )
    
    def _show_content_optimizer(self):
        """Show content strategy optimizer"""
        st.markdown("#### 🎯 Content Strategy Optimizer")
        
        # Load existing follower data
        from src.utils.high_value_utils import get_consolidated_high_value_followers
        followers = get_consolidated_high_value_followers()
        
        if followers:
            selected_followers = st.multiselect(
                "Select followers to optimize for:",
                list(followers.keys())[:20],  # Show first 20
                default=list(followers.keys())[:5]
            )
            
            if selected_followers:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        content_types = st.multiselect(
                            "Content Types to Test:",
                            ["photo", "video", "album"],
                            default=["photo", "video"]
                        )
                    
                    with col2:
                        categories = st.multiselect(
                            "Categories to Test:",
                            ["fashion", "travel", "food", "lifestyle", "tech"],
                            default=["fashion", "travel"]
                        )
                    
                    if st.button("🎯 Optimize Content Strategy", type="primary"):
                        with st.spinner("Optimizing content strategy..."):
                            optimization_results = self._optimize_content_strategy(
                                selected_followers, content_types, categories, followers
                            )
                            self._show_optimization_results(optimization_results)
            else:
                st.warning("⚠️ No high-value followers found.")
        else:
            st.warning("⚠️ Please select high-value followers first.")
    
    def _create_feature_vector(self, follower_count, avg_likes, avg_comments, 
                              comment_likes, media_type, category):
        """Create feature vector for prediction"""
        # Create base features
        features = {
            '#Followers': follower_count,
            'likes': avg_likes,
            'comments_count': avg_comments,
            'comment_likes': comment_likes,
        }
        
        # Add media type dummy variables
        media_types = ['photo', 'video', 'album']
        for mt in media_types:
            features[f'media_{mt}'] = 1 if media_type == mt else 0
        
        # Add category dummy variables  
        categories = ['fashion', 'travel', 'food', 'lifestyle', 'tech']
        for cat in categories:
            features[f'Category_{cat}'] = 1 if category == cat else 0
        
        return features
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of provided text"""
        try:
            # Use the existing sentiment analyzer
            temp_df = pd.DataFrame({'comment_text': [text]})
            results = self.sentiment_analyzer.analyze_sentiment(temp_df)
            
            if results and text in results:
                return results[text]
            else:
                return {'sentiment': 'neutral', 'confidence': 0.5}
        except:
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def _make_predictions(self, features, sentiment_score):
        """Make engagement predictions using trained models"""
        try:
            # Load best performing model
            with open("outputs/metrics.json", "r") as f:
                metrics = json.load(f)
            
            # Find best model based on F1-Score
            best_model = max(metrics.items(), key=lambda x: x[1].get('F1-Score', 0))
            model_name = best_model[0]
            
            # Load the model
            model_file_map = {
                'Random Forest': 'outputs/rf_model.pkl',
                'XGBoost': 'outputs/xgb_model.pkl', 
                'LightGBM': 'outputs/lgb_model.pkl'
            }
            
            model_file = model_file_map.get(model_name)
            if model_file and os.path.exists(model_file):
                model = joblib.load(model_file)
                
                # Create feature array (ensure proper order)
                feature_names = [
                    '#Followers', 'likes', 'comments_count', 'comment_likes',
                    'media_album', 'media_photo', 'media_video',
                    'Category_fashion', 'Category_food', 'Category_lifestyle', 
                    'Category_tech', 'Category_travel'
                ]
                
                # Create feature array
                feature_array = []
                for fname in feature_names:
                    if fname in features:
                        feature_array.append(features[fname])
                    else:
                        feature_array.append(0)
                
                # Add sentiment score
                feature_array.append(sentiment_score)
                
                # Make prediction
                X = np.array([feature_array])
                
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[0]
                    engagement_prob = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
                else:
                    engagement_prob = model.predict(X)[0]
                
                # Calculate derived metrics
                like_probability = min(engagement_prob * 1.2, 1.0)
                comment_probability = min(engagement_prob * 0.8, 1.0)
                influence_score = self._calculate_influence_score(features)
                
                return {
                    'model_used': model_name,
                    'engagement_probability': float(engagement_prob),
                    'like_probability': float(like_probability),
                    'comment_probability': float(comment_probability),
                    'influence_score': float(influence_score),
                    'confidence': float(best_model[1].get('F1-Score', 0))
                }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
        
        # Fallback prediction
        return {
            'model_used': 'Fallback',
            'engagement_probability': 0.5,
            'like_probability': 0.6,
            'comment_probability': 0.4,
            'influence_score': 0.5,
            'confidence': 0.5
        }
    
    def _calculate_influence_score(self, features):
        """Calculate influence score based on features"""
        follower_score = min(features['#Followers'] / 10000, 1.0)
        engagement_score = min((features['likes'] + features['comments_count']) / 100, 1.0)
        return (follower_score * 0.6 + engagement_score * 0.4)
    
    def _show_prediction_results(self, predictions, sentiment_label, sentiment_score):
        """Display prediction results"""
        st.markdown("### 📊 Prediction Results")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Engagement Probability", 
                f"{predictions['engagement_probability']:.1%}",
                delta=f"Confidence: {predictions['confidence']:.1%}"
            )
        
        with col2:
            st.metric(
                "Like Probability",
                f"{predictions['like_probability']:.1%}"
            )
        
        with col3:
            st.metric(
                "Comment Probability", 
                f"{predictions['comment_probability']:.1%}"
            )
        
        with col4:
            st.metric(
                "Influence Score",
                f"{predictions['influence_score']:.1%}"
            )
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability breakdown
            probs = [
                predictions['engagement_probability'],
                predictions['like_probability'], 
                predictions['comment_probability']
            ]
            labels = ['Engagement', 'Likes', 'Comments']
            
            fig = px.bar(
                x=labels, 
                y=probs,
                title="Prediction Breakdown",
                color=probs,
                color_continuous_scale="viridis"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="prediction_breakdown")
        
        with col2:
            # Sentiment analysis
            if sentiment_label != "neutral":
                st.markdown("**Sentiment Analysis:**")
                st.write(f"Detected Sentiment: **{sentiment_label.title()}**")
                st.write(f"Confidence: **{sentiment_score:.1%}**")
                
                # Sentiment impact
                sentiment_impact = self._calculate_sentiment_impact(sentiment_label, predictions)
                st.write(f"Sentiment Impact: **{sentiment_impact}**")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = self._generate_recommendations(predictions, sentiment_label)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
        
        # Model info
        st.info(f"🤖 Prediction made using: **{predictions['model_used']}** model")
    
    def _calculate_sentiment_impact(self, sentiment, predictions):
        """Calculate sentiment impact on engagement"""
        base_engagement = predictions['engagement_probability']
        
        if sentiment == 'positive':
            impact = "Positive sentiment likely to boost engagement by 10-15%"
        elif sentiment == 'negative':
            impact = "Negative sentiment may reduce engagement by 5-10%"
        else:
            impact = "Neutral sentiment - no significant impact expected"
        
        return impact
    
    def _generate_recommendations(self, predictions, sentiment):
        """Generate actionable recommendations"""
        recommendations = []
        
        engagement_prob = predictions['engagement_probability']
        influence_score = predictions['influence_score']
        
        # Engagement-based recommendations
        if engagement_prob > 0.7:
            recommendations.append("High engagement probability - prioritize this follower for targeted content")
        elif engagement_prob > 0.5:
            recommendations.append("Moderate engagement potential - consider A/B testing different content types")
        else:
            recommendations.append("Lower engagement probability - focus on building relationship through comments and interactions")
        
        # Influence-based recommendations
        if influence_score > 0.7:
            recommendations.append("High influence score - ideal for brand partnerships and sponsored content")
        elif influence_score > 0.5:
            recommendations.append("Moderate influence - good for user-generated content campaigns")
        
        # Sentiment-based recommendations
        if sentiment == 'positive':
            recommendations.append("Positive sentiment detected - leverage this for testimonials and reviews")
        elif sentiment == 'negative':
            recommendations.append("Negative sentiment detected - consider addressing concerns or improving content strategy")
        
        # General recommendations
        like_prob = predictions['like_probability']
        comment_prob = predictions['comment_probability']
        
        if like_prob > comment_prob:
            recommendations.append("Higher like probability - focus on visually appealing content")
        else:
            recommendations.append("Higher comment probability - create engaging, conversation-starting content")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _process_batch_predictions(self, df):
        """Process batch predictions for multiple followers"""
        results = []
        
        progress_bar = st.progress(0)
        for i, row in df.iterrows():
            # Create features for each row
            features = self._create_feature_vector(
                row.get('followers', 1000),
                row.get('avg_likes', 50),
                row.get('avg_comments', 5),
                row.get('comment_likes', 2),
                row.get('preferred_media', 'photo'),
                row.get('preferred_category', 'lifestyle')
            )
            
            # Make prediction
            predictions = self._make_predictions(features, 0.5)  # Default neutral sentiment
            
            # Add to results
            result = {
                'username': row.get('username', f'user_{i}'),
                'engagement_prob': predictions['engagement_probability'],
                'like_prob': predictions['like_probability'],
                'comment_prob': predictions['comment_probability'],
                'influence_score': predictions['influence_score']
            }
            results.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(df))
        
        return pd.DataFrame(results)
    
    def _show_batch_results(self, results_df):
        """Show batch prediction results"""
        st.markdown("### 📊 Batch Prediction Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_engagement = results_df['engagement_prob'].mean()
            st.metric("Average Engagement", f"{avg_engagement:.1%}")
        
        with col2:
            high_engagement = (results_df['engagement_prob'] > 0.7).sum()
            st.metric("High Engagement Users", high_engagement)
        
        with col3:
            top_influence = results_df['influence_score'].max()
            st.metric("Top Influence Score", f"{top_influence:.1%}")
        
        # Results table
        st.dataframe(results_df.round(3))
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement distribution
            fig = px.histogram(
                results_df,
                x='engagement_prob',
                title="Engagement Probability Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True, key="batch_engagement_dist")
        
        with col2:
            # Top performers
            top_10 = results_df.nlargest(10, 'engagement_prob')
            fig = px.bar(
                top_10,
                x='username',
                y='engagement_prob',
                title="Top 10 Engagement Predictions"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True, key="batch_top_performers")
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Results CSV",
            data=csv,
            file_name="engagement_predictions.csv",
            mime="text/csv"
        )
    
    def _generate_sample_batch_data(self):
        """Generate sample batch data for testing"""
        np.random.seed(42)
        n_samples = 20
        
        sample_data = pd.DataFrame({
            'username': [f'user_{i}' for i in range(1, n_samples + 1)],
            'followers': np.random.randint(500, 5000, n_samples),
            'avg_likes': np.random.randint(10, 200, n_samples),
            'avg_comments': np.random.randint(1, 50, n_samples),
            'comment_likes': np.random.randint(0, 20, n_samples),
            'preferred_media': np.random.choice(['photo', 'video', 'album'], n_samples),
            'preferred_category': np.random.choice(['fashion', 'travel', 'food', 'lifestyle', 'tech'], n_samples)
        })
        
        return sample_data
    
    def _optimize_content_strategy(self, selected_followers, content_types, categories, followers_data):
        """Optimize content strategy for selected followers"""
        optimization_results = []
        
        for follower in selected_followers:
            follower_data = followers_data.get(follower, {})
            
            best_combination = None
            best_score = 0
            
            # Test all combinations
            for content_type in content_types:
                for category in categories:
                    # Create features for this combination
                    features = self._create_feature_vector(
                        follower_data.get('influence_score', 1000) * 1000,  # Approximate followers
                        follower_data.get('engagement_score', 50) * 50,     # Approximate likes
                        follower_data.get('engagement_score', 5) * 5,       # Approximate comments
                        2,  # Default comment likes
                        content_type,
                        category
                    )
                    
                    # Get prediction
                    prediction = self._make_predictions(features, 0.5)
                    engagement_score = prediction['engagement_probability']
                    
                    if engagement_score > best_score:
                        best_score = engagement_score
                        best_combination = {
                            'content_type': content_type,
                            'category': category,
                            'predicted_engagement': engagement_score
                        }
            
            optimization_results.append({
                'follower': follower,
                'best_content': best_combination['content_type'],
                'best_category': best_combination['category'],
                'predicted_engagement': best_combination['predicted_engagement']
            })
        
        return optimization_results
    
    def _show_optimization_results(self, results):
        """Show content strategy optimization results"""
        st.markdown("### 🎯 Content Strategy Optimization Results")
        
        results_df = pd.DataFrame(results)
        
        # Summary
        avg_engagement = results_df['predicted_engagement'].mean()
        st.metric("Average Predicted Engagement", f"{avg_engagement:.1%}")
        
        # Results table
        st.dataframe(results_df.round(3))
        
        # Content type recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            content_counts = results_df['best_content'].value_counts()
            fig = px.pie(
                values=content_counts.values,
                names=content_counts.index,
                title="Recommended Content Types"
            )
            st.plotly_chart(fig, use_container_width=True, key="content_type_optimization")
        
        with col2:
            category_counts = results_df['best_category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Recommended Categories"
            )
            st.plotly_chart(fig, use_container_width=True, key="category_optimization")
        
        # Strategic recommendations
        st.markdown("### 💡 Strategic Recommendations")
        
        # Most recommended content type
        top_content = results_df['best_content'].mode()[0]
        top_category = results_df['best_category'].mode()[0]
        
        st.write(f"**Primary Content Strategy:** Focus on **{top_content}** content in the **{top_category}** category")
        st.write(f"**Expected Engagement:** {avg_engagement:.1%} average engagement rate")
        
        # High performers
        high_performers = results_df[results_df['predicted_engagement'] > results_df['predicted_engagement'].quantile(0.8)]
        if len(high_performers) > 0:
            st.write(f"**High-Impact Followers:** {len(high_performers)} followers show >80th percentile engagement potential")
            st.write("Focus your best content on these followers for maximum impact.")