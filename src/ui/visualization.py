"""
Visualization UI Component
Handles results visualization interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from .base import BaseUIComponent


class VisualizationComponent(BaseUIComponent):
    """Visualization component"""
    
    def show(self):
        """Visualization interface"""
        st.markdown("### 📋 Results Visualization")
        
        # Check available outputs
        from src.utils.high_value_utils import check_high_value_data_exists
        
        available_outputs = {
            "Preprocessed Data": "outputs/preprocessed_data.csv",
            "High-Value Followers": check_high_value_data_exists(),
            "Sentiment Scores": "outputs/sentiment_scores.json",
            "Model Metrics": "outputs/metrics.json",
            "User Profiles": "outputs/profiles.json",
            "Guidelines": "outputs/guidelines.json"
        }
        
        existing_outputs = {name: (path if isinstance(path, str) else True) 
                          for name, path in available_outputs.items() 
                          if (os.path.exists(path) if isinstance(path, str) else path)}
        
        if not existing_outputs:
            st.warning("⚠️ No outputs available for visualization!")
            return
        
        # Visualization options
        viz_type = st.selectbox(
            "Select Visualization:",
            ["Model Performance", "Engagement Profiles", "Sentiment Analysis", 
             "Follower Distribution", "Content Preferences"]
        )
        
        if viz_type == "Model Performance" and "Model Metrics" in existing_outputs:
            self._show_model_performance_viz()
        elif viz_type == "Engagement Profiles" and "User Profiles" in existing_outputs:
            self._show_engagement_profiles_viz()
        elif viz_type == "Sentiment Analysis" and "Sentiment Scores" in existing_outputs:
            self._show_sentiment_analysis_viz()
        elif viz_type == "Follower Distribution" and "High-Value Followers" in existing_outputs:
            self._show_follower_distribution_viz()
        elif viz_type == "Content Preferences" and "User Profiles" in existing_outputs:
            self._show_content_preferences_viz()
        else:
            st.info(f"📊 {viz_type} visualization requires additional data processing.")
    
    def _show_model_performance_viz(self):
        """Show model performance visualizations"""
        try:
            with open("outputs/metrics.json", "r") as f:
                metrics = json.load(f)
            
            if not metrics:
                st.warning("⚠️ No model metrics found.")
                return
            
            # Performance comparison
            metrics_df = pd.DataFrame(metrics).T
            
            if len(metrics_df) == 0:
                st.warning("⚠️ No model performance data to visualize.")
                return
            
            # Multiple metrics comparison using radar chart with go.Scatterpolar
            if len(metrics_df) > 0:
                fig = go.Figure()
                
                # Create radar chart manually using scatterpolar
                metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
                available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
                
                if available_metrics:
                    for model_name in metrics_df.index:
                        values = [metrics_df.loc[model_name, metric] if metric in metrics_df.columns else 0 
                                 for metric in available_metrics]
                        # Close the radar chart by repeating first value
                        values.append(values[0])
                        metrics_labels = available_metrics + [available_metrics[0]]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=metrics_labels,
                            fill='toself',
                            name=model_name
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )),
                        showlegend=True,
                        title="Model Performance Radar Chart"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="model_performance_radar")
                else:
                    st.warning("⚠️ No performance metrics available for radar chart.")
            
            # Confusion matrices (if available)
            st.markdown("#### 🔄 Model Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                # Accuracy comparison
                if 'Accuracy' in metrics_df.columns:
                    fig = px.bar(
                        metrics_df.reset_index(),
                        x='index',
                        y='Accuracy',
                        title="Model Accuracy Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="model_accuracy_comparison")
                else:
                    st.info("Accuracy metrics not available")
            
            with col2:
                # F1-Score comparison
                if 'F1-Score' in metrics_df.columns:
                    fig = px.bar(
                        metrics_df.reset_index(),
                        x='index',
                        y='F1-Score',
                        title="Model F1-Score Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="model_f1_comparison")
                else:
                    st.info("F1-Score metrics not available")
                
        except Exception as e:
            st.error(f"Error loading model metrics: {str(e)}")
    
    def _show_engagement_profiles_viz(self):
        """Show engagement profiles visualizations"""
        try:
            with open("outputs/profiles.json", "r") as f:
                profiles = json.load(f)
            
            if not profiles:
                st.warning("⚠️ No engagement profiles found.")
                return
            
            # Extract engagement probabilities
            engagement_data = []
            for user, profile in profiles.items():
                if 'engagement_probability' in profile:
                    engagement_data.append({
                        'user': user,
                        'engagement_prob': profile['engagement_probability'],
                        'sentiment': profile.get('dominant_sentiment', 'neutral')
                    })
            
            if engagement_data:
                df = pd.DataFrame(engagement_data)
                
                # Engagement probability distribution
                fig = px.histogram(
                    df,
                    x='engagement_prob',
                    color='sentiment',
                    title="Engagement Probability Distribution by Sentiment",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True, key="engagement_prob_distribution")
                
                # Top users by engagement
                top_users = df.nlargest(10, 'engagement_prob')
                fig = px.bar(
                    top_users,
                    x='user',
                    y='engagement_prob',
                    color='sentiment',
                    title="Top 10 Users by Engagement Probability"
                )
                st.plotly_chart(fig, use_container_width=True, key="top_users_engagement")
            else:
                st.warning("⚠️ No engagement probability data found in profiles.")
                
        except Exception as e:
            st.error(f"Error loading profiles: {str(e)}")
    
    def _show_sentiment_analysis_viz(self):
        """Show sentiment analysis visualizations"""
        try:
            with open("outputs/sentiment_scores.json", "r") as f:
                sentiment_data = json.load(f)
            
            if not sentiment_data:
                st.warning("⚠️ No sentiment analysis data found.")
                return
            
            # Convert to DataFrame
            sentiment_df = pd.DataFrame([
                {"comment": k, **v} for k, v in sentiment_data.items()
            ])
            
            if len(sentiment_df) == 0:
                st.warning("⚠️ No sentiment data to visualize.")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution
                sentiment_counts = sentiment_df['sentiment'].value_counts()
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Overall Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True, key="viz_sentiment_distribution")
            
            with col2:
                # Confidence vs Sentiment
                fig = px.box(
                    sentiment_df,
                    x='sentiment',
                    y='confidence',
                    title="Confidence Scores by Sentiment"
                )
                st.plotly_chart(fig, use_container_width=True, key="viz_sentiment_confidence")
                
        except Exception as e:
            st.error(f"Error loading sentiment data: {str(e)}")
    
    def _show_follower_distribution_viz(self):
        """Show follower distribution visualizations"""
        try:
            from src.utils.high_value_utils import get_consolidated_high_value_followers
            followers = get_consolidated_high_value_followers()
            
            if not followers:
                st.warning("⚠️ No high-value followers found.")
                return
            
            # Follower metrics
            follower_data = []
            for username, data in followers.items():
                follower_data.append({
                    'username': username,
                    'engagement_score': data.get('engagement_score', 0),
                    'influence_score': data.get('influence_score', 0),
                    'total_score': data.get('total_score', 0)
                })
            
            if follower_data:
                df = pd.DataFrame(follower_data)
                
                # Scatter plot: Engagement vs Influence
                fig = px.scatter(
                    df,
                    x='engagement_score',
                    y='influence_score',
                    size='total_score',
                    hover_data=['username'],
                    title="High-Value Followers: Engagement vs Influence"
                )
                st.plotly_chart(fig, use_container_width=True, key="engagement_vs_influence_scatter")
                
                # Top followers
                top_followers = df.nlargest(10, 'total_score')
                fig = px.bar(
                    top_followers,
                    x='username',
                    y='total_score',
                    title="Top 10 High-Value Followers"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True, key="top_followers_bar")
            else:
                st.warning("⚠️ No follower data found to visualize.")
                
        except Exception as e:
            st.error(f"Error loading follower data: {str(e)}")
    
    def _show_content_preferences_viz(self):
        """Show content preferences visualizations"""
        try:
            with open("outputs/profiles.json", "r") as f:
                profiles = json.load(f)
            
            if not profiles:
                st.warning("⚠️ No user profiles found.")
                return
            
            # Extract content preferences
            content_data = []
            for user, profile in profiles.items():
                if 'content_preferences' in profile:
                    for content_type in profile['content_preferences']:
                        content_data.append({
                            'user': user,
                            'content_type': content_type,
                            'engagement_prob': profile.get('engagement_probability', 0)
                        })
            
            if content_data:
                df = pd.DataFrame(content_data)
                
                # Content type distribution
                content_counts = df['content_type'].value_counts()
                fig = px.bar(
                    x=content_counts.index,
                    y=content_counts.values,
                    title="Content Type Preferences Distribution"
                )
                st.plotly_chart(fig, use_container_width=True, key="content_type_distribution")
                
                # Engagement by content type
                avg_engagement = df.groupby('content_type')['engagement_prob'].mean().sort_values(ascending=False)
                fig = px.bar(
                    x=avg_engagement.index,
                    y=avg_engagement.values,
                    title="Average Engagement Probability by Content Type"
                )
                st.plotly_chart(fig, use_container_width=True, key="engagement_by_content_type")
            else:
                st.warning("⚠️ No content preference data found in profiles.")
                
        except Exception as e:
            st.error(f"Error loading content preferences: {str(e)}")
