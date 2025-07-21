"""
Sentiment Analysis UI Component - Memory Optimized Version
Handles BERT sentiment analysis interface with enhanced features and large dataset support
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
import random
import numpy as np
from .base import BaseUIComponent

class SentimentAnalysisComponent(BaseUIComponent):
    """Enhanced sentiment analysis component with robust error handling and memory optimization"""
    
    def __init__(self, app_instance=None):
        super().__init__(app_instance)
        self.sentiment_analyzer = None
        
    def _initialize_analyzer(self, use_fine_tuning=False):
        """Initialize the sentiment analyzer"""
        try:
            if self.sentiment_analyzer is None:
                from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
                self.sentiment_analyzer = BERTSentimentAnalyzer(use_fine_tuning=use_fine_tuning)
                return True
        except ImportError as e:
            st.error(f"❌ Could not import sentiment analyzer: {str(e)}")
            return False
        except Exception as e:
            st.error(f"❌ Error initializing sentiment analyzer: {str(e)}")
            return False
        return True
    
    def show(self):
        """Enhanced sentiment analysis interface"""
        st.markdown("### 🧠 Sentiment Analysis with BERT")
        st.markdown("*Analyze comment sentiment using fine-tuned BERT models*")
        
        # Check prerequisites
        if not os.path.exists("outputs/preprocessed_data.csv"):
            st.warning("⚠️ Please preprocess data first!")
            return
        
        # Load and validate data
        try:
            df = pd.read_csv("outputs/preprocessed_data.csv")
            st.info(f"📊 Dataset loaded: {len(df)} records")
            
            # Check for required columns
            required_cols = ['comment_text', 'comment_owner_username']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.info("💡 Required columns: comment_text, comment_owner_username, post_id (optional)")
                return
                
            # Show data summary
            valid_comments = df['comment_text'].dropna()
            st.success(f"✅ Found {len(valid_comments)} valid comments to analyze")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
        
        # Model configuration
        st.markdown("#### 🎛️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Model Selection**")
            model_options = [
                "distilbert-base-uncased",
                "bert-base-uncased", 
                "roberta-base",
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            ]
            selected_model = st.selectbox("Select BERT Model:", model_options)
            
            # Fine-tuning option
            use_fine_tuning = st.checkbox(
                "Enable Fine-tuning", 
                help="Use specified hyperparameters (LR: 2e-5, Batch: 16, Epochs: 3)"
            )
            
            if use_fine_tuning:
                st.info("🔧 Fine-tuning enabled with hyperparameters: LR=2e-5, Batch=16, Epochs=3")
        
        with col2:
            st.markdown("**⚙️ Processing Parameters**")
            batch_size = st.slider("Batch Size", 8, 64, 16, help="Number of comments processed together")
            max_length = st.slider("Max Sequence Length", 64, 512, 128, help="Maximum text length for BERT")
            use_gpu = st.checkbox("Use GPU (if available)", value=True)
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                chunk_size = st.number_input("Chunk Size for Large Datasets", 1000, 10000, 5000)
                save_detailed_results = st.checkbox("Save Detailed Results", value=True)
        
        # Analysis section
        st.markdown("#### 🚀 Run Analysis")
        
        # Initialize analyzer
        if not self._initialize_analyzer(use_fine_tuning):
            return
        
        # Analysis button
        col1, col2 = st.columns([2, 1])
        with col1:
            run_analysis = st.button("🧠 Analyze Comment Sentiment", type="primary")
        with col2:
            if len(valid_comments) > 1000:
                st.warning(f"⚠️ Large dataset ({len(valid_comments)} comments)")
                st.info("💡 Consider using chunked processing")
        
        if run_analysis:
            with st.spinner(f"Running sentiment analysis with {selected_model}..."):
                try:
                    # Show progress information
                    progress_container = st.container()
                    with progress_container:
                        st.info(f"🔄 Processing {len(valid_comments)} comments...")
                        st.info(f"📊 Model: {selected_model}")
                        st.info(f"⚙️ Settings: Batch={batch_size}, MaxLen={max_length}, GPU={use_gpu}")
                    
                    # Run sentiment analysis
                    sentiment_scores = self.sentiment_analyzer.analyze_sentiment(
                        df,
                        model_name=selected_model,
                        batch_size=batch_size,
                        max_length=max_length,
                        use_gpu=use_gpu
                    )
                    
                    if not sentiment_scores:
                        st.error("❌ No sentiment scores generated. Check your data.")
                        return
                    
                    # Save results with enhanced format
                    results_data = {
                        "sentiment_scores": sentiment_scores,
                        "metadata": {
                            "model_used": selected_model,
                            "batch_size": batch_size,
                            "max_length": max_length,
                            "total_comments": len(sentiment_scores),
                            "analysis_timestamp": pd.Timestamp.now().isoformat(),
                            "fine_tuning_enabled": use_fine_tuning,
                            "hyperparameters": {
                                "learning_rate": "2e-5",
                                "batch_size": 16,
                                "epochs": 3
                            } if use_fine_tuning else None
                        }
                    }
                    
                    with open("outputs/sentiment_scores.json", "w") as f:
                        json.dump(results_data, f, indent=2)
                    
                    st.success(f"✅ Sentiment analysis completed! Analyzed {len(sentiment_scores)} comments")
                    
                    # Show results
                    self._show_enhanced_sentiment_results(results_data, context="new")
                    
                except KeyError as e:
                    st.error(f"❌ Data validation error: {str(e)}")
                    st.info("💡 Check that your dataset has the required columns")
                except Exception as e:
                    st.error(f"❌ Error during sentiment analysis: {str(e)}")
                    st.info("💡 Try reducing batch size or using CPU instead of GPU")
        
        # Show existing results if available
        self._show_existing_results()

    def _show_existing_results(self):
        """Show existing sentiment analysis results if available"""
        if os.path.exists("outputs/sentiment_scores.json"):
            st.markdown("#### 📋 Previous Analysis Results")
            
            try:
                with open("outputs/sentiment_scores.json", "r") as f:
                    data = json.load(f)
                
                # Handle both old and new format
                if "sentiment_scores" in data:
                    sentiment_scores = data["sentiment_scores"]
                    metadata = data.get("metadata", {})
                else:
                    sentiment_scores = data
                    metadata = {}
                
                # Show metadata if available
                if metadata:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Comments Analyzed", metadata.get("total_comments", len(sentiment_scores)))
                    with col2:
                        timestamp = metadata.get("analysis_timestamp", "Unknown")
                        date_str = timestamp.split('T')[0] if 'T' in str(timestamp) else str(timestamp)
                        st.metric("Analysis Date", date_str)
                    with col3:
                        model = metadata.get("model_used", "Unknown")
                        st.metric("Model Used", model)
                
                if st.button("🔄 Load Previous Results"):
                    self._show_enhanced_sentiment_results(data, context="existing")
                    
            except Exception as e:
                st.warning(f"⚠️ Error loading previous results: {str(e)}")
        else:
            st.info("ℹ️ No previous sentiment analysis found. Run a new analysis to see results.")

    def _show_enhanced_sentiment_results(self, sentiment_data, context="default"):
        """Enhanced display with comprehensive visualizations and robust error handling for large datasets"""
        
        # Add high-precision timestamp and random component to ensure unique keys
        timestamp = str(time.time()).replace('.', '')
        random_id = random.randint(1000, 9999)
        key_suffix = f"{context}_{timestamp}_{random_id}"
        
        # Extract sentiment scores and metadata
        try:
            if isinstance(sentiment_data, dict) and "sentiment_scores" in sentiment_data:
                # New format with metadata
                sentiment_scores = sentiment_data["sentiment_scores"]
                metadata = sentiment_data.get("metadata", {})
            else:
                # Old format - direct scores
                sentiment_scores = sentiment_data
                metadata = {}
            
            if not sentiment_scores:
                st.warning("⚠️ No sentiment scores to display")
                return
                
            # Check dataset size and apply sampling for large datasets
            data_size = len(sentiment_scores)
            
            if data_size > 3000:
                st.warning(f"⚠️ Large dataset detected ({data_size:,} comments)")
                st.info("🔄 Using optimized visualization for large datasets")
                
                # Sample data for visualizations to prevent memory issues
                sample_size = min(2000, data_size)
                sampled_keys = random.sample(list(sentiment_scores.keys()), sample_size)
                viz_scores = {k: sentiment_scores[k] for k in sampled_keys}
                
                st.info(f"📊 Showing visualizations for {sample_size:,} sampled comments (out of {data_size:,} total)")
            else:
                viz_scores = sentiment_scores
            
            # Convert to DataFrame efficiently with error handling
            processed_data = []
            skipped_count = 0
            
            for k, v in viz_scores.items():
                try:
                    # Validate data structure
                    if not isinstance(v, dict):
                        skipped_count += 1
                        continue
                        
                    processed_data.append({
                        "comment_key": str(k)[:50],  # Truncate long keys
                        "comment_owner_username": str(v.get("comment_owner_username", "unknown"))[:30],
                        "post_owner_username": str(v.get("post_owner_username", "unknown"))[:30],  # Add post owner
                        "comment_text": str(v.get("comment_text", ""))[:150],  # Truncate long comments for display
                        "comment_full_text": str(v.get("comment_full_text", v.get("comment_text", ""))),  # Keep full text
                        "post_id": str(v.get("post_id", ""))[:20],
                        "sentiment": str(v.get("sentiment", "neutral")),
                        "confidence": float(v.get("confidence", 0.0)),
                        "positive": float(v.get("positive", 0.0)),
                        "negative": float(v.get("negative", 0.0)),
                        "neutral": float(v.get("neutral", 0.0)),
                        "comment_likes": int(v.get("comment_likes", 0)),
                        "post_likes": int(v.get("post_likes", 0)),
                        "engagement_rate": float(v.get("engagement_rate", 0.0))
                    })
                except (ValueError, TypeError, KeyError) as e:
                    skipped_count += 1
                    continue
            
            if not processed_data:
                st.error("❌ No valid sentiment data found")
                return
            
            if skipped_count > 0:
                st.warning(f"⚠️ Skipped {skipped_count} malformed entries")
            
            sentiment_df = pd.DataFrame(processed_data)
            
        except Exception as e:
            st.error(f"❌ Error processing sentiment data: {str(e)}")
            return
        
        st.markdown("#### 📊 Sentiment Analysis Results")
        
        # Summary metrics
        st.markdown("##### 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Comments", f"{data_size:,}" if data_size != len(sentiment_df) else len(sentiment_df))
        with col2:
            avg_confidence = sentiment_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        with col3:
            dominant_sentiment = sentiment_df['sentiment'].mode().iloc[0] if not sentiment_df.empty else "Unknown"
            st.metric("Dominant Sentiment", dominant_sentiment.title())
        with col4:
            high_confidence = (sentiment_df['confidence'] > 0.8).sum()
            st.metric("High Confidence", f"{high_confidence:,} ({high_confidence/len(sentiment_df)*100:.1f}%)")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🥧 Distribution", "📊 Confidence", "👤 By Post Owner", "💬 By Comment User", "📝 Sample Comments"])
        
        with tab1:
            self._show_distribution_tab(sentiment_df, key_suffix)
        
        with tab2:
            self._show_confidence_tab(sentiment_df, key_suffix)
        
        with tab3:
            self._show_post_owner_analysis(sentiment_df, key_suffix)
        
        with tab4:
            self._show_comment_user_analysis(sentiment_df, key_suffix)
        
        with tab5:
            self._show_sample_comments(sentiment_df, key_suffix)
        
        # Export options
        st.markdown("##### 📥 Export Results")
        self._show_export_options(sentiment_scores, sentiment_df, key_suffix)

    def _show_distribution_tab(self, sentiment_df, key_suffix):
        """Display sentiment distribution visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Sentiment distribution pie chart
                sentiment_counts = sentiment_df['sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True, key=f"sentiment_pie_{key_suffix}")
            except Exception as e:
                st.error(f"Error creating pie chart: {str(e)}")
        
        with col2:
            try:
                # Sentiment scores distribution as bar chart (more memory efficient than box plot)
                avg_scores = sentiment_df[['positive', 'negative', 'neutral']].mean()
                fig_scores = px.bar(
                    x=avg_scores.index,
                    y=avg_scores.values,
                    title="Average Sentiment Scores",
                    color=avg_scores.index,
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                st.plotly_chart(fig_scores, use_container_width=True, key=f"sentiment_bar_{key_suffix}")
            except Exception as e:
                st.error(f"Error creating bar chart: {str(e)}")

    def _show_confidence_tab(self, sentiment_df, key_suffix):
        """Display confidence analysis visualizations"""
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Confidence histogram - reduced bins for large datasets
                bins = min(20, max(5, len(sentiment_df) // 100))
                fig_conf = px.histogram(
                    sentiment_df, 
                    x='confidence',
                    title="Confidence Score Distribution",
                    nbins=bins,
                    color='sentiment',
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                st.plotly_chart(fig_conf, use_container_width=True, key=f"confidence_hist_{key_suffix}")
            except Exception as e:
                st.error(f"Error creating confidence histogram: {str(e)}")
        
        with col2:
            try:
                # Confidence statistics instead of scatter plot for large datasets
                conf_stats = sentiment_df.groupby('sentiment')['confidence'].agg(['mean', 'std', 'count']).round(3)
                st.markdown("**Confidence Statistics by Sentiment**")
                st.dataframe(conf_stats)
                
                # Simple confidence vs sentiment bar chart
                fig_conf_mean = px.bar(
                    x=conf_stats.index,
                    y=conf_stats['mean'],
                    title="Average Confidence by Sentiment",
                    color=conf_stats.index,
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                st.plotly_chart(fig_conf_mean, use_container_width=True, key=f"confidence_mean_{key_suffix}")
            except Exception as e:
                st.error(f"Error creating confidence analysis: {str(e)}")

    def _show_user_analysis_tab(self, sentiment_df, key_suffix):
        """Display user-level sentiment analysis"""
        if 'username' not in sentiment_df.columns:
            st.info("👥 User analysis not available - missing username data")
            return
        
        try:
            # User-level sentiment analysis - limit to top users for performance
            user_sentiment = sentiment_df.groupby('username').agg({
                'sentiment': lambda x: x.mode().iloc[0] if not x.empty else 'neutral',
                'confidence': 'mean',
                'positive': 'mean',
                'negative': 'mean',
                'neutral': 'mean',
                'comment_key': 'count'
            }).rename(columns={'comment_key': 'comment_count'})
            
            user_sentiment = user_sentiment.sort_values('comment_count', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top users by comment count (limited to 10 for performance)
                top_users = user_sentiment.head(10)
                fig_users = px.bar(
                    x=top_users.index,
                    y=top_users['comment_count'],
                    color=top_users['sentiment'],
                    title="Top 10 Users by Comment Count",
                    labels={'x': 'Username', 'y': 'Comments'},
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                fig_users.update_xaxes(tickangle=45)
                st.plotly_chart(fig_users, use_container_width=True, key=f"users_bar_{key_suffix}")
            
            with col2:
                # User sentiment summary stats
                st.markdown("**User Sentiment Summary**")
                user_summary = {
                    "Total Users": len(user_sentiment),
                    "Avg Comments/User": round(user_sentiment['comment_count'].mean(), 1),
                    "Most Active User": user_sentiment.index[0],
                    "Max Comments": user_sentiment['comment_count'].max()
                }
                for k, v in user_summary.items():
                    st.metric(k, v)
            
            # User summary table (limited for performance)
            st.markdown("**📊 Top Users Summary**")
            display_users = user_sentiment.head(20)
            st.dataframe(display_users, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error in user analysis: {str(e)}")

    def _show_post_owner_analysis(self, sentiment_df, key_suffix):
        """Show sentiment analysis by post owners"""
        st.markdown("##### 👤 Sentiment Analysis by Post Owner")
        
        try:
            # Group by post owner
            post_owner_stats = sentiment_df.groupby('post_owner_username').agg({
                'sentiment': lambda x: x.value_counts().to_dict(),
                'confidence': 'mean',
                'post_id': 'nunique',
                'comment_owner_username': 'count'
            }).round(3)
            
            post_owner_stats.columns = ['Sentiment_Distribution', 'Avg_Confidence', 'Unique_Posts', 'Total_Comments']
            
            # Show top post owners by comment count
            top_post_owners = post_owner_stats.nlargest(10, 'Total_Comments')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Top Post Owners by Comment Volume**")
                fig_owners = px.bar(
                    x=top_post_owners.index,
                    y=top_post_owners['Total_Comments'],
                    title="Comments Received by Post Owner",
                    labels={'x': 'Post Owner', 'y': 'Number of Comments'}
                )
                fig_owners.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_owners, use_container_width=True, key=f"post_owners_bar_{key_suffix}")
            
            with col2:
                st.markdown("**Average Confidence by Post Owner**")
                fig_conf = px.bar(
                    x=top_post_owners.index,
                    y=top_post_owners['Avg_Confidence'],
                    title="Average Sentiment Confidence by Post Owner",
                    labels={'x': 'Post Owner', 'y': 'Average Confidence'}
                )
                fig_conf.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_conf, use_container_width=True, key=f"post_owners_conf_{key_suffix}")
            
            # Post-level sentiment analysis
            st.markdown("##### 📱 Sentiment Analysis by Post")
            
            # Group by post_id and post_owner
            post_stats = sentiment_df.groupby(['post_id', 'post_owner_username']).agg({
                'sentiment': lambda x: x.value_counts().to_dict(),
                'confidence': 'mean',
                'comment_owner_username': 'count',
                'post_likes': 'first',
                'engagement_rate': 'mean'
            }).round(3)
            
            post_stats.columns = ['Sentiment_Distribution', 'Avg_Confidence', 'Comment_Count', 'Post_Likes', 'Avg_Engagement']
            
            # Show posts with most comments
            top_posts = post_stats.nlargest(10, 'Comment_Count')
            
            if not top_posts.empty:
                st.markdown("**Posts with Most Comments**")
                st.dataframe(
                    top_posts[['Comment_Count', 'Avg_Confidence', 'Post_Likes', 'Avg_Engagement']],
                    use_container_width=True
                )
                
                # Sentiment distribution for top posts
                col3, col4 = st.columns(2)
                
                with col3:
                    # Create post sentiment visualization
                    post_sentiment_data = []
                    for (post_id, owner), row in top_posts.head(5).iterrows():
                        sentiment_dist = row['Sentiment_Distribution']
                        for sentiment, count in sentiment_dist.items():
                            post_sentiment_data.append({
                                'Post': f"{owner[:15]}...\n({post_id[:8]}...)",
                                'Sentiment': sentiment,
                                'Count': count
                            })
                    
                    if post_sentiment_data:
                        post_sentiment_df = pd.DataFrame(post_sentiment_data)
                        fig_post_sent = px.bar(
                            post_sentiment_df,
                            x='Post',
                            y='Count',
                            color='Sentiment',
                            title="Sentiment Distribution for Top 5 Posts",
                            color_discrete_map={
                                'positive': '#2E8B57',
                                'negative': '#DC143C',
                                'neutral': '#4682B4'
                            }
                        )
                        fig_post_sent.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_post_sent, use_container_width=True, key=f"post_sentiment_{key_suffix}")
                
                with col4:
                    # Post engagement vs sentiment
                    engagement_sentiment = []
                    for (post_id, owner), row in top_posts.iterrows():
                        sentiment_dist = row['Sentiment_Distribution']
                        positive_ratio = sentiment_dist.get('positive', 0) / row['Comment_Count']
                        engagement_sentiment.append({
                            'Post_Owner': owner[:20],
                            'Positive_Ratio': positive_ratio,
                            'Engagement_Rate': row['Avg_Engagement'],
                            'Comment_Count': row['Comment_Count']
                        })
                    
                    if engagement_sentiment:
                        engagement_df = pd.DataFrame(engagement_sentiment)
                        fig_engagement = px.scatter(
                            engagement_df,
                            x='Positive_Ratio',
                            y='Engagement_Rate',
                            size='Comment_Count',
                            hover_data=['Post_Owner'],
                            title="Engagement vs Positive Sentiment",
                            labels={
                                'Positive_Ratio': 'Positive Sentiment Ratio',
                                'Engagement_Rate': 'Engagement Rate'
                            }
                        )
                        st.plotly_chart(fig_engagement, use_container_width=True, key=f"engagement_sentiment_{key_suffix}")
            
        except Exception as e:
            st.error(f"Error in post owner analysis: {str(e)}")
    
    def _show_comment_user_analysis(self, sentiment_df, key_suffix):
        """Show sentiment analysis by comment users"""
        st.markdown("##### 💬 Sentiment Analysis by Comment Users")
        
        try:
            # Group by comment users
            user_stats = sentiment_df.groupby('comment_owner_username').agg({
                'sentiment': lambda x: x.value_counts().to_dict(),
                'confidence': 'mean',
                'post_owner_username': 'nunique',
                'comment_likes': 'sum',
                'engagement_rate': 'mean'
            }).round(3)
            
            user_stats.columns = ['Sentiment_Distribution', 'Avg_Confidence', 'Posts_Commented', 'Total_Likes', 'Avg_Engagement']
            
            # Calculate user activity metrics
            user_stats['Total_Comments'] = sentiment_df.groupby('comment_owner_username').size()
            user_stats['Positive_Ratio'] = user_stats['Sentiment_Distribution'].apply(
                lambda x: x.get('positive', 0) / sum(x.values()) if sum(x.values()) > 0 else 0
            )
            
            # Show top commenters
            top_commenters = user_stats.nlargest(15, 'Total_Comments')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Most Active Comment Users**")
                fig_users = px.bar(
                    x=top_commenters.index[:10],
                    y=top_commenters['Total_Comments'][:10],
                    title="Top Comment Users by Activity",
                    labels={'x': 'Comment User', 'y': 'Number of Comments'}
                )
                fig_users.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_users, use_container_width=True, key=f"comment_users_bar_{key_suffix}")
            
            with col2:
                st.markdown("**User Sentiment Positivity**")
                positive_users = top_commenters.nlargest(10, 'Positive_Ratio')
                fig_positive = px.bar(
                    x=positive_users.index,
                    y=positive_users['Positive_Ratio'],
                    title="Most Positive Comment Users",
                    labels={'x': 'Comment User', 'y': 'Positive Sentiment Ratio'},
                    color=positive_users['Positive_Ratio'],
                    color_continuous_scale='Greens'
                )
                fig_positive.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_positive, use_container_width=True, key=f"positive_users_{key_suffix}")
            
            # User engagement analysis
            st.markdown("**User Engagement vs Sentiment Analysis**")
            
            # Filter users with multiple comments for better analysis
            active_users = user_stats[user_stats['Total_Comments'] >= 2].head(20)
            
            if not active_users.empty:
                col3, col4 = st.columns(2)
                
                with col3:
                    fig_scatter = px.scatter(
                        active_users,
                        x='Total_Comments',
                        y='Avg_Confidence',
                        size='Posts_Commented',
                        color='Positive_Ratio',
                        hover_data=['Total_Likes'],
                        title="User Activity vs Confidence",
                        labels={
                            'Total_Comments': 'Number of Comments',
                            'Avg_Confidence': 'Average Confidence',
                            'Positive_Ratio': 'Positive Ratio'
                        },
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True, key=f"user_scatter_{key_suffix}")
                
                with col4:
                    # User sentiment distribution
                    user_sentiment_data = []
                    for user, row in active_users.head(8).iterrows():
                        sentiment_dist = row['Sentiment_Distribution']
                        for sentiment, count in sentiment_dist.items():
                            user_sentiment_data.append({
                                'User': user[:20] + '...' if len(user) > 20 else user,
                                'Sentiment': sentiment,
                                'Count': count
                            })
                    
                    if user_sentiment_data:
                        user_sentiment_df = pd.DataFrame(user_sentiment_data)
                        fig_user_sent = px.bar(
                            user_sentiment_df,
                            x='User',
                            y='Count',
                            color='Sentiment',
                            title="Sentiment Distribution by Active Users",
                            color_discrete_map={
                                'positive': '#2E8B57',
                                'negative': '#DC143C',
                                'neutral': '#4682B4'
                            }
                        )
                        fig_user_sent.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig_user_sent, use_container_width=True, key=f"user_sentiment_dist_{key_suffix}")
                
                # Summary statistics
                st.markdown("**Comment User Summary Statistics**")
                summary_stats = pd.DataFrame({
                    'Metric': [
                        'Total Unique Comment Users',
                        'Average Comments per User',
                        'Most Active User Comments',
                        'Average Positive Ratio',
                        'Users with High Confidence (>0.8)'
                    ],
                    'Value': [
                        len(user_stats),
                        f"{user_stats['Total_Comments'].mean():.1f}",
                        user_stats['Total_Comments'].max(),
                        f"{user_stats['Positive_Ratio'].mean():.3f}",
                        (user_stats['Avg_Confidence'] > 0.8).sum()
                    ]
                })
                st.dataframe(summary_stats, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error in comment user analysis: {str(e)}")
    
    def _show_sample_comments(self, sentiment_df, key_suffix):
        """Display sample comments for each sentiment category"""
        st.markdown("##### 📝 Sample Comments by Sentiment")
        
        try:
            # Get samples for each sentiment
            sentiment_categories = ['positive', 'negative', 'neutral']
            
            for sentiment in sentiment_categories:
                sentiment_data = sentiment_df[sentiment_df['sentiment'] == sentiment]
                
                if not sentiment_data.empty:
                    # Get highest confidence samples
                    top_samples = sentiment_data.nlargest(5, 'confidence')
                    
                    emoji_map = {'positive': '😊', 'negative': '😞', 'neutral': '😐'}
                    color_map = {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#4682B4'}
                    
                    st.markdown(f"**{emoji_map[sentiment]} {sentiment.title()} Comments (Top {len(top_samples)} by confidence)**")
                    
                    for idx, row in top_samples.iterrows():
                        comment_text = row.get('comment_text', row.get('comment_full_text', 'No text available'))
                        confidence = row.get('confidence', 0.0)
                        username = row.get('comment_owner_username', 'Unknown')
                        
                        # Truncate very long comments for display
                        display_text = comment_text[:200] + "..." if len(str(comment_text)) > 200 else comment_text
                        
                        with st.container():
                            st.markdown(f"""
                            <div style="border-left: 4px solid {color_map[sentiment]}; padding: 10px; margin: 5px 0; background-color: #f8f9fa;">
                                <strong>@{username}</strong> (Confidence: {confidence:.3f})<br>
                                <em>"{display_text}"</em>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                else:
                    st.info(f"No {sentiment} comments found in the dataset")
                    
        except Exception as e:
            st.error(f"Error displaying sample comments: {str(e)}")

    def _show_export_options(self, sentiment_scores, sentiment_df, key_suffix):
        """Display export options for the analysis results"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                # Download sample results as JSON (not full dataset to prevent memory issues)
                sample_size = min(1000, len(sentiment_scores))
                if len(sentiment_scores) > 1000:
                    sampled_keys = random.sample(list(sentiment_scores.keys()), sample_size)
                    sample_scores = {k: sentiment_scores[k] for k in sampled_keys}
                    results_json = json.dumps(sample_scores, indent=2)
                    label = f"📄 Download Sample JSON ({sample_size} items)"
                else:
                    results_json = json.dumps(sentiment_scores, indent=2)
                    label = "📄 Download Full JSON Results"
                    
                st.download_button(
                    label=label,
                    data=results_json,
                    file_name=f"sentiment_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error preparing JSON download: {str(e)}")
        
        with col2:
            try:
                # Download summary as CSV
                csv_data = sentiment_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Summary",
                    data=csv_data,
                    file_name=f"sentiment_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error preparing CSV download: {str(e)}")
        
        with col3:
            # Save detailed analysis
            if st.button("💾 Save Analysis Report", key=f"save_detailed_{key_suffix}"):
                try:
                    # Create summary report instead of full data for large datasets
                    analysis_report = {
                        "summary_statistics": {
                            "total_comments": len(sentiment_scores),
                            "average_confidence": float(sentiment_df['confidence'].mean()),
                            "sentiment_distribution": sentiment_df['sentiment'].value_counts().to_dict(),
                            "high_confidence_count": int((sentiment_df['confidence'] > 0.8).sum()),
                            "analysis_timestamp": pd.Timestamp.now().isoformat()
                        }
                    }
                    
                    # Only include full scores for smaller datasets
                    if len(sentiment_scores) <= 5000:
                        analysis_report["sentiment_scores"] = sentiment_scores
                    
                    with open("outputs/sentiment_analysis_report.json", "w") as f:
                        json.dump(analysis_report, f, indent=2)
                    
                    st.success("✅ Analysis report saved to outputs/sentiment_analysis_report.json")
                    
                except Exception as e:
                    st.error(f"❌ Error saving analysis: {str(e)}")
