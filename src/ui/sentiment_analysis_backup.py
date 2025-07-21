"""
Sentiment Analysis UI Component
Handles BERT sentiment analysis interface with enhanced features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
import random
from .base import BaseUIComponent

class SentimentAnalysisComponent(BaseUIComponent):
    """Enhanced sentiment analysis component with robust error handling"""
    
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
                    self._show_enhanced_sentiment_results(sentiment_scores, context="new")
                    
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
                    self._show_enhanced_sentiment_results(sentiment_scores, context="existing")
                    
            except Exception as e:
                st.warning(f"⚠️ Error loading previous results: {str(e)}")
        else:
            st.info("ℹ️ No previous sentiment analysis found. Run a new analysis to see results.")

    def _show_sentiment_results(self, sentiment_scores, context="default"):
        """Display sentiment analysis results"""
        if not sentiment_scores:
            return
        
        # Add high-precision timestamp and random component to ensure unique keys
        timestamp = str(time.time()).replace('.', '')
        random_id = random.randint(1000, 9999)
        key_suffix = f"{context}_{timestamp}_{random_id}"
        
        # Convert to DataFrame for visualization
        sentiment_df = pd.DataFrame([
            {"comment": k, **v} for k, v in sentiment_scores.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            sentiment_counts = sentiment_df['sentiment'].value_counts()
            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"sentiment_distribution_pie_{key_suffix}")
        
        with col2:
            # Confidence scores
            fig = px.histogram(
                sentiment_df, 
                x='confidence',
                title="Confidence Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True, key=f"sentiment_confidence_histogram_{key_suffix}")
        
        # Sample results
        with st.expander("📊 Sample Results"):
            st.dataframe(sentiment_df.head(10))
    
    def _show_enhanced_sentiment_results(self, sentiment_scores, context="default"):
        """Display enhanced sentiment analysis results with comprehensive visualizations"""
        if not sentiment_scores:
            st.warning("⚠️ No sentiment scores to display")
            return
        
        st.markdown("#### 📊 Sentiment Analysis Results")
        
        # Add unique key suffix for Streamlit widgets
        timestamp = str(time.time()).replace('.', '')
        random_id = random.randint(1000, 9999)
        key_suffix = f"{context}_{timestamp}_{random_id}"
        
        # Convert to DataFrame for analysis
        sentiment_data = []
        for comment_key, data in sentiment_scores.items():
            sentiment_data.append({
                "comment_key": comment_key,
                "sentiment": data.get('sentiment', 'unknown'),
                "confidence": data.get('confidence', 0.0),
                "positive": data.get('positive', 0.0),
                "negative": data.get('negative', 0.0),
                "neutral": data.get('neutral', 0.0),
                "post_id": data.get('post_id', 'unknown'),
                "username": data.get('comment_owner_username', 'unknown'),
                "comment_text": data.get('comment_text', '')
            })
        
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Summary metrics
        st.markdown("##### 📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Comments", len(sentiment_df))
        with col2:
            avg_confidence = sentiment_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.3f}")
        with col3:
            dominant_sentiment = sentiment_df['sentiment'].mode().iloc[0] if not sentiment_df.empty else "Unknown"
            st.metric("Dominant Sentiment", dominant_sentiment.title())
        with col4:
            high_confidence = (sentiment_df['confidence'] > 0.8).sum()
            st.metric("High Confidence", f"{high_confidence} ({high_confidence/len(sentiment_df)*100:.1f}%)")
        
        # Visualizations
        st.markdown("##### 📊 Sentiment Visualizations")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["🥧 Distribution", "📊 Confidence", "👥 By User", "📝 Sample Comments"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
                # Sentiment scores distribution
                fig_scores = go.Figure()
                
                fig_scores.add_trace(go.Box(
                    y=sentiment_df['positive'],
                    name='Positive',
                    marker_color='#2E8B57'
                ))
                fig_scores.add_trace(go.Box(
                    y=sentiment_df['negative'],
                    name='Negative',
                    marker_color='#DC143C'
                ))
                fig_scores.add_trace(go.Box(
                    y=sentiment_df['neutral'],
                    name='Neutral',
                    marker_color='#4682B4'
                ))
                
                fig_scores.update_layout(
                    title="Sentiment Score Distributions",
                    yaxis_title="Score",
                    showlegend=True
                )
                st.plotly_chart(fig_scores, use_container_width=True, key=f"sentiment_box_{key_suffix}")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence histogram
                fig_conf = px.histogram(
                    sentiment_df, 
                    x='confidence',
                    title="Confidence Score Distribution",
                    nbins=20,
                    color='sentiment',
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                st.plotly_chart(fig_conf, use_container_width=True, key=f"confidence_hist_{key_suffix}")
            
            with col2:
                # Confidence vs sentiment scatter
                fig_scatter = px.scatter(
                    sentiment_df, 
                    x='confidence', 
                    y='positive',
                    color='sentiment',
                    title="Confidence vs Positive Score",
                    hover_data=['username', 'comment_text'],
                    color_discrete_map={
                        'positive': '#2E8B57',
                        'negative': '#DC143C', 
                        'neutral': '#4682B4'
                    }
                )
                st.plotly_chart(fig_scatter, use_container_width=True, key=f"confidence_scatter_{key_suffix}")
        
        with tab3:
            # User-level sentiment analysis
            if 'username' in sentiment_df.columns:
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
                    # Top users by comment count
                    top_users = user_sentiment.head(10)
                    fig_users = px.bar(
                        x=top_users.index,
                        y=top_users['comment_count'],
                        color=top_users['sentiment'],
                        title="Top Users by Comment Count",
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
                    # User sentiment heatmap
                    if len(user_sentiment) > 1:
                        sentiment_matrix = user_sentiment[['positive', 'negative', 'neutral']].head(10)
                        fig_heatmap = px.imshow(
                            sentiment_matrix.T,
                            x=sentiment_matrix.index,
                            y=['Positive', 'Negative', 'Neutral'],
                            title="User Sentiment Heatmap",
                            color_continuous_scale='RdYlBu_r'
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True, key=f"users_heatmap_{key_suffix}")
                
                # User summary table
                st.markdown("**📊 User Sentiment Summary**")
                st.dataframe(user_sentiment.head(20), use_container_width=True)
        
        with tab4:
            # Sample comments by sentiment
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**😊 Most Positive Comments**")
                positive_comments = sentiment_df[sentiment_df['sentiment'] == 'positive'].nlargest(5, 'confidence')
                for _, comment in positive_comments.iterrows():
                    with st.expander(f"@{comment['username']} ({comment['confidence']:.3f})"):
                        st.write(comment['comment_text'])
            
            with col2:
                st.markdown("**😐 Most Neutral Comments**")
                neutral_comments = sentiment_df[sentiment_df['sentiment'] == 'neutral'].nlargest(5, 'confidence')
                for _, comment in neutral_comments.iterrows():
                    with st.expander(f"@{comment['username']} ({comment['confidence']:.3f})"):
                        st.write(comment['comment_text'])
            
            with col3:
                st.markdown("**😞 Most Negative Comments**")
                negative_comments = sentiment_df[sentiment_df['sentiment'] == 'negative'].nlargest(5, 'confidence')
                for _, comment in negative_comments.iterrows():
                    with st.expander(f"@{comment['username']} ({comment['confidence']:.3f})"):
                        st.write(comment['comment_text'])
        
        # Download options
        st.markdown("##### 📥 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download full results as JSON
            results_json = json.dumps(sentiment_scores, indent=2)
            st.download_button(
                label="📄 Download JSON Results",
                data=results_json,
                file_name=f"sentiment_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Download summary as CSV
            csv_data = sentiment_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Summary",
                data=csv_data,
                file_name=f"sentiment_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            # Save detailed analysis
            if st.button("💾 Save Detailed Analysis", key=f"save_detailed_{key_suffix}"):
                try:
                    # Create comprehensive analysis report
                    analysis_report = {
                        "sentiment_scores": sentiment_scores,
                        "summary_statistics": {
                            "total_comments": len(sentiment_df),
                            "average_confidence": float(sentiment_df['confidence'].mean()),
                            "sentiment_distribution": sentiment_df['sentiment'].value_counts().to_dict(),
                            "high_confidence_count": int((sentiment_df['confidence'] > 0.8).sum()),
                            "analysis_timestamp": pd.Timestamp.now().isoformat()
                        },
                        "user_analysis": user_sentiment.to_dict() if 'username' in sentiment_df.columns else None
                    }
                    
                    with open("outputs/detailed_sentiment_analysis.json", "w") as f:
                        json.dump(analysis_report, f, indent=2)
                    
                    st.success("✅ Detailed analysis saved to outputs/detailed_sentiment_analysis.json")
                    
                except Exception as e:
                    st.error(f"❌ Error saving analysis: {str(e)}")

    def _show_sentiment_results(self, sentiment_scores, context="default"):
        """Legacy method for backward compatibility - redirects to enhanced version"""
        self._show_enhanced_sentiment_results(sentiment_scores, context)
