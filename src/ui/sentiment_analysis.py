"""
Sentiment Analysis UI Component
Handles BERT sentiment analysis interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
import time
import random
from .base import BaseUIComponent


class SentimentAnalysisComponent(BaseUIComponent):
    """Sentiment analysis component"""
    
    def show(self):
        """Sentiment analysis interface"""
        st.markdown("### 🧠 Sentiment Analysis with BERT")
        
        # Check prerequisites
        if not os.path.exists("outputs/preprocessed_data.csv"):
            st.warning("⚠️ Please preprocess data first!")
            return
        
        # BERT model selection
        model_options = [
            "distilbert-base-uncased",
            "bert-base-uncased", 
            "roberta-base",
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ]
        
        col1, col2 = st.columns(2)
        with col1:
            selected_model = st.selectbox("Select BERT Model:", model_options)
            batch_size = st.slider("Batch Size", 8, 64, 16)
        
        with col2:
            max_length = st.slider("Max Sequence Length", 64, 512, 128)
            use_gpu = st.checkbox("Use GPU (if available)", value=True)
        
        # Analysis button
        if st.button("🧠 Analyze Sentiment", type="primary"):
            with st.spinner(f"Running sentiment analysis with {selected_model}..."):
                try:
                    df = pd.read_csv("outputs/preprocessed_data.csv")
                    
                    sentiment_scores = self.sentiment_analyzer.analyze_sentiment(
                        df,
                        model_name=selected_model,
                        batch_size=batch_size,
                        max_length=max_length,
                        use_gpu=use_gpu
                    )
                    
                    # Save results
                    with open("outputs/sentiment_scores.json", "w") as f:
                        json.dump(sentiment_scores, f)
                    st.success("✅ Sentiment analysis completed!")
                    
                    # Show results
                    self._show_sentiment_results(sentiment_scores, context="new")
                    
                except Exception as e:
                    st.error(f"❌ Error during sentiment analysis: {str(e)}")
        
        # Show existing results if available
        if os.path.exists("outputs/sentiment_scores.json"):
            with open("outputs/sentiment_scores.json", "r") as f:
                existing_scores = json.load(f)
            
            st.info(f"📊 Previously analyzed: {len(existing_scores)} comments")
            self._show_sentiment_results(existing_scores, context="existing")

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
