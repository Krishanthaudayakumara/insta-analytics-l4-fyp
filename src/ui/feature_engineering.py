"""
Feature Engineering UI Component for Instagram Engagement Modeling
Aggregates comment-level data to post-level, merges sentiment, and computes features for modeling.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from .base import BaseUIComponent

class FeatureEngineeringComponent(BaseUIComponent):
    """Feature engineering UI for post-level sentiment-weighted engagement"""
    def __init__(self, app_instance=None):
        super().__init__(app_instance)

    def run_feature_engineering(self,
        preprocessed_path="outputs/preprocessed_data.csv",
        sentiment_path="outputs/sentiment_scores.json",
        output_path="outputs/engineered_data.csv"
    ):
        # Import the robust pipeline from src/feature_engineering/pipeline.py
        from src.feature_engineering.pipeline import run_feature_engineering as pipeline_run
        return pipeline_run(preprocessed_path, sentiment_path, output_path)

    def filter_posts_with_comments(self, input_path="outputs/engineered_data.csv", output_path="outputs/engineered_data_filtered.csv"):
        from src.feature_engineering.pipeline import filter_posts_with_comments as pipeline_filter
        return pipeline_filter(input_path, output_path)

    def show(self):
        st.markdown("### 🛠️ Feature Engineering: Sentiment-Weighted Engagement")
        st.markdown("*Aggregate sentiment and engagement features at the post level for modeling*")
        if not os.path.exists("outputs/preprocessed_data.csv"):
            st.warning("⚠️ Please preprocess data first!")
            return
        if not os.path.exists("outputs/sentiment_scores.json"):
            st.warning("⚠️ Please run sentiment analysis first!")
            return
        if st.button("Run Feature Engineering", type="primary"):
            with st.spinner("Running feature engineering..."):
                df = self.run_feature_engineering()
                st.success("Feature engineering complete! See outputs/engineered_data.csv.")
                st.dataframe(df.head(20))
        if os.path.exists("outputs/engineered_data.csv"):
            st.subheader("Preview of Engineered Data")
            df = pd.read_csv("outputs/engineered_data.csv")
            st.dataframe(df.head(20))
            if st.button("Filter Out Posts with Zero Comments", type="secondary"):
                with st.spinner("Filtering posts with zero comments..."):
                    filtered_df = self.filter_posts_with_comments()
                    st.success("Filtered data saved to outputs/engineered_data_filtered.csv.")
                    st.dataframe(filtered_df.head(20))
        if os.path.exists("outputs/engineered_data_filtered.csv"):
            st.subheader("Preview of Filtered Engineered Data")
            df = pd.read_csv("outputs/engineered_data_filtered.csv")
            st.dataframe(df.head(20))
