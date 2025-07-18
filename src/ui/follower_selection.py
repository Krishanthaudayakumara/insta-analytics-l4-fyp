"""
Follower Selection UI Component
Handles high-value follower selection interface
"""

import streamlit as st
import pandas as pd
import json
import os
from .base import BaseUIComponent


class FollowerSelectionComponent(BaseUIComponent):
    """High-value follower selection component"""
    
    def show(self):
        """Show high-value follower selection"""
        st.markdown("### 👑 High-Value Follower Selection")
        
        # Check if preprocessed data exists
        if not os.path.exists("outputs/preprocessed_data.csv"):
            st.warning("⚠️ Please preprocess data first!")
            return
        
        # Load data
        df = pd.read_csv("outputs/preprocessed_data.csv")
        st.info(f"📊 Working with {len(df)} records")
        
        # Selection parameters
        col1, col2 = st.columns(2)
        with col1:
            top_percent = st.slider("Top Followers Percentage", 5, 25, 10)
            clustering_method = st.selectbox(
                "Clustering Method:",
                ["K-Means", "DBSCAN", "Hierarchical"]
            )
        
        with col2:
            engagement_weight = st.slider("Engagement Weight", 0.0, 1.0, 0.7)
            influence_weight = st.slider("Influence Weight", 0.0, 1.0, 0.3)
        
        # Selection button
        if st.button("🎯 Select High-Value Followers", type="primary"):
            with st.spinner("Identifying high-value followers..."):
                try:
                    high_value_followers = self.follower_selector.select_followers(
                        df,
                        top_percent=top_percent,
                        method=clustering_method,
                        engagement_weight=engagement_weight,
                        influence_weight=influence_weight
                    )
                    
                    # Save results
                    with open("outputs/high_value_followers.json", "w") as f:
                        json.dump(high_value_followers, f)
                    
                    st.success(f"✅ Selected {len(high_value_followers)} high-value followers!")
                    
                    # Show results
                    self._show_selection_results(high_value_followers, df)
                    
                except Exception as e:
                    st.error(f"❌ Error during selection: {str(e)}")
        
        # Show existing results if available
        if os.path.exists("outputs/high_value_followers.json"):
            with open("outputs/high_value_followers.json", "r") as f:
                existing_followers = json.load(f)
            
            st.info(f"📊 Previously selected: {len(existing_followers)} followers")
    
    def _show_selection_results(self, high_value_followers, df):
        """Show follower selection results"""
        col1, col2 = st.columns(2)
        with col1:
            st.metric("High-Value Followers", len(high_value_followers))
            st.metric("Selection Rate", f"{len(high_value_followers)/len(df)*100:.1f}%")
        
        with col2:
            if high_value_followers:
                sample_followers = list(high_value_followers.keys())[:5]
                st.write("**Sample Followers:**")
                for follower in sample_followers:
                    st.write(f"- {follower}")
