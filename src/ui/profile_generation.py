"""
Profile Generation UI Component
Handles user profile generation interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from .base import BaseUIComponent


class ProfileGenerationComponent(BaseUIComponent):
    """Profile generation component"""
    
    def show(self):
        """Profile generation interface"""
        st.markdown("### 👤 Generate User Profiles")
        
        # Check prerequisites
        prerequisites = [
            ("High-Value Followers", "outputs/high_value_followers.json"),
            ("Sentiment Scores", "outputs/sentiment_scores.json"),
            ("Model Metrics", "outputs/metrics.json")
        ]
        
        if self.show_prerequisites_warning(prerequisites):
            return
        
        # Profile generation options
        col1, col2 = st.columns(2)
        with col1:
            include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
            include_content_prefs = st.checkbox("Include Content Preferences", value=True)
        
        with col2:
            min_confidence = st.slider("Minimum Prediction Confidence", 0.5, 0.95, 0.7)
            max_profiles = st.slider("Maximum Profiles to Generate", 10, 100, 50)
        
        # Generate button
        if st.button("👤 Generate Profiles", type="primary"):
            with st.spinner("Generating user profiles..."):
                try:
                    profiles = self.profile_generator.generate_profiles(
                        include_sentiment=include_sentiment,
                        include_content_prefs=include_content_prefs,
                        min_confidence=min_confidence,
                        max_profiles=max_profiles
                    )
                    
                    # Save results
                    with open("outputs/profiles.json", "w") as f:
                        json.dump(profiles, f)
                    
                    # Generate guidelines
                    guidelines = self.profile_generator.generate_guidelines(profiles)
                    with open("outputs/guidelines.json", "w") as f:
                        json.dump(guidelines, f)
                    
                    st.success(f"✅ Generated {len(profiles)} user profiles!")
                    
                    # Show profile results
                    self._show_profile_results(profiles, guidelines, "new")
                    
                except Exception as e:
                    st.error(f"❌ Error during profile generation: {str(e)}")
        
        # Show existing results if available
        if os.path.exists("outputs/profiles.json"):
            with open("outputs/profiles.json", "r") as f:
                existing_profiles = json.load(f)
            with open("outputs/guidelines.json", "r") as f:
                existing_guidelines = json.load(f)
            
            st.info(f"📊 Previously generated: {len(existing_profiles)} profiles")
            self._show_profile_results(existing_profiles, existing_guidelines, "previous")
    
    def _show_profile_results(self, profiles, guidelines, context="default"):
        """Display profile generation results"""
        if not profiles:
            return
        
        # Create unique key based on context
        unique_key = f"content_preference_distribution_{context}"
        
        # Profile summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Profiles", len(profiles))
            
            # Sample profile
            if profiles:
                sample_user = list(profiles.keys())[0]
                sample_profile = profiles[sample_user]
                
                st.markdown("**Sample Profile:**")
                st.json(sample_profile)
        
        with col2:
            # Content preferences distribution
            if profiles:
                content_prefs = []
                for profile in profiles.values():
                    if 'content_preferences' in profile:
                        content_prefs.extend(profile['content_preferences'])
                
                if content_prefs:
                    pref_counts = pd.Series(content_prefs).value_counts()
                    fig = px.pie(
                        values=pref_counts.values,
                        names=pref_counts.index,
                        title="Content Preference Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=unique_key)
        
        # Guidelines
        if guidelines:
            st.markdown("### 📋 Content Strategy Guidelines")
            for i, guideline in enumerate(guidelines[:5], 1):
                st.markdown(f"**{i}.** {guideline}")
