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
        """Show high-value follower selection with tabs for different analysis types"""
        st.markdown("### 👑 High-Value Follower Selection & Analysis")
        
        # Create tabs for different analysis types
        tab1, tab2 = st.tabs(["🎯 Account-Specific Analysis", "🌐 Dataset-Wide Analysis"])
        
        with tab1:
            self._show_account_specific_analysis()
        
        with tab2:
            self._show_dataset_wide_analysis()
    
    def _show_account_specific_analysis(self):
        """Show the original account-specific follower selection"""
        st.markdown("#### 🎯 Account-Specific High-Value Follower Selection")
        st.markdown("*Select high-value followers for a specific Instagram account*")
        
        # Check if preprocessed data exists
        if not os.path.exists("outputs/preprocessed_data.csv"):
            st.warning("⚠️ Please preprocess data first!")
            return
        
        # Load data
        try:
            df = pd.read_csv("outputs/preprocessed_data.csv")
            st.info(f"📊 Working with {len(df)} records")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
        
        # Account selection
        if 'owner_id' not in df.columns:
            st.error("❌ Dataset missing 'owner_id' column. Please check data preprocessing.")
            st.info("💡 The dataset needs an 'owner_id' column to identify different Instagram accounts.")
            return
        
        # Get available accounts
        try:
            account_stats = self.follower_selector.get_available_accounts(df)
            st.success(f"✅ Found {len(account_stats)} Instagram accounts in dataset")
            
            # Show account statistics
            with st.expander("📊 Account Statistics"):
                for account in account_stats:
                    st.write(f"**@{account['username']}** (ID: {account['owner_id']}): {account['unique_followers']} followers, {account['total_interactions']} interactions")
            
        except Exception as e:
            st.error(f"❌ Error analyzing accounts: {str(e)}")
            return
        
        # Account selection dropdown - Show username as primary, ID as secondary
        account_options = [f"@{acc['username']} (ID: {acc['owner_id']})" for acc in account_stats]
        selected_account_display = st.selectbox(
            "📱 Select Instagram Account:",
            account_options,
            help="Choose which account's followers to analyze"
        )
        
        # Extract the selected owner_id from the display string
        selected_owner = None
        selected_username = None
        for acc in account_stats:
            if f"@{acc['username']} (ID: {acc['owner_id']})" == selected_account_display:
                selected_owner = str(acc['owner_id'])
                selected_username = acc['username']
                break
        
        # Show selected account info with username prominence
        if selected_owner and selected_username:
            selected_stats = next((acc for acc in account_stats if str(acc['owner_id']) == selected_owner), None)
            if selected_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Instagram Account", f"@{selected_username}")
                with col2:
                    st.metric("Unique Followers", selected_stats['unique_followers'])
                with col3:
                    st.metric("Total Interactions", selected_stats['total_interactions'])
        
        # Selection parameters
        st.markdown("#### 🎛️ Selection Parameters")
        col1, col2 = st.columns(2)
        with col1:
            top_percent = st.slider("Top Followers Percentage", 5, 25, 10, 
                                   help="Select top X% of followers based on engagement scores")
            clustering_method = st.selectbox(
                "Clustering Method:",
                ["K-Means", "DBSCAN", "Hierarchical"],
                help="Choose clustering algorithm to group similar followers"
            )
        
        with col2:
            engagement_weight = st.slider("Engagement Weight", 0.0, 1.0, 0.7,
                                        help="Weight for engagement metrics (likes, comments, interactions)")
            influence_weight = st.slider("Influence Weight", 0.0, 1.0, 0.3,
                                       help="Weight for influence metrics (follower count)")
        
        # Validation
        if abs(engagement_weight + influence_weight - 1.0) > 0.01:
            st.warning("⚠️ Engagement Weight + Influence Weight should equal 1.0 for optimal results")
        
        # Selection button
        if st.button("🎯 Select High-Value Followers", type="primary"):
            with st.spinner(f"Identifying high-value followers for account {selected_owner}..."):
                try:
                    high_value_followers = self.follower_selector.select_high_value_followers(
                        df,
                        owner_id=selected_owner,
                        top_percentage=top_percent,
                        clustering_method=clustering_method,
                        engagement_weight=engagement_weight,
                        influence_weight=influence_weight
                    )
                    
                    st.success(f"✅ Selected {len(high_value_followers)} high-value followers for @{selected_username}!")
                    
                    # Show results
                    self._show_selection_results(high_value_followers, df, selected_owner, selected_username)
                    
                except Exception as e:
                    st.error(f"❌ Error during selection: {str(e)}")
                    st.info("💡 Try adjusting parameters or check if the account has sufficient follower data.")
        
        # Show existing results if available
        self._show_existing_results(selected_owner, selected_username)
    
    def _show_selection_results(self, high_value_followers, df, owner_id, username):
        """Show follower selection results"""
        st.markdown("#### 📊 Selection Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High-Value Followers", len(high_value_followers))
        with col2:
            account_data = df[df['owner_id'] == int(owner_id)]
            unique_followers = account_data['comment_owner_username'].nunique()
            selection_rate = (len(high_value_followers) / unique_followers * 100) if unique_followers > 0 else 0
            st.metric("Selection Rate", f"{selection_rate:.1f}%")
        with col3:
            st.metric("Instagram Account", f"@{username}")
        
        # Show sample followers
        if high_value_followers:
            st.markdown("#### 👥 Selected Followers Sample")
            sample_size = min(10, len(high_value_followers))
            sample_followers = list(high_value_followers.keys())[:sample_size]
            
            # Create a nice display table
            sample_data = []
            for username_follower in sample_followers:
                follower_data = high_value_followers[username_follower]
                sample_data.append({
                    'Username': username_follower,
                    'Total Score': f"{follower_data['total_score']:.3f}",
                    'Engagement Score': f"{follower_data['engagement_score']:.3f}",
                    'Influence Score': f"{follower_data['influence_score']:.3f}",
                    'Followers': f"{follower_data['followers']:,}",
                    'Cluster': follower_data['cluster']
                })
            
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
            
            # Download option
            results_json = json.dumps(high_value_followers, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=results_json,
                file_name=f"high_value_followers_{username}_{owner_id}.json",
                mime="application/json"
            )

    def _show_existing_results(self, selected_owner, selected_username):
        """Show existing results for the selected account"""
        # Check for account-specific results
        account_results_file = f"outputs/high_value_followers_{selected_owner}.json"
        general_results_file = "outputs/high_value_followers.json"
        
        if os.path.exists(account_results_file):
            try:
                with open(account_results_file, "r") as f:
                    existing_results = json.load(f)
                
                if "high_value_followers" in existing_results:
                    followers = existing_results["high_value_followers"]
                    metadata = existing_results.get("selection_metadata", {})
                    
                    st.markdown("#### 📋 Previous Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Previously Selected", len(followers))
                    with col2:
                        st.metric("Top Percentage", f"{metadata.get('top_percentage', 'N/A')}%")
                    with col3:
                        st.metric("Method", metadata.get('clustering_method', 'N/A'))
                    
                    st.info(f"📊 Previous results for **@{selected_username}** (ID: {selected_owner})")
                    
                    if st.button("🔄 Load Previous Results"):
                        self._show_selection_results(followers, pd.read_csv("outputs/preprocessed_data.csv"), selected_owner, selected_username)
                        
            except Exception as e:
                st.warning(f"⚠️ Error loading previous results: {str(e)}")
        
        elif os.path.exists(general_results_file):
            try:
                with open(general_results_file, "r") as f:
                    existing_followers = json.load(f)
                st.info(f"📊 General results available: {len(existing_followers)} followers (not account-specific)")
            except Exception as e:
                st.warning(f"⚠️ Error loading general results: {str(e)}")
        else:
            st.info(f"ℹ️ No previous results found for @{selected_username}")
    
    def _show_dataset_wide_analysis(self):
        """Show dataset-wide analysis using the new component"""
        try:
            from .dataset_analysis import DatasetAnalysisComponent
            
            # Create and show the dataset analysis component
            dataset_component = DatasetAnalysisComponent(self.app)
            dataset_component.show()
            
        except ImportError:
            st.error("❌ Dataset analysis module not available. Please check the installation.")
        except Exception as e:
            st.error(f"❌ Error loading dataset analysis: {str(e)}")
