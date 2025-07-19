"""
Dataset-Wide Analysis UI Component
Provides comprehensive analysis and visualization of high-value followers across the entire dataset
"""

import streamlit as st
import pandas as pd
import json
import os
from .base import BaseUIComponent


class DatasetAnalysisComponent(BaseUIComponent):
    """Dataset-wide high-value follower analysis component"""
    
    def show(self):
        """Show dataset-wide analysis interface"""
        st.markdown("### 🌐 Dataset-Wide High-Value Follower Analysis")
        st.markdown("*Analyze patterns, networks, and insights across all Instagram accounts in the dataset*")
        
        # Check if preprocessed data exists
        if not os.path.exists("outputs/preprocessed_data.csv"):
            st.warning("⚠️ Please preprocess data first!")
            return
        
        # Load data
        try:
            df = pd.read_csv("outputs/preprocessed_data.csv")
            st.info(f"📊 Working with {len(df)} records across the entire dataset")
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
        
        # Validate required columns
        required_cols = ['owner_id', 'username', 'comment_owner_username']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Dataset missing required columns: {missing_cols}")
            return
        
        # Import analyzer and visualizer with robust path handling
        analyzer = None
        visualizer = None
        use_simple_mode = False
        
        # Get the absolute path to the source directory
        import sys
        
        # Multiple strategies to find the correct path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up from ui -> src -> project_root
        src_dir = os.path.join(project_root, 'src')
        
        # Ensure src directory is in Python path
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            # Strategy 1: Direct import after path setup
            from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
            from follower_selection.network_visualizer import FollowerNetworkVisualizer
            
            # Initialize components
            analyzer = DatasetFollowerAnalyzer()
            visualizer = FollowerNetworkVisualizer()
            st.success("✅ Full analysis modules loaded successfully")
            
        except ImportError as e:
            # Strategy 2: Import from current app context
            try:
                # Try to import from the app's context
                import importlib.util
                
                # Load dataset analyzer
                analyzer_path = os.path.join(src_dir, 'follower_selection', 'dataset_analyzer.py')
                spec = importlib.util.spec_from_file_location("dataset_analyzer", analyzer_path)
                analyzer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(analyzer_module)
                
                # Load network visualizer
                visualizer_path = os.path.join(src_dir, 'follower_selection', 'network_visualizer.py')
                spec = importlib.util.spec_from_file_location("network_visualizer", visualizer_path)
                visualizer_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(visualizer_module)
                
                # Initialize components
                analyzer = analyzer_module.DatasetFollowerAnalyzer()
                visualizer = visualizer_module.FollowerNetworkVisualizer()
                st.success("✅ Analysis modules loaded via file import")
                
            except Exception as e2:
                st.warning(f"⚠️ Full analysis modules unavailable: {str(e)}")
                st.info("🔄 Switching to simplified analysis mode...")
                
                # Strategy 3: Fallback to simple analysis
                try:
                    from .simple_dataset_analyzer import SimpleDatasetAnalyzer
                    analyzer = SimpleDatasetAnalyzer()
                    visualizer = None  # No visualizer in simple mode
                    use_simple_mode = True
                    st.success("✅ Simple analysis mode activated")
                    
                except Exception as e3:
                    st.error(f"❌ Could not load any analysis modules")
                    
                    with st.expander("🔍 Debug Information"):
                        st.write("**Import error:**", str(e))
                        st.write("**File import error:**", str(e2))
                        st.write("**Simple mode error:**", str(e3))
                        st.write("**Current directory:**", current_dir)
                        st.write("**Project root:**", project_root)
                        st.write("**Source directory:**", src_dir)
                        st.write("**Python path:**", sys.path[:3])
                        
                        # Check if files exist
                        analyzer_file = os.path.join(src_dir, 'follower_selection', 'dataset_analyzer.py')
                        visualizer_file = os.path.join(src_dir, 'follower_selection', 'network_visualizer.py')
                        
                        st.write("**dataset_analyzer.py exists:**", os.path.exists(analyzer_file))
                        st.write("**network_visualizer.py exists:**", os.path.exists(visualizer_file))
                        st.write("**follower_selection dir exists:**", os.path.exists(os.path.join(src_dir, 'follower_selection')))
                    
                    st.info("💡 **Alternative**: Use the individual account analysis in '👑 Select High-Value Followers'")
                    return
        
        # Analysis parameters
        st.markdown("#### 🎛️ Analysis Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            top_percentage = st.slider(
                "Top Followers Percentage per Account", 
                5, 25, 10,
                help="Percentage of top followers to select for each account"
            )
            clustering_method = st.selectbox(
                "Clustering Method:",
                ["K-Means", "DBSCAN", "Hierarchical"],
                help="Algorithm used for clustering followers"
            )
            min_followers = st.number_input(
                "Minimum Followers per Account",
                min_value=1,
                max_value=50,
                value=5,
                help="Minimum number of followers required for account analysis"
            )
        
        with col2:
            engagement_weight = st.slider(
                "Engagement Weight", 
                0.0, 1.0, 0.7,
                help="Weight for engagement metrics in scoring"
            )
            influence_weight = st.slider(
                "Influence Weight", 
                0.0, 1.0, 0.3,
                help="Weight for influence metrics in scoring"
            )
            
            if abs(engagement_weight + influence_weight - 1.0) > 0.01:
                st.warning("⚠️ Weights should sum to 1.0 for optimal results")
        
        # Run analysis button
        if st.button("🚀 Analyze Entire Dataset", type="primary"):
            if analyzer is None:
                st.error("❌ No analysis modules available. Please check the installation.")
                return
                
            self._run_dataset_analysis(
                df, analyzer, visualizer, top_percentage, clustering_method,
                engagement_weight, influence_weight, min_followers, use_simple_mode
            )
        
        # Show existing results if available
        self._show_existing_analysis()
    
    def _run_dataset_analysis(self, df, analyzer, visualizer, top_percentage, 
                            clustering_method, engagement_weight, influence_weight, min_followers, use_simple_mode=False):
        """Run the complete dataset analysis"""
        
        with st.spinner("🔍 Analyzing entire dataset... This may take a few minutes..."):
            try:
                # Run analysis
                analysis_results = analyzer.analyze_entire_dataset(
                    data=df,
                    top_percentage=top_percentage,
                    clustering_method=clustering_method,
                    engagement_weight=engagement_weight,
                    influence_weight=influence_weight,
                    min_followers_per_account=min_followers
                )
                
                st.success("✅ Dataset analysis completed successfully!")
                
                # Show analysis results
                if use_simple_mode:
                    self._display_simple_analysis_results(analysis_results)
                else:
                    self._display_analysis_results(analysis_results, visualizer)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Try adjusting parameters or check if the dataset has sufficient data.")
    
    def _display_analysis_results(self, analysis_results, visualizer):
        """Display comprehensive analysis results"""
        
        # Summary metrics
        st.markdown("#### 📊 Analysis Summary")
        
        metadata = analysis_results['metadata']
        insights = analysis_results['insights']
        cross_analysis = analysis_results['cross_account_analysis']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Accounts Analyzed",
                metadata['total_accounts_analyzed'],
                f"of {metadata['total_accounts_in_dataset']} total"
            )
        
        with col2:
            st.metric(
                "Total High-Value Followers",
                insights['summary_statistics']['total_high_value_followers']
            )
        
        with col3:
            st.metric(
                "Power Followers",
                len(cross_analysis['multi_account_followers']),
                "Multi-account valuable"
            )
        
        with col4:
            cross_pollination = cross_analysis['network_stats']['cross_pollination_rate']
            st.metric(
                "Cross-Pollination Rate",
                f"{cross_pollination:.1%}",
                "Shared high-value followers"
            )
        
        # Insights and recommendations
        st.markdown("#### 💡 Key Insights & Recommendations")
        
        recommendations = insights['recommendations']
        if recommendations:
            for i, rec in enumerate(recommendations):
                with st.expander(f"{rec['title']} ({rec['type'].replace('_', ' ').title()})"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write(f"**Recommended Action:** {rec['action']}")
        else:
            st.info("No specific recommendations generated for this dataset.")
        
        # Network visualization
        st.markdown("#### 🕸️ Network Visualization")
        
        try:
            # Network graph options
            col1, col2 = st.columns(2)
            with col1:
                layout = st.selectbox(
                    "Network Layout:",
                    ["spring", "circular", "kamada_kawai"],
                    help="Choose how nodes are arranged in the network"
                )
            with col2:
                node_size = st.slider(
                    "Node Size Factor",
                    0.5, 2.0, 1.0,
                    help="Adjust the size of nodes in the network"
                )
            
            # Create and display network graph
            network_fig = visualizer.create_network_graph(
                analysis_results, 
                layout=layout, 
                node_size_factor=node_size
            )
            st.plotly_chart(network_fig, use_container_width=True)
            
            # Network insights
            with st.expander("🔍 Network Analysis Details"):
                st.write(f"**Total Nodes:** {len(analysis_results['account_results']) + len(analysis_results['high_value_followers_global'])}")
                st.write(f"**Account Nodes:** {len(analysis_results['account_results'])}")
                st.write(f"**Follower Nodes:** {len(analysis_results['high_value_followers_global'])}")
                st.write(f"**Power Followers:** {len(cross_analysis['multi_account_followers'])}")
                
        except Exception as e:
            st.error(f"❌ Error creating network visualization: {str(e)}")
        
        # Additional visualizations
        st.markdown("#### 📈 Additional Analysis")
        
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["🔥 Similarity Heatmap", "📊 Insights Dashboard", "🏆 Top Performers"])
        
        with tab1:
            try:
                similarity_fig = visualizer.create_account_similarity_heatmap(analysis_results)
                st.plotly_chart(similarity_fig, use_container_width=True)
                
                st.markdown("**Interpretation:**")
                st.write("- Darker colors indicate higher similarity between accounts")
                st.write("- Similar accounts share many high-value followers")
                st.write("- Identify collaboration opportunities and account clusters")
                
            except Exception as e:
                st.error(f"Error creating similarity heatmap: {str(e)}")
        
        with tab2:
            try:
                dashboard_fig = visualizer.create_insights_dashboard(analysis_results)
                st.plotly_chart(dashboard_fig, use_container_width=True)
                
                st.markdown("**Dashboard Insights:**")
                st.write("- **Top Left:** Distribution of high-value followers across accounts")
                st.write("- **Top Right:** Power followers who are valuable to multiple accounts")
                st.write("- **Bottom Left:** Statistical distribution of follower counts")
                st.write("- **Bottom Right:** Engagement vs Influence scatter plot")
                
            except Exception as e:
                st.error(f"Error creating insights dashboard: {str(e)}")
        
        with tab3:
            try:
                performers_fig = visualizer.create_top_performers_chart(analysis_results)
                st.plotly_chart(performers_fig, use_container_width=True)
                
                # Show detailed top performers data
                top_performers = cross_analysis['top_performers']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏅 Top Accounts by High-Value Followers:**")
                    for i, (account, count) in enumerate(top_performers['top_accounts_by_count'][:5]):
                        st.write(f"{i+1}. @{account}: {count} followers")
                
                with col2:
                    st.markdown("**⭐ Top Followers by Average Score:**")
                    for i, (follower, score) in enumerate(top_performers['top_followers_by_score'][:5]):
                        st.write(f"{i+1}. @{follower}: {score:.3f}")
                
            except Exception as e:
                st.error(f"Error creating top performers chart: {str(e)}")
        
        # Download options
        st.markdown("#### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download analysis results as JSON
            analysis_json = json.dumps(analysis_results, indent=2, default=str)
            st.download_button(
                label="📄 Download Analysis Results (JSON)",
                data=analysis_json,
                file_name=f"dataset_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Save visualizations button
            if st.button("💾 Save All Visualizations"):
                try:
                    saved_files = visualizer.save_visualizations(analysis_results)
                    st.success(f"✅ Saved {len(saved_files)} visualization files to outputs/visualizations/")
                    for file_path in saved_files:
                        st.write(f"- {os.path.basename(file_path)}")
                except Exception as e:
                    st.error(f"❌ Error saving visualizations: {str(e)}")
    
    def _display_simple_analysis_results(self, analysis_results):
        """Display simplified analysis results when full modules aren't available"""
        
        st.markdown("#### 📊 Analysis Summary (Simple Mode)")
        
        metadata = analysis_results['metadata']
        insights = analysis_results['insights']
        cross_analysis = analysis_results['cross_account_analysis']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Accounts Analyzed",
                metadata['total_accounts_analyzed'],
                f"of {metadata['total_accounts_in_dataset']} total"
            )
        
        with col2:
            st.metric(
                "Total High-Value Followers",
                insights['summary_statistics']['total_high_value_followers']
            )
        
        with col3:
            st.metric(
                "Power Followers",
                len(cross_analysis['multi_account_followers']),
                "Multi-account valuable"
            )
        
        with col4:
            cross_pollination = cross_analysis['network_stats']['cross_pollination_rate']
            st.metric(
                "Cross-Pollination Rate",
                f"{cross_pollination:.1%}",
                "Shared high-value followers"
            )
        
        # Insights and recommendations
        st.markdown("#### 💡 Key Insights & Recommendations")
        
        recommendations = insights['recommendations']
        if recommendations:
            for i, rec in enumerate(recommendations):
                with st.expander(f"{rec['title']} ({rec['type'].replace('_', ' ').title()})"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write(f"**Recommended Action:** {rec['action']}")
        else:
            st.info("No specific recommendations generated for this dataset.")
        
        # Simple data tables instead of visualizations
        st.markdown("#### 📈 Analysis Details")
        
        tab1, tab2 = st.tabs(["🏆 Top Performers", "📊 Account Summary"])
        
        with tab1:
            top_performers = cross_analysis['top_performers']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🏅 Top Accounts by High-Value Followers:**")
                for i, (account, count) in enumerate(top_performers['top_accounts_by_count'][:10]):
                    st.write(f"{i+1}. @{account}: {count} followers")
            
            with col2:
                st.markdown("**⭐ Top Followers by Average Score:**")
                for i, (follower, score) in enumerate(top_performers['top_followers_by_score'][:10]):
                    st.write(f"{i+1}. @{follower}: {score:.3f}")
        
        with tab2:
            st.markdown("**📊 Account Analysis Summary:**")
            
            # Create a simple dataframe for account results
            account_data = []
            for username, data in analysis_results['account_results'].items():
                account_data.append({
                    'Username': f"@{username}",
                    'High-Value Followers': data['high_value_count'],
                    'Total Followers': data['followers_count'],
                    'Selection Rate': f"{(data['high_value_count'] / data['followers_count'] * 100):.1f}%" if data['followers_count'] > 0 else "0%"
                })
            
            if account_data:
                df_accounts = pd.DataFrame(account_data)
                st.dataframe(df_accounts, use_container_width=True)
            else:
                st.info("No account data available to display.")
        
        # Download option
        st.markdown("#### 📥 Download Results")
        
        analysis_json = json.dumps(analysis_results, indent=2, default=str)
        st.download_button(
            label="📄 Download Analysis Results (JSON)",
            data=analysis_json,
            file_name=f"simple_dataset_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def _show_existing_analysis(self):
        """Show existing analysis results if available"""
        
        analysis_file = "outputs/dataset_follower_analysis.json"
        
        if os.path.exists(analysis_file):
            st.markdown("#### 📋 Previous Analysis Results")
            
            try:
                with open(analysis_file, 'r') as f:
                    previous_results = json.load(f)
                
                metadata = previous_results.get('metadata', {})
                timestamp = metadata.get('analysis_timestamp', 'Unknown')
                accounts_analyzed = metadata.get('total_accounts_analyzed', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Previous Analysis", f"{accounts_analyzed} accounts")
                
                with col2:
                    st.metric("Analysis Date", timestamp.split('T')[0] if 'T' in str(timestamp) else str(timestamp))
                
                with col3:
                    if st.button("🔄 Load Previous Results"):
                        # Import visualizer
                        try:
                            from ..follower_selection.network_visualizer import FollowerNetworkVisualizer
                            visualizer = FollowerNetworkVisualizer()
                            self._display_analysis_results(previous_results, visualizer)
                        except Exception as e:
                            # Fallback to simple display
                            st.warning("⚠️ Full visualization unavailable, showing simplified results")
                            self._display_simple_analysis_results(previous_results)
                            
            except Exception as e:
                st.warning(f"⚠️ Error loading previous analysis: {str(e)}")
        else:
            st.info("ℹ️ No previous dataset analysis found. Run a new analysis to see results.")