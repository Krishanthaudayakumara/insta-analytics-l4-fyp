"""
UI Components for Advanced ML Integration
Provides modular UI components that can be integrated into existing applications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLComponents:
    """Reusable UI components for advanced ML features"""
    
    @staticmethod
    def render_model_selector():
        """Render advanced model selection component"""
        st.sidebar.markdown("### 🧠 Advanced ML Models")
        
        advanced_features = st.sidebar.multiselect(
            "Select Advanced Features:",
            [
                "🚀 Boosting Models",
                "🎭 Multimodal Analysis", 
                "💬 NLP & LLM",
                "🕸️ Graph Neural Networks"
            ],
            default=["🚀 Boosting Models"]
        )
        
        return advanced_features
    
    @staticmethod
    def render_boosting_controls():
        """Render boosting model controls"""
        with st.expander("🚀 Boosting Model Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Model Type:",
                    ["ensemble", "xgboost", "lightgbm", "catboost"]
                )
                
                target_metric = st.selectbox(
                    "Target Metric:",
                    ["engagement_rate", "likes", "comments", "shares", "reach"]
                )
            
            with col2:
                use_optuna = st.checkbox("Use Optuna Optimization", value=True)
                cross_validation = st.selectbox("CV Folds:", [3, 5, 10], index=1)
            
            return {
                'model_type': model_type,
                'target_metric': target_metric,
                'use_optuna': use_optuna,
                'cv_folds': cross_validation
            }
    
    @staticmethod
    def render_multimodal_controls():
        """Render multimodal analysis controls"""
        with st.expander("🎭 Multimodal Analysis Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                vision_model = st.selectbox(
                    "Vision Model:",
                    ["ViT-Base", "ResNet-50", "EfficientNet-B4"]
                )
                
                text_model = st.selectbox(
                    "Text Model:",
                    ["BERT-Base", "RoBERTa-Base", "DistilBERT"]
                )
            
            with col2:
                fusion_method = st.selectbox(
                    "Fusion Method:",
                    ["attention", "concatenation", "bilinear"]
                )
                
                analyze_audio = st.checkbox("Include Audio Analysis", value=False)
            
            return {
                'vision_model': vision_model,
                'text_model': text_model,
                'fusion_method': fusion_method,
                'analyze_audio': analyze_audio
            }
    
    @staticmethod
    def render_nlp_controls():
        """Render NLP/LLM controls"""
        with st.expander("💬 NLP & LLM Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_llm = st.checkbox("Enable LLM Features", value=True)
                llm_provider = st.selectbox(
                    "LLM Provider:",
                    ["huggingface", "openai", "local"]
                )
            
            with col2:
                analyze_sentiment = st.checkbox("Sentiment Analysis", value=True)
                analyze_emotions = st.checkbox("Emotion Detection", value=True)
                generate_hashtags = st.checkbox("Hashtag Generation", value=False)
            
            return {
                'enable_llm': enable_llm,
                'llm_provider': llm_provider,
                'analyze_sentiment': analyze_sentiment,
                'analyze_emotions': analyze_emotions,
                'generate_hashtags': generate_hashtags
            }
    
    @staticmethod
    def render_gnn_controls():
        """Render GNN controls"""
        with st.expander("🕸️ Graph Neural Network Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                gnn_type = st.selectbox(
                    "GNN Architecture:",
                    ["GCN", "GAT", "GraphSAGE"]
                )
                
                num_layers = st.slider("Number of Layers:", 2, 6, 3)
            
            with col2:
                community_detection = st.checkbox("Community Detection", value=True)
                influencer_detection = st.checkbox("Influencer Detection", value=True)
            
            return {
                'gnn_type': gnn_type,
                'num_layers': num_layers,
                'community_detection': community_detection,
                'influencer_detection': influencer_detection
            }
    
    @staticmethod
    def render_performance_metrics(results_dict):
        """Render performance metrics in a standardized way"""
        if not results_dict or 'metrics' not in results_dict:
            return
        
        st.subheader("📊 Model Performance")
        
        metrics = results_dict['metrics']
        cols = st.columns(len(metrics))
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i]:
                if isinstance(value, float):
                    st.metric(
                        metric_name.replace('_', ' ').title(),
                        f"{value:.4f}"
                    )
                else:
                    st.metric(
                        metric_name.replace('_', ' ').title(),
                        str(value)
                    )
    
    @staticmethod
    def render_feature_importance(importance_data, title="Feature Importance"):
        """Render feature importance visualization"""
        if not importance_data:
            return
        
        st.subheader(f"🎯 {title}")
        
        if isinstance(importance_data, dict):
            # Convert dict to DataFrame
            df = pd.DataFrame(list(importance_data.items()), 
                            columns=['feature', 'importance'])
        else:
            df = importance_data
        
        # Sort by importance
        df = df.sort_values('importance', ascending=True).tail(15)
        
        fig = px.bar(
            df, 
            x='importance', 
            y='feature',
            orientation='h',
            title=title,
            color='importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_prediction_results(predictions, confidence_scores=None):
        """Render prediction results with confidence scores"""
        st.subheader("🎯 Predictions")
        
        if isinstance(predictions, (list, np.ndarray)):
            # Create DataFrame for multiple predictions
            df = pd.DataFrame({
                'Sample': range(len(predictions)),
                'Prediction': predictions
            })
            
            if confidence_scores is not None:
                df['Confidence'] = confidence_scores
                
                # Color-coded predictions based on confidence
                fig = px.scatter(
                    df, 
                    x='Sample', 
                    y='Prediction',
                    color='Confidence',
                    color_continuous_scale='RdYlGn',
                    title="Predictions with Confidence Scores"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Simple line plot for predictions
                fig = px.line(df, x='Sample', y='Prediction', title="Predictions")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Single prediction
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", f"{predictions:.4f}")
            if confidence_scores is not None:
                with col2:
                    st.metric("Confidence", f"{confidence_scores:.2%}")
    
    @staticmethod
    def render_model_comparison_chart(model_results):
        """Render model comparison visualization"""
        if not model_results:
            return
        
        st.subheader("📊 Model Comparison")
        
        # Prepare data for comparison
        models = list(model_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        comparison_data = []
        for model in models:
            for metric in metrics:
                if metric in model_results[model].get('metrics', {}):
                    comparison_data.append({
                        'Model': model,
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': model_results[model]['metrics'][metric]
                    })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                df, 
                x='Model', 
                y='Score',
                color='Metric',
                barmode='group',
                title="Model Performance Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_advanced_insights(insights_data):
        """Render advanced insights and recommendations"""
        if not insights_data:
            return
        
        st.subheader("💡 Advanced Insights")
        
        # Key insights
        if 'key_insights' in insights_data:
            st.markdown("#### 🔍 Key Insights")
            for insight in insights_data['key_insights']:
                st.info(f"• {insight}")
        
        # Recommendations
        if 'recommendations' in insights_data:
            st.markdown("#### 🎯 Recommendations")
            for rec in insights_data['recommendations']:
                st.success(f"✅ {rec}")
        
        # Warnings
        if 'warnings' in insights_data:
            st.markdown("#### ⚠️ Considerations")
            for warning in insights_data['warnings']:
                st.warning(f"⚠️ {warning}")
    
    @staticmethod
    def render_data_quality_check(data):
        """Render data quality assessment"""
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(data))
        
        with col2:
            missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
            st.metric("Missing Data", f"{missing_pct:.1f}%")
        
        with col3:
            duplicates = data.duplicated().sum()
            st.metric("Duplicates", duplicates)
        
        with col4:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outliers = 0
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers += ((data[col] < (Q1 - 1.5 * IQR)) | 
                           (data[col] > (Q3 + 1.5 * IQR))).sum()
            st.metric("Outliers", outliers)
        
        # Data types and basic statistics
        if st.checkbox("Show Detailed Data Info"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("**Data Types:**")
                st.dataframe(pd.DataFrame({
                    'Column': data.dtypes.index,
                    'Type': data.dtypes.values
                }))
            
            with col_b:
                st.markdown("**Basic Statistics:**")
                st.dataframe(data.describe())
    
    @staticmethod
    def render_progress_tracker(current_step, total_steps, step_names=None):
        """Render progress tracking for multi-step processes"""
        progress = current_step / total_steps
        st.progress(progress)
        
        if step_names and current_step <= len(step_names):
            st.info(f"Step {current_step}/{total_steps}: {step_names[current_step-1]}")
        else:
            st.info(f"Progress: {current_step}/{total_steps}")
    
    @staticmethod
    def render_error_handler(error_msg, suggestions=None):
        """Render standardized error messages with suggestions"""
        st.error(f"❌ Error: {error_msg}")
        
        if suggestions:
            st.markdown("**💡 Suggestions:**")
            for suggestion in suggestions:
                st.markdown(f"• {suggestion}")
    
    @staticmethod
    def render_export_options(data, filename_prefix="instagram_analysis"):
        """Render data export options"""
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to CSV"):
                csv = data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{filename_prefix}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export to Excel"):
                # Note: This would require openpyxl
                st.info("Excel export functionality requires openpyxl package")
        
        with col3:
            if st.button("📋 Copy to Clipboard"):
                st.code(data.to_string())
                st.info("Data displayed above - copy manually")

class AdvancedMLIntegration:
    """Main integration class for adding advanced ML to existing apps"""
    
    def __init__(self):
        self.components = AdvancedMLComponents()
    
    def add_advanced_sidebar(self):
        """Add advanced ML options to existing sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🧠 Advanced ML Features")
        
        enable_advanced = st.sidebar.checkbox(
            "Enable Advanced ML", 
            value=False,
            help="Enable advanced machine learning features"
        )
        
        if enable_advanced:
            return self.components.render_model_selector()
        
        return []
    
    def integrate_with_existing_analysis(self, data, selected_features):
        """Integrate advanced ML with existing analysis workflow"""
        results = {}
        
        for feature in selected_features:
            if "🚀 Boosting Models" in feature:
                st.markdown("### 🚀 Advanced Boosting Analysis")
                boosting_config = self.components.render_boosting_controls()
                
                if st.button("Run Boosting Analysis", key="boosting_btn"):
                    with st.spinner("Running advanced boosting models..."):
                        # Placeholder for actual boosting analysis
                        results['boosting'] = self._mock_boosting_results()
                        st.success("✅ Boosting analysis completed!")
            
            elif "🎭 Multimodal Analysis" in feature:
                st.markdown("### 🎭 Multimodal Analysis")
                multimodal_config = self.components.render_multimodal_controls()
                
                uploaded_files = st.file_uploader(
                    "Upload images for analysis:",
                    accept_multiple_files=True,
                    type=['png', 'jpg', 'jpeg']
                )
                
                if uploaded_files and st.button("Run Multimodal Analysis", key="multimodal_btn"):
                    with st.spinner("Analyzing multimodal content..."):
                        results['multimodal'] = self._mock_multimodal_results()
                        st.success("✅ Multimodal analysis completed!")
            
            elif "💬 NLP & LLM" in feature:
                st.markdown("### 💬 Advanced NLP Analysis")
                nlp_config = self.components.render_nlp_controls()
                
                text_input = st.text_area("Enter text for analysis:")
                
                if text_input and st.button("Run NLP Analysis", key="nlp_btn"):
                    with st.spinner("Performing advanced NLP analysis..."):
                        results['nlp'] = self._mock_nlp_results()
                        st.success("✅ NLP analysis completed!")
            
            elif "🕸️ Graph Neural Networks" in feature:
                st.markdown("### 🕸️ Graph Neural Network Analysis")
                gnn_config = self.components.render_gnn_controls()
                
                if st.button("Run GNN Analysis", key="gnn_btn"):
                    with st.spinner("Building graph and running GNN analysis..."):
                        results['gnn'] = self._mock_gnn_results()
                        st.success("✅ GNN analysis completed!")
        
        # Display results
        self._display_integrated_results(results)
        
        return results
    
    def _mock_boosting_results(self):
        """Mock boosting results for demonstration"""
        return {
            'metrics': {
                'accuracy': 0.891,
                'precision': 0.884,
                'recall': 0.876,
                'f1_score': 0.880
            },
            'feature_importance': {
                'follower_count': 0.245,
                'engagement_rate': 0.198,
                'hashtag_count': 0.156,
                'post_time': 0.134,
                'caption_length': 0.112,
                'content_type': 0.089,
                'sentiment_score': 0.066
            }
        }
    
    def _mock_multimodal_results(self):
        """Mock multimodal results for demonstration"""
        return {
            'aesthetic_score': 8.7,
            'engagement_prediction': 'High',
            'content_quality': 9.2,
            'mood_alignment': 0.89,
            'visual_appeal': 8.9
        }
    
    def _mock_nlp_results(self):
        """Mock NLP results for demonstration"""
        return {
            'sentiment_score': 0.78,
            'emotions': {
                'joy': 0.45,
                'excitement': 0.32,
                'trust': 0.28,
                'anticipation': 0.15
            },
            'engagement_prediction': 'High',
            'content_quality': 0.85,
            'readability': 'Good'
        }
    
    def _mock_gnn_results(self):
        """Mock GNN results for demonstration"""
        return {
            'communities_detected': 5,
            'influencers_identified': 12,
            'network_density': 0.23,
            'clustering_coefficient': 0.67,
            'top_influencers': [
                'user_1234', 'user_5678', 'user_9012'
            ]
        }
    
    def _display_integrated_results(self, results):
        """Display integrated analysis results"""
        if not results:
            return
        
        st.markdown("## 📈 Advanced Analysis Results")
        
        # Create tabs for different result types
        tab_names = list(results.keys())
        if tab_names:
            tabs = st.tabs([name.title() for name in tab_names])
            
            for i, (key, result_data) in enumerate(results.items()):
                with tabs[i]:
                    if key == 'boosting':
                        self.components.render_performance_metrics(result_data)
                        self.components.render_feature_importance(
                            result_data.get('feature_importance', {})
                        )
                    
                    elif key == 'multimodal':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Aesthetic Score", f"{result_data.get('aesthetic_score', 0)}/10")
                            st.metric("Content Quality", f"{result_data.get('content_quality', 0)}/10")
                        with col2:
                            st.metric("Engagement Prediction", result_data.get('engagement_prediction', 'N/A'))
                            st.metric("Visual Appeal", f"{result_data.get('visual_appeal', 0)}/10")
                    
                    elif key == 'nlp':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Sentiment Score", f"{result_data.get('sentiment_score', 0):.2f}")
                            st.metric("Content Quality", f"{result_data.get('content_quality', 0):.2f}")
                        with col2:
                            st.metric("Engagement Prediction", result_data.get('engagement_prediction', 'N/A'))
                            st.metric("Readability", result_data.get('readability', 'N/A'))
                        
                        # Emotion chart
                        if 'emotions' in result_data:
                            emotions_df = pd.DataFrame(
                                list(result_data['emotions'].items()),
                                columns=['Emotion', 'Score']
                            )
                            fig = px.bar(emotions_df, x='Emotion', y='Score',
                                       title="Emotion Analysis")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif key == 'gnn':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Communities", result_data.get('communities_detected', 0))
                        with col2:
                            st.metric("Influencers", result_data.get('influencers_identified', 0))
                        with col3:
                            st.metric("Network Density", f"{result_data.get('network_density', 0):.2f}")
                        
                        if 'top_influencers' in result_data:
                            st.subheader("🌟 Top Influencers")
                            for influencer in result_data['top_influencers']:
                                st.write(f"• {influencer}")

# Usage example function
def integrate_advanced_ml_into_existing_app():
    """
    Example of how to integrate advanced ML into an existing Streamlit app
    Add this to your main app file
    """
    # Initialize integration
    advanced_ml = AdvancedMLIntegration()
    
    # Add to sidebar
    selected_features = advanced_ml.add_advanced_sidebar()
    
    # If advanced features are selected, integrate them
    if selected_features:
        # Assuming you have data loaded
        # data = load_your_data()  # Your existing data loading
        
        # Mock data for example
        data = pd.DataFrame({
            'engagement_rate': np.random.uniform(0.01, 0.15, 100),
            'likes': np.random.exponential(100, 100),
            'comments': np.random.exponential(20, 100)
        })
        
        # Integrate advanced analysis
        advanced_results = advanced_ml.integrate_with_existing_analysis(
            data, selected_features
        )
        
        return advanced_results
    
    return None
