"""
Instagram User Behavior Analysis: ML-Driven Personalized Engagement Modeling
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import json
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import modules
from preprocessing.data_processor import DataProcessor
from preprocessing.clustered_data_processor import ClusteredDataProcessor
from follower_selection.high_value_selector import HighValueFollowerSelector
from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
from models.model_trainer import ModelTrainer
from evaluation.model_evaluator import ModelEvaluator
from profiling.profile_generator import ProfileGenerator
from ui.live_prediction import LivePredictionComponent

# Configure page
st.set_page_config(
    page_title="Instagram Engagement Modeling",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E4405F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #405DE6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class InstagramEngagementApp:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.clustered_data_processor = ClusteredDataProcessor()
        self.follower_selector = HighValueFollowerSelector()
        self.sentiment_analyzer = BERTSentimentAnalyzer()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.profile_generator = ProfileGenerator()
        # Initialize live prediction component with app instance
        self.live_prediction = LivePredictionComponent(self)
        
    def main(self):
        """Main application interface"""
        st.markdown('<h1 class="main-header">📸 Instagram User Behavior Analysis</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">ML-Driven Personalized Engagement Modeling for High-Value Followers</h2>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("🚀 Navigation")
        page = st.sidebar.selectbox(
            "Choose Pipeline:",
            ["🏠 Overview", "📊 Preprocess Data", "👑 Select High-Value Followers", 
             "🧠 Sentiment Analysis", "🤖 Train Models", "📈 Evaluate Models", 
             "👤 Generate Profiles", "📋 Visualize Results", "🔮 Live Predictions"]
        )
        
        # Pipeline status in sidebar
        self.show_pipeline_status()
        
        # Route to appropriate page
        if page == "🏠 Overview":
            self.show_overview()
        elif page == "📊 Preprocess Data":
            self.show_preprocessing()
        elif page == "👑 Select High-Value Followers":
            self.show_follower_selection()
        elif page == "🧠 Sentiment Analysis":
            self.show_sentiment_analysis()
        elif page == "🤖 Train Models":
            self.show_model_training()
        elif page == "📈 Evaluate Models":
            self.show_model_evaluation()
        elif page == "👤 Generate Profiles":
            self.show_profile_generation()
        elif page == "📋 Visualize Results":
            self.show_visualization()
        elif page == "🔮 Live Predictions":
            self.show_live_predictions()
    
    def show_pipeline_status(self):
        """Show pipeline execution status"""
        st.sidebar.markdown("### 📋 Pipeline Status")
        
        # Check if files exist to determine status
        from src.utils.high_value_utils import check_high_value_data_exists
        
        statuses = {
            "Data Preprocessed": os.path.exists("outputs/preprocessed_data.csv"),
            "High-Value Followers": check_high_value_data_exists(),
            "Sentiment Analysis": os.path.exists("outputs/sentiment_scores.json"),
            "Models Trained": os.path.exists("outputs/rf_model.pkl"),
            "Models Evaluated": os.path.exists("outputs/metrics.json"),
            "Profiles Generated": os.path.exists("outputs/profiles.json")
        }
        
        for step, completed in statuses.items():
            status_icon = "✅" if completed else "⏳"
            status_class = "status-success" if completed else "status-error"
            st.sidebar.markdown(f'{status_icon} <span class="{status_class}">{step}</span>', 
                              unsafe_allow_html=True)
    
    def show_overview(self):
        """Show project overview and capabilities"""
        st.markdown("### 🎯 Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚀 Key Features:**
            - **Individual Engagement Profiles**: Granular profiles for high-value followers
            - **Advanced ML Models**: Random Forest, XGBoost, LightGBM, TabNet, GNN, BERT
            - **Sentiment Integration**: BERT-based sentiment analysis
            - **High-Value Focus**: Target top 10% followers by engagement/influence
            """)
            
            st.markdown("""
            **🔬 Novel Contributions:**
            - Individual vs. group-based segmentation
            - High-value follower prioritization
            - Sentiment-engagement integration
            - Advanced ML model comparison
            """)
        
        with col2:
            st.markdown("""
            **📊 Features Used:**
            - `media_type`: Content type preference
            - `Category`: Content theme preference  
            - `likes`: Aggregate engagement
            - `comments_count`: Aggregate engagement
            - `comment_text`: Sentiment analysis input
            - `comment_owner_username`: Follower ID
            - `comment_likes`: Engagement frequency
            - `#Followers`: Influence scoring
            """)
        
        # Architecture diagram
        st.markdown("### 🏗️ System Architecture")
        self.show_architecture_diagram()
        
        # Quick start button
        st.markdown("### 🚀 Quick Start")
        if st.button("🎯 Start Full Pipeline", type="primary"):
            st.info("💡 Use the sidebar to navigate through each pipeline step!")
    
    def show_architecture_diagram(self):
        """Show system architecture flow"""
        fig = go.Figure()
        
        # Create flow diagram
        steps = [
            "Dataset\n(data.csv)",
            "Preprocessing\nModule",
            "Follower Selection\nModule",
            "Sentiment Analysis\nModule", 
            "Model Training\nModule",
            "Model Evaluation\nModule",
            "Profile Generation\nModule",
            "Streamlit\nInterface"
        ]
        
        y_positions = [7, 6, 5, 4, 3, 2, 1, 0]
        
        for i, (step, y) in enumerate(zip(steps, y_positions)):
            fig.add_shape(
                type="rect",
                x0=0, y0=y-0.3, x1=2, y1=y+0.3,
                fillcolor="lightblue",
                line=dict(color="blue", width=2)
            )
            fig.add_annotation(
                x=1, y=y,
                text=step,
                showarrow=False,
                font=dict(size=12, color="black")
            )
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=1, y=y-0.5,
                    ax=1, ay=y-0.7,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="black"
                )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True, key="architecture_diagram")

    def show_preprocessing(self):
        """Show data preprocessing options"""
        st.markdown("### 📊 Data Preprocessing")
        
        data_source = st.radio(
            "Select Data Source",
            ("Upload CSV", "Process Clustered Data")
        )

        if data_source == "Upload CSV":
            self.handle_csv_upload()
        else:
            self.handle_clustered_data()

    def handle_csv_upload(self):
        """Handle CSV upload and processing"""
        st.markdown("#### Upload Raw Instagram Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                
                st.markdown("#### Raw Data Preview")
                st.dataframe(raw_df.head())
                
                if st.button("🚀 Preprocess Data", type="primary"):
                    with st.spinner("⚙️ Processing data... This may take a moment."):
                        self.data_processor.fit(raw_df)
                        processed_df = self.data_processor.transform()
                        
                        # Save processed data
                        processed_df.to_csv("outputs/preprocessed_data.csv", index=False)
                        
                        st.success("✅ Data preprocessing complete!")
                        st.markdown("#### Processed Data Preview")
                        st.dataframe(processed_df.head())
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

    def handle_clustered_data(self):
        """Handle clustered data processing"""
        st.markdown("#### Process Clustered Instagram Data")
        
        cluster_names = self.clustered_data_processor.get_cluster_names()
        if not cluster_names:
            st.warning("⚠️ No cluster directories found in `data/clustered_data`.")
            return

        selected_cluster = st.selectbox("Select a cluster to process", cluster_names)
        
        if st.button("⚙️ Process Cluster", type="primary"):
            with st.spinner(f"Processing {selected_cluster}... This might take a while."):
                try:
                    # Process the cluster data
                    raw_df = self.clustered_data_processor.process_cluster(selected_cluster)
                    
                    if raw_df.empty:
                        st.error(f"❌ No data found in cluster '{selected_cluster}'. Check the data format.")
                        return
                    
                    # Make it compatible with existing pipeline
                    processed_df = self.clustered_data_processor.create_compatible_dataframe(raw_df)
                    
                    # Store in session state
                    st.session_state['clustered_df'] = processed_df
                    st.session_state['selected_cluster'] = selected_cluster
                    
                    st.success(f"✅ Cluster '{selected_cluster}' processed successfully!")
                    st.markdown("#### Processed Cluster Data Preview")
                    st.dataframe(processed_df.head())
                    
                    # Show data statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Posts", len(processed_df))
                    with col2:
                        st.metric("Unique Users", processed_df['username'].nunique())
                    with col3:
                        st.metric("Total Comments", processed_df['comment_text'].notna().sum())
                        
                except Exception as e:
                    st.error(f"❌ Error processing cluster: {str(e)}")
                    st.info("This might be due to data format issues. Please check the .info files structure.")

        if 'clustered_df' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Generate and Save Intermediate CSV"):
                    output_path = f"outputs/{st.session_state['selected_cluster']}_preprocessed.csv"
                    st.session_state['clustered_df'].to_csv(output_path, index=False)
                    st.success(f"✅ Intermediate CSV saved to `{output_path}`")
                    st.info("You can now use the 'Upload CSV' option with this generated file.")
            
            with col2:
                if st.button("🚀 Continue with This Data"):
                    # Save as the main preprocessed data for the pipeline
                    st.session_state['clustered_df'].to_csv("outputs/preprocessed_data.csv", index=False)
                    st.success("✅ Data set as main dataset for the pipeline!")
                    st.info("You can now proceed to the next steps in the pipeline.")

    def show_follower_selection(self):
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
                    
                    # Save results using account-specific approach
                    from src.utils.high_value_utils import get_owner_id_from_data
                    owner_id = get_owner_id_from_data()
                    
                    if owner_id:
                        output_file = f"outputs/high_value_followers_{owner_id}.json"
                        st.info(f"Saving results for owner_id: {owner_id}")
                    else:
                        output_file = "outputs/high_value_followers_general.json"
                        st.info("No specific owner_id found, saving as general results")
                    
                    with open(output_file, "w") as f:
                        json.dump(high_value_followers, f)
                    
                    st.success(f"✅ Selected {len(high_value_followers)} high-value followers!")
                    
                    # Show results
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
                    
                except Exception as e:
                    st.error(f"❌ Error during selection: {str(e)}")
        
        # Show existing results if available
        from src.utils.high_value_utils import get_consolidated_high_value_followers
        existing_followers = get_consolidated_high_value_followers()
        
        if existing_followers:
            st.info(f"📊 Previously selected: {len(existing_followers)} followers")
    
    def show_sentiment_analysis(self):
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
                    self.show_sentiment_results(sentiment_scores, context="new")
                    
                except Exception as e:
                    st.error(f"❌ Error during sentiment analysis: {str(e)}")
        
        # Show existing results if available
        if os.path.exists("outputs/sentiment_scores.json"):
            with open("outputs/sentiment_scores.json", "r") as f:
                existing_scores = json.load(f)
            
            st.info(f"📊 Previously analyzed: {len(existing_scores)} comments")
            self.show_sentiment_results(existing_scores, context="existing")

    def show_sentiment_results(self, sentiment_scores, context="default"):
        """Display sentiment analysis results"""
        if not sentiment_scores:
            return
        
        # Add high-precision timestamp and random component to ensure unique keys
        import time
        import random
        timestamp = str(time.time()).replace('.', '')  # High precision timestamp
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
    
    def show_model_training(self):
        """Model training interface"""
        st.markdown("### 🤖 Advanced ML Model Training")
        
        # Check prerequisites
        from src.utils.high_value_utils import check_high_value_data_exists
        
        prerequisites = [
            ("Preprocessed Data", "outputs/preprocessed_data.csv"),
            ("High-Value Followers", check_high_value_data_exists()),
            ("Sentiment Scores", "outputs/sentiment_scores.json")
        ]
        
        missing = []
        for name, path_or_status in prerequisites:
            if isinstance(path_or_status, bool):
                if not path_or_status:
                    missing.append(name)
            elif isinstance(path_or_status, str):
                if not os.path.exists(path_or_status):
                    missing.append(name)
        if missing:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
            return
        
        # Model selection
        st.markdown("#### 🎯 Model Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            models_to_train = st.multiselect(
                "Select Models to Train:",
                ["Random Forest", "XGBoost", "LightGBM", "TabNet", "GNN", "BERT"],
                default=["Random Forest", "XGBoost", "LightGBM"]
            )
        
        with col2:
            target_variable = st.selectbox(
                "Target Variable:",
                ["engagement_probability", "comment_likelihood", "like_probability"]
            )
        
        # Training parameters
        st.markdown("#### ⚙️ Training Parameters")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
            random_state = st.number_input("Random State", value=42)
        
        with col2:
            cross_validation = st.checkbox("Cross Validation", value=True)
            cv_folds = st.slider("CV Folds", 3, 10, 5) if cross_validation else 5
        
        with col3:
            optimize_hyperparams = st.checkbox("Hyperparameter Optimization", value=False)
            n_trials = st.slider("Optimization Trials", 10, 100, 20) if optimize_hyperparams else 20
        
        # Training button
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take several minutes"):
                try:
                    results = self.model_trainer.train_models(
                        models=models_to_train,
                        target=target_variable,
                        test_size=test_size,
                        random_state=random_state,
                        cv_folds=cv_folds if cross_validation else None,
                        optimize_hyperparams=optimize_hyperparams,
                        n_trials=n_trials if optimize_hyperparams else None
                    )
                    
                    st.success("✅ Model training completed!")
                    
                    # Show training results
                    self.show_training_results(results)
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
        
        # Show existing models if available
        model_files = [
            "outputs/rf_model.pkl", "outputs/xgb_model.pkl", 
            "outputs/lgb_model.pkl", "outputs/tabnet_model.pt",
            "outputs/gnn_model.pt", "outputs/bert_model.pt"
        ]
        
        existing_models = [f for f in model_files if os.path.exists(f)]
        if existing_models:
            st.info(f"📊 Existing models: {len(existing_models)}")
            for model in existing_models:
                st.write(f"- {os.path.basename(model)}")
    
    def show_training_results(self, results):
        """Display training results"""
        if not results:
            return
        
        # Training metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Models Trained", len(results))
            if results:
                best_model = max(results.items(), key=lambda x: x[1].get('f1_score', 0))
                st.metric("Best Model", best_model[0])
                st.metric("Best F1-Score", f"{best_model[1].get('f1_score', 0):.3f}")
        
        with col2:
            # Training time comparison
            if results:
                training_times = {model: data.get('training_time', 0) for model, data in results.items()}
                fig = px.bar(
                    x=list(training_times.keys()),
                    y=list(training_times.values()),
                    title="Training Time Comparison (seconds)"
                )
                st.plotly_chart(fig, use_container_width=True, key="training_time_comparison")
    
    def show_model_evaluation(self):
        """Model evaluation interface"""
        st.markdown("### 📈 Model Evaluation")
        
        # Check if models exist
        model_files = [
            "outputs/rf_model.pkl", "outputs/xgb_model.pkl", 
            "outputs/lgb_model.pkl", "outputs/tabnet_model.pt",
            "outputs/gnn_model.pt", "outputs/bert_model.pt"
        ]
        
        existing_models = [f for f in model_files if os.path.exists(f)]
        if not existing_models:
            st.warning("⚠️ No trained models found! Please train models first.")
            return
        
        # Evaluation metrics selection
        metrics_to_show = st.multiselect(
            "Select Evaluation Metrics:",
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            default=["Accuracy", "F1-Score", "ROC-AUC"]
        )
        
        # Evaluate button
        if st.button("📊 Evaluate Models", type="primary"):
            with st.spinner("Evaluating models..."):
                try:
                    evaluation_results = self.model_evaluator.evaluate_models(
                        metrics=metrics_to_show
                    )
                    
                    # Save results
                    with open("outputs/metrics.json", "w") as f:
                        json.dump(evaluation_results, f)
                    
                    st.success("✅ Model evaluation completed!")
                    
                    # Show evaluation results
                    self.show_evaluation_results(evaluation_results, "new")
                    
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
        
        # Show existing results if available
        if os.path.exists("outputs/metrics.json"):
            with open("outputs/metrics.json", "r") as f:
                existing_results = json.load(f)
            
            st.info("📊 Previous evaluation results:")
            self.show_evaluation_results(existing_results, "previous")
    
    def show_evaluation_results(self, results, context="default"):
        """Display evaluation results"""
        if not results:
            return
        
        # Convert to DataFrame
        eval_df = pd.DataFrame(results).T
        
        # Metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart of metrics
            fig = px.bar(
                eval_df.reset_index(),
                x='index',
                y=['Accuracy', 'F1-Score', 'ROC-AUC'],
                title="Model Performance Comparison",
                barmode='group'
            )
            unique_key = f"evaluation_performance_comparison_{context}"
            st.plotly_chart(fig, use_container_width=True, key=unique_key)
        
        with col2:
            # Detailed metrics table
            st.markdown("**Detailed Metrics:**")
            st.dataframe(eval_df)
        
        # Best model identification
        if 'F1-Score' in eval_df.columns:
            best_model = eval_df['F1-Score'].idxmax()
            best_score = eval_df.loc[best_model, 'F1-Score']
            st.success(f"🏆 Best Model: {best_model} (F1-Score: {best_score:.3f})")
    
    def show_profile_generation(self):
        """Profile generation interface"""
        st.markdown("### 👤 Generate User Profiles")
        
        # Check prerequisites
        from src.utils.high_value_utils import check_high_value_data_exists
        
        prerequisites = [
            ("High-Value Followers", check_high_value_data_exists()),
            ("Sentiment Scores", "outputs/sentiment_scores.json"),
            ("Model Metrics", "outputs/metrics.json")
        ]
        
        missing = []
        for name, path_or_status in prerequisites:
            if isinstance(path_or_status, bool):
                if not path_or_status:
                    missing.append(name)
            elif isinstance(path_or_status, str):
                if not os.path.exists(path_or_status):
                    missing.append(name)
        if missing:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
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
                    self.show_profile_results(profiles, guidelines, "new")
                    
                except Exception as e:
                    st.error(f"❌ Error during profile generation: {str(e)}")
        
        # Show existing results if available
        if os.path.exists("outputs/profiles.json"):
            with open("outputs/profiles.json", "r") as f:
                existing_profiles = json.load(f)
            with open("outputs/guidelines.json", "r") as f:
                existing_guidelines = json.load(f)
            
            st.info(f"📊 Previously generated: {len(existing_profiles)} profiles")
            self.show_profile_results(existing_profiles, existing_guidelines, "previous")
    
    def show_profile_results(self, profiles, guidelines, context="default"):
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
    
    def show_visualization(self):
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
            self.show_model_performance_viz()
        elif viz_type == "Engagement Profiles" and "User Profiles" in existing_outputs:
            self.show_engagement_profiles_viz()
        elif viz_type == "Sentiment Analysis" and "Sentiment Scores" in existing_outputs:
            self.show_sentiment_analysis_viz()
        elif viz_type == "Follower Distribution" and "High-Value Followers" in existing_outputs:
            self.show_follower_distribution_viz()
        elif viz_type == "Content Preferences" and "User Profiles" in existing_outputs:
            self.show_content_preferences_viz()
        else:
            st.info(f"📊 {viz_type} visualization requires additional data processing.")
    
    def show_model_performance_viz(self):
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
    
    def show_engagement_profiles_viz(self):
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
    
    def show_sentiment_analysis_viz(self):
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
    
    def show_follower_distribution_viz(self):
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
    
    def show_content_preferences_viz(self):
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
    
    def show_live_predictions(self):
        """Show live predictions interface"""
        try:
            self.live_prediction.show()
        except Exception as e:
            st.error(f"Error showing live predictions: {str(e)}")
            st.error("Please ensure all required models and data files are available.")
    
    def generate_sample_data(self):
        """Generate sample Instagram data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data matching the required schema
        data = {
            'post_id': range(n_samples),
            'owner_id': np.random.randint(1000, 9999, n_samples),
            'likes': np.random.exponential(100, n_samples).astype(int),
            'comments_count': np.random.exponential(20, n_samples).astype(int),
            'comment_text': [f"Sample comment {i}" for i in range(n_samples)],
            'comment_owner_username': [f"user_{i%200}" for i in range(n_samples)],
            'comment_likes': np.random.exponential(5, n_samples).astype(int),
            'media_type': np.random.choice(['photo', 'video', 'album'], n_samples),
            'Category': np.random.choice(['fashion', 'travel', 'food', 'lifestyle', 'tech'], n_samples),
            '#Followers': np.random.exponential(1000, n_samples).astype(int),
        }
        
        return pd.DataFrame(data)

# Initialize and run the app
if __name__ == "__main__":
    app = InstagramEngagementApp()
    app.main()
