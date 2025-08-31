"""
Model Training Component
Handles ML model training interface
"""

import streamlit as st
import plotly.express as px
import os
from src.ui.base import BaseUIComponent


class ModelTrainingComponent(BaseUIComponent):
    """Model training component"""
    
    def show(self):
        """Model training interface"""
        st.markdown("### 🤖 Advanced ML Model Training")
        
        # Check prerequisites
        from src.utils.high_value_utils import check_high_value_data_exists
        
        prerequisites = [
            ("Preprocessed Data", "outputs/preprocessed_data.csv"),
            ("High-Value Followers", check_high_value_data_exists()),
            ("Sentiment Scores", "outputs/sentiment_scores.json")
        ]
        
        if self.show_prerequisites_warning(prerequisites):
            return
        
        # Model selection
        st.markdown("#### 🎯 Model Selection")
        col1, col2 = st.columns(2)
        with col1:
            models_to_train = st.multiselect(
                "Select Models to Train:",
                ["Random Forest", "XGBoost", "LightGBM", "TabNet", "GNN", "BERT", "Linear Regression", "Ridge Regression"],
                default=["Random Forest", "XGBoost", "LightGBM", "Linear Regression", "Ridge Regression"]
            )
        with col2:
            target_variable = st.selectbox(
                "Target Variable:",
                [
                    "engagement_rate", "sentiment_weighted_engagement", "comment_count", "like_count", "comment_likelihood", "engagement_score"
                ]
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
        
        # Owner selection & Multimodal training
        st.markdown("#### Owner Selection & Multimodal Training")
        owner_ids = [None] + [f.split('_')[-1].split('.')[0] for f in os.listdir("outputs") if f.startswith("high_value_followers_")]
        owner_id = st.selectbox("Select Owner (user-specific or All)", owner_ids, format_func=lambda x: "All" if x is None else str(x))
        st.caption("Select a specific owner for personalized engagement prediction, or 'All' for general dataset mode.")
        use_multimodal = st.checkbox("Use Multimodal (BERT + TabNet/GNN)", value=True)
        st.caption("Multimodal combines text, graph, and tabular features for advanced prediction.")
        
        # Feature toggles
        st.markdown("#### Feature Toggles")
        use_sentiment = st.checkbox("Include Sentiment-Weighted Engagement", value=True)
        use_hashtags = st.checkbox("Include Hashtag Features", value=True)
        use_emoji = st.checkbox("Include Emoji Features", value=True)
        feature_toggles = {
            'sentiment_weighted_engagement': use_sentiment,
            'hashtags': use_hashtags,
            'emoji': use_emoji
        }
        
        # Training button
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models... This may take several minutes"):
                try:
                    results = self.model_trainer.train_engagement_model(
                        owner_id=owner_id,
                        use_multimodal=use_multimodal,
                        models=models_to_train,
                        target=target_variable if not use_sentiment else "sentiment_weighted_engagement",
                        test_size=test_size,
                        random_state=random_state,
                        cv_folds=cv_folds if cross_validation else None,
                        optimize_hyperparams=optimize_hyperparams,
                        n_trials=n_trials if optimize_hyperparams else None,
                        feature_toggles=feature_toggles
                    )
                    st.success("✅ Model training completed!")
                    # Show training results
                    self._show_training_results(results)
                    # Show model comparison chart and actionable insights
                    self._show_model_comparison_and_insights()
                    # Show keyword/hashtag recommendations if available
                    self._show_keyword_recommendations()
                    # Show recommended keywords/phrases
                    import json
                    if os.path.exists("outputs/guidelines.json"):
                        with open("outputs/guidelines.json", "r") as f:
                            guidelines = json.load(f)
                        if "recommended_keywords" in guidelines:
                            st.subheader("Recommended Keywords for Future Posts")
                            st.write(guidelines["recommended_keywords"])
                        if "caption_phrases" in guidelines:
                            st.subheader("Recommended Caption Phrases")
                            st.write(guidelines["caption_phrases"])
                        if "recommended_hashtags" in guidelines:
                            st.subheader("Recommended Hashtags")
                            st.write(guidelines["recommended_hashtags"])
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")
        
        # Show existing models if available
        self._show_existing_models()
    
    def _show_training_results(self, results):
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
    
    def _show_existing_models(self):
        """Show existing trained models"""
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
    
    def _show_model_comparison_and_insights(self):
        """Display model comparison, actionable recommendations, and insights"""
        import json
        # Model comparison
        if os.path.exists("outputs/model_comparison.json"):
            with open("outputs/model_comparison.json", "r") as f:
                comp = json.load(f)
            metrics = {k: v.get("f1_score", 0) for k, v in comp.items()}
            fig = px.bar(x=list(metrics.keys()), y=list(metrics.values()), title="Model F1-Score Comparison")
            st.plotly_chart(fig, use_container_width=True)
            # Show top features and confusion matrices
            for model, data in comp.items():
                st.subheader(f"{model} Insights")
                if "top_features" in data:
                    st.write("Top Features:", data["top_features"])
                if "confusion_matrix" in data:
                    st.write("Confusion Matrix:", data["confusion_matrix"])
                if "recommendations" in data:
                    st.write("Recommendations:", data["recommendations"])
        # Predictions
        if os.path.exists("outputs/predictions.json"):
            with open("outputs/predictions.json", "r") as f:
                preds = json.load(f)
            st.subheader("Predictions")
            st.write(preds)
        # Profiles
        if os.path.exists("outputs/profiles.json"):
            with open("outputs/profiles.json", "r") as f:
                profiles = json.load(f)
            st.subheader("Profile Insights")
            st.write(profiles)
        # Guidelines
        if os.path.exists("outputs/guidelines.json"):
            with open("outputs/guidelines.json", "r") as f:
                guidelines = json.load(f)
            st.subheader("Actionable Guidelines")
            st.write(guidelines)
    
    def _show_keyword_recommendations(self):
        """Display recommended keywords/hashtags for future posts"""
        import json
        if os.path.exists("outputs/guidelines.json"):
            with open("outputs/guidelines.json", "r") as f:
                guidelines = json.load(f)
            if "recommended_keywords" in guidelines:
                st.subheader("Recommended Keywords for Future Posts")
                st.write(guidelines["recommended_keywords"])
            if "recommended_hashtags" in guidelines:
                st.subheader("Recommended Hashtags for Future Posts")
                st.write(guidelines["recommended_hashtags"])
            if "guidelines" in guidelines:
                st.subheader("Actionable Guidelines")
                for g in guidelines["guidelines"]:
                    st.write(f"- {g}")
