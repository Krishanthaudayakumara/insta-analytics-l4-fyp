"""
Model Training UI Component
Handles ML model training interface
"""

import streamlit as st
import plotly.express as px
import os
from .base import BaseUIComponent


class ModelTrainingComponent(BaseUIComponent):
    """Model training component"""
    
    def show(self):
        """Model training interface"""
        st.markdown("### 🤖 Advanced ML Model Training")
        
        # Check prerequisites
        prerequisites = [
            ("Preprocessed Data", "outputs/preprocessed_data.csv"),
            ("High-Value Followers", "outputs/high_value_followers.json"),
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
                    self._show_training_results(results)
                    
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
