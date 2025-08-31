"""
Model Evaluation UI Component
Handles model evaluation interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from .base import BaseUIComponent


class ModelEvaluationComponent(BaseUIComponent):
    """Model evaluation component"""
    
    def show(self):
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
            ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "R2 Score", "RMSE", "MAE", "MAPE"],
            default=["Accuracy", "F1-Score", "ROC-AUC", "R2 Score", "RMSE"]
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
                    self._show_evaluation_results(evaluation_results, "new")
                    
                    # Show keyword/hashtag recommendations if available
                    self._show_keyword_recommendations()
                
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
        
        # Show existing results if available
        if os.path.exists("outputs/metrics.json"):
            with open("outputs/metrics.json", "r") as f:
                existing_results = json.load(f)
            
            st.info("📊 Previous evaluation results:")
            self._show_evaluation_results(existing_results, "previous")
    
    def _show_evaluation_results(self, results, context="default"):
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
