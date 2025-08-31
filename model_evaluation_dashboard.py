"""
Model Evaluation Dashboard for Instagram User Behavior Analysis
Enhanced with Advanced ModelTrainer Integration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import sys

# Add src directory to path for advanced modules
sys.path.append('src')
try:
    from models.model_trainer import ModelTrainer
    from evaluation.model_evaluator import ModelEvaluator
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False
    
# Fallback to existing analysis module
try:
    from analysis.engagement_prediction import run, train_and_save_like_comment_models
    BASIC_MODELS_AVAILABLE = True
except ImportError:
    BASIC_MODELS_AVAILABLE = False

def show_model_evaluation_dashboard():
    """
    Display the comprehensive model evaluation dashboard with both advanced and basic model support
    """
    st.header("🎯 Model Performance Evaluation Dashboard")
    
    # Show availability status
    if ADVANCED_MODELS_AVAILABLE:
        st.success("✅ Advanced multimodal models available")
    elif BASIC_MODELS_AVAILABLE:
        st.info("ℹ️ Using basic models (Advanced models not available)")
    else:
        st.error("❌ No model training modules available")
        return
    
    # Action buttons section
    st.subheader("🚀 Model Training & Evaluation Actions")
    
    # Check if data is available in session state or try to load default data
    if 'df' not in st.session_state:
        try:
            df = pd.read_csv('data/processed_data/cleaned_merged_user_post_data.csv')
            st.session_state.df = df
            st.info("✅ Data loaded successfully for model training")
        except Exception as e:
            df = None
            st.warning(f"⚠️ No data available for training. Please load data first. Error: {e}")
    else:
        df = st.session_state.df
    
    # Configuration panel
    if ADVANCED_MODELS_AVAILABLE:
        with st.expander("⚙️ Advanced Training Configuration", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Target Configuration**")
                target_variable = st.selectbox(
                    "Target Variable:",
                    [
                        "engagement_probability",
                        "comment_likelihood", 
                        "engagement_rate",
                        "sentiment_weighted_engagement"
                    ],
                    help="Choose the target variable for prediction"
                )
                
                owner_id = st.selectbox(
                    "Owner-Specific Training:",
                    ["All Users (General)", "Custom Owner ID"],
                    help="Train models for specific Instagram account or all users"
                )
                
                if owner_id == "Custom Owner ID":
                    custom_owner = st.text_input("Enter Owner ID:", value="")
                    owner_id = custom_owner if custom_owner else None
                else:
                    owner_id = None
            
            with col2:
                st.markdown("**🤖 Model Selection**")
                models_to_train = st.multiselect(
                    "Select Models:",
                    ["Random Forest", "XGBoost", "LightGBM", "TabNet", "GNN", "BERT"],
                    default=["Random Forest", "XGBoost", "LightGBM"],
                    help="Choose which models to train"
                )
                
                use_multimodal = st.checkbox(
                    "Enable Multimodal Analysis", 
                    value=True,
                    help="Use advanced multimodal ML for enhanced prediction"
                )
            
            with col3:
                st.markdown("**⚙️ Training Options**")
                optimize_hyperparams = st.checkbox(
                    "Optimize Hyperparameters",
                    value=False,
                    help="Enable hyperparameter optimization (slower but better performance)"
                )
                
                cv_folds = st.slider(
                    "Cross-Validation Folds:",
                    min_value=3,
                    max_value=10,
                    value=5,
                    help="Number of cross-validation folds"
                )
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if ADVANCED_MODELS_AVAILABLE:
            button_text = "🎯 Train Advanced Models"
        else:
            button_text = "🎯 Train Engagement Models"
            
        if st.button(button_text, 
                     help="Train engagement prediction models",
                     disabled=(df is None)):
            if df is not None:
                with st.spinner("Training models..."):
                    try:
                        if ADVANCED_MODELS_AVAILABLE:
                            # Use advanced ModelTrainer
                            trainer = ModelTrainer()
                            
                            # Get configuration from UI
                            models = models_to_train if 'models_to_train' in locals() else ["Random Forest", "XGBoost", "LightGBM"]
                            target = target_variable if 'target_variable' in locals() else "engagement_probability"
                            optimize = optimize_hyperparams if 'optimize_hyperparams' in locals() else False
                            folds = cv_folds if 'cv_folds' in locals() else 5
                            
                            results = trainer.train_models(
                                models=models,
                                target=target,
                                optimize_hyperparams=optimize,
                                cv_folds=folds
                            )
                            
                            # Display results
                            successful_models = [k for k, v in results.items() if "error" not in v]
                            failed_models = [k for k, v in results.items() if "error" in v]
                            
                            if successful_models:
                                st.success(f"✅ Successfully trained {len(successful_models)} models!")
                                for model in successful_models:
                                    st.write(f"   • {model}")
                            
                            if failed_models:
                                st.warning(f"⚠️ Failed to train {len(failed_models)} models:")
                                for model in failed_models:
                                    error_msg = results[model].get("error", "Unknown error")
                                    st.write(f"   • {model}: {error_msg}")
                            
                            # Show feature columns saved
                            if os.path.exists("outputs/feature_columns.json"):
                                with open("outputs/feature_columns.json", "r") as f:
                                    feature_cols = json.load(f)
                                st.info(f"📝 Feature columns saved: {len(feature_cols)} features")
                        
                        else:
                            # Use basic engagement_prediction module
                            run(df)
                            st.success("✅ Engagement models trained and evaluated!")
                        
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error training models: {e}")
                        st.exception(e)
    
    with col2:
        if st.button("👍💬 Train Likes/Comments Models", 
                     help="Train separate models for likes and comments prediction",
                     disabled=(df is None)):
            if df is not None:
                with st.spinner("Training likes and comments prediction models..."):
                    try:
                        if BASIC_MODELS_AVAILABLE:
                            train_and_save_like_comment_models(df)
                            st.success("✅ Likes/Comments models trained and evaluated!")
                        else:
                            st.error("❌ Basic model training not available")
                        
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error training likes/comments models: {e}")
    
    with col3:
        if st.button("🔄 Retrain All Models", 
                     help="Train all models in sequence",
                     disabled=(df is None)):
            if df is not None:
                with st.spinner("Training all models..."):
                    try:
                        if ADVANCED_MODELS_AVAILABLE:
                            trainer = ModelTrainer()
                            targets = ["engagement_probability", "comment_likelihood", "engagement_rate"]
                            all_models = ["Random Forest", "XGBoost", "LightGBM"]
                            
                            total_results = {}
                            for target in targets:
                                st.info(f"Training models for target: {target}")
                                target_results = trainer.train_models(
                                    models=all_models,
                                    target=target,
                                    optimize_hyperparams=False
                                )
                                total_results[target] = target_results
                            
                            total_successful = sum([len([k for k, v in results.items() if "error" not in v]) for results in total_results.values()])
                            st.success(f"✅ Comprehensive training completed! {total_successful} models trained successfully!")
                        
                        else:
                            # Use basic modules
                            run(df)  # Train engagement models
                            train_and_save_like_comment_models(df)  # Train likes/comments models
                            st.success("✅ All basic models trained and evaluated successfully!")
                        
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error training all models: {e}")
    
    with col4:
        if st.button("🗑️ Clear All Evaluations", 
                     help="Clear all stored evaluation results"):
            try:
                # Clear evaluation files
                files_to_clear = [
                    "outputs/model_evaluation_engagement.json",
                    "outputs/model_evaluation_likes.json", 
                    "outputs/model_evaluation_comments.json",
                    "outputs/feature_columns.json",
                    "outputs/model_comparison.json",
                    "outputs/predictions.json",
                    "outputs/profiles.json",
                    "outputs/guidelines.json"
                ]
                
                cleared_count = 0
                for file_path in files_to_clear:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        cleared_count += 1
                
                st.success(f"✅ Cleared {cleared_count} evaluation files")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error clearing evaluations: {e}")

def main():
    """Main function for running the dashboard"""
    show_model_evaluation_dashboard()

if __name__ == "__main__":
    main()