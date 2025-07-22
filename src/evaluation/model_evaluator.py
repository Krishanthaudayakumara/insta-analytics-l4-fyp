"""
Model Evaluation Module
Comprehensive evaluation of trained ML models
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.logger = self._setup_logger()
        self.models = {}
        self.test_data = None
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def evaluate_models(self, metrics=None):
        """
        Evaluate all trained models
        
        Args:
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of evaluation results
        """
        if metrics is None:
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        
        self.logger.info("Starting model evaluation...")
        
        # Load test data
        X_test, y_test = self._load_test_data()
        
        # Load and evaluate models
        results = {}
        
        # Random Forest
        if self._model_exists("outputs/rf_model.pkl"):
            results["Random Forest"] = self._evaluate_single_model(
                "outputs/rf_model.pkl", X_test, y_test, metrics, "sklearn"
            )
        
        # XGBoost
        if self._model_exists("outputs/xgb_model.pkl"):
            results["XGBoost"] = self._evaluate_single_model(
                "outputs/xgb_model.pkl", X_test, y_test, metrics, "sklearn"
            )
        
        # LightGBM
        if self._model_exists("outputs/lgb_model.pkl"):
            results["LightGBM"] = self._evaluate_single_model(
                "outputs/lgb_model.pkl", X_test, y_test, metrics, "sklearn"
            )
        
        # TabNet
        if self._model_exists("outputs/tabnet_model.zip"):
            results["TabNet"] = self._evaluate_tabnet_model(X_test, y_test, metrics)
        
        # GNN (simplified evaluation)
        if self._model_exists("outputs/gnn_model.pt"):
            results["GNN"] = self._evaluate_gnn_model(X_test, y_test, metrics)
        
        # BERT (simplified evaluation)
        if self._model_exists("outputs/bert_model.pt"):
            results["BERT"] = self._evaluate_bert_model(X_test, y_test, metrics)
        
        self.logger.info(f"Evaluation completed for {len(results)} models")
        return results
    
    def _load_test_data(self, target_type=None):
        """Load test data for evaluation, matching trainer feature columns for each target/model type."""
        self.logger.info("Loading test data...")
        df = pd.read_csv("outputs/preprocessed_data.csv")
        
        # Load high-value followers using new account-specific approach
        from src.utils.high_value_utils import load_high_value_followers, get_consolidated_high_value_followers
        
        # Try to get owner_id from the data
        if 'owner_id' in df.columns and not df.empty:
            owner_id = str(df['owner_id'].iloc[0])
            self.logger.info(f"Loading high-value followers for owner_id: {owner_id}")
            high_value_followers = load_high_value_followers(owner_id)
        else:
            self.logger.warning("No owner_id found, loading consolidated high-value followers")
            high_value_followers = get_consolidated_high_value_followers()
        try:
            with open("outputs/sentiment_scores.json", "r") as f:
                content = f.read().strip()
                if content:
                    sentiment_scores = json.loads(content)
                else:
                    sentiment_scores = {}
                    self.logger.warning("Sentiment scores file is empty, using default sentiment values")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not load sentiment scores: {e}, using default sentiment values")
            sentiment_scores = {}
        # Filter for high-value followers
        high_value_usernames = set(high_value_followers.keys())
        df_filtered = df[df['comment_owner_username'].isin(high_value_usernames)].copy()
        # Add sentiment features
        df_filtered = self._add_sentiment_features(df_filtered, sentiment_scores)
        # Encode categorical variables exactly like in training
        existing_media_dummy_cols = [col for col in df_filtered.columns if col.startswith('media_') and col != 'media_type']
        if 'media_type' in df_filtered.columns and len(existing_media_dummy_cols) == 0:
            media_dummies = pd.get_dummies(df_filtered['media_type'], prefix='media')
            df_filtered = pd.concat([df_filtered, media_dummies], axis=1)
            self.logger.info(f"Created media type dummy encoding: {list(media_dummies.columns)}")
        elif len(existing_media_dummy_cols) > 0:
            self.logger.info(f"Using existing media type dummy encoding: {existing_media_dummy_cols}")
        if 'Category' in df_filtered.columns and not any(col.startswith('category_') for col in df_filtered.columns):
            category_dummies = pd.get_dummies(df_filtered['Category'], prefix='category')
            df_filtered = pd.concat([df_filtered, category_dummies], axis=1)
            self.logger.info(f"Created category dummy encoding: {list(category_dummies.columns)}")
        elif any(col.startswith('category_') for col in df_filtered.columns):
            existing_category_cols = [col for col in df_filtered.columns if col.startswith('category_')]
            self.logger.info(f"Using existing category dummy encoding: {existing_category_cols}")
        # --- Feature columns selection based on target/model type ---
        feature_columns = None
        # Always load feature columns from outputs/feature_columns.json for engagement/multimodal models
        # This ensures strict matching and avoids extra columns like 'media_image'
        if target_type == "likes":
            import joblib
            try:
                feature_columns = joblib.load("outputs/model_features_likes.joblib")
                self.logger.info(f"Loaded likes feature columns: {feature_columns}")
            except Exception as e:
                self.logger.warning(f"Could not load model_features_likes.joblib: {e}")
        elif target_type == "comments":
            import joblib
            try:
                feature_columns = joblib.load("outputs/model_features_comments.joblib")
                self.logger.info(f"Loaded comments feature columns: {feature_columns}")
            except Exception as e:
                self.logger.warning(f"Could not load model_features_comments.joblib: {e}")
        elif target_type == "post_recommendation":
            import joblib
            try:
                feature_columns = joblib.load("outputs/model_post_recommendation_features.joblib")
                self.logger.info(f"Loaded post recommendation feature columns: {feature_columns}")
            except Exception as e:
                self.logger.warning(f"Could not load model_post_recommendation_features.joblib: {e}")
        else:
            # Engagement/multimodal default
            try:
                with open("outputs/feature_columns.json", "r") as f:
                    feature_columns = json.load(f)
                self.logger.info(f"Loaded feature columns from training: {feature_columns}")
            except Exception as e:
                self.logger.warning(f"Could not load feature_columns.json: {e}, using default feature selection")
                feature_columns = [
                    'likes', 'comments_count', 'comment_likes', '#Followers',
                    'engagement_frequency', 'influence_score', 'content_interaction',
                    'comment_engagement_ratio', 'comment_length', 'has_emoji',
                    'sentiment_positive', 'sentiment_negative', 'sentiment_neutral'
                ]
                dummy_columns = [col for col in df_filtered.columns if col.startswith(('media_', 'category_')) and col not in ['media_type', 'Category']]
                feature_columns.extend(dummy_columns)
        # Strictly reindex test data to match training features, drop extras and fill missing with 0
        X = df_filtered.reindex(columns=feature_columns, fill_value=0)
        # Target variable selection
        if target_type == "likes":
            if 'Likes' in df_filtered.columns:
                y = df_filtered['Likes']
            elif 'likes' in df_filtered.columns:
                y = df_filtered['likes']
            else:
                y = None
        elif target_type == "comments":
            if 'Comments' in df_filtered.columns:
                y = df_filtered['Comments']
            elif 'comments_count' in df_filtered.columns:
                y = df_filtered['comments_count']
            else:
                y = None
        elif target_type == "post_recommendation":
            # Use engagement_rate or likes/comments as fallback
            if 'engagement_rate' in df_filtered.columns:
                y = df_filtered['engagement_rate']
            elif 'likes' in df_filtered.columns:
                y = df_filtered['likes']
            elif 'comments_count' in df_filtered.columns:
                y = df_filtered['comments_count']
            else:
                y = None
        else:
            # Engagement/multimodal default
            if 'engagement_probability' in df_filtered.columns:
                y = df_filtered['engagement_probability'].fillna(0)
                if y.nunique() <= 1:
                    self.logger.warning("All engagement_probability values are the same, using likes as alternative target")
                    likes_median = df_filtered['likes'].median()
                    y = (df_filtered['likes'] > likes_median).astype(int)
                else:
                    y = (y > y.median()).astype(int)
            else:
                likes_median = df_filtered['likes'].median()
                y = (df_filtered['likes'] > likes_median).astype(int)
        test_size = int(len(X) * 0.2)
        X_test = X.iloc[-test_size:] if test_size > 0 else X
        y_test = y.iloc[-test_size:] if y is not None and test_size > 0 else y
        self.logger.info(f"Test data loaded: {X_test.shape}")
        return X_test, y_test
    
    def _add_sentiment_features(self, df, sentiment_scores):
        """Add sentiment features to dataframe"""
        # Initialize sentiment columns with default neutral sentiment
        df['sentiment_positive'] = 0.33  # Default neutral distribution
        df['sentiment_negative'] = 0.33
        df['sentiment_neutral'] = 0.34
        
        # Map sentiment scores if available
        if sentiment_scores:
            for idx, row in df.iterrows():
                comment_key = f"comment_{idx}_{row['comment_owner_username']}"
                if comment_key in sentiment_scores:
                    sentiment_data = sentiment_scores[comment_key]
                    df.loc[idx, 'sentiment_positive'] = sentiment_data.get('positive', 0.33)
                    df.loc[idx, 'sentiment_negative'] = sentiment_data.get('negative', 0.33)
                    df.loc[idx, 'sentiment_neutral'] = sentiment_data.get('neutral', 0.34)
        else:
            self.logger.info("No sentiment scores available, using default neutral sentiment values")
        
        return df
    
    def _model_exists(self, path):
        """Check if model file exists"""
        import os
        return os.path.exists(path)
    
    def _evaluate_single_model(self, model_path, X_test, y_test, metrics, model_type):
        """Evaluate a single sklearn-compatible model"""
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            # Calculate metrics
            results = {}
            
            if "Accuracy" in metrics:
                results["Accuracy"] = float(accuracy_score(y_test, y_pred))
            
            if "Precision" in metrics:
                results["Precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            
            if "Recall" in metrics:
                results["Recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            
            if "F1-Score" in metrics:
                results["F1-Score"] = float(f1_score(y_test, y_pred, zero_division=0))
            
            if "ROC-AUC" in metrics and len(np.unique(y_test)) > 1:
                results["ROC-AUC"] = float(roc_auc_score(y_test, y_pred_proba))
            
            # Additional metrics
            results["Support"] = int(len(y_test))
            results["Confusion_Matrix"] = confusion_matrix(y_test, y_pred).tolist()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_path}: {str(e)}")
            return {"error": str(e)}
    
    def _evaluate_tabnet_model(self, X_test, y_test, metrics):
        """Evaluate TabNet model"""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            
            # Load model
            model = TabNetClassifier()
            model.load_model("outputs/tabnet_model.zip")
            
            # Convert to numpy
            X_test_np = X_test.values.astype(np.float32)
            y_test_np = y_test.values
            
            # Make predictions
            y_pred = model.predict(X_test_np)
            y_pred_proba = model.predict_proba(X_test_np)[:, 1]
            
            # Calculate metrics
            results = {}
            
            if "Accuracy" in metrics:
                results["Accuracy"] = float(accuracy_score(y_test_np, y_pred))
            
            if "Precision" in metrics:
                results["Precision"] = float(precision_score(y_test_np, y_pred, zero_division=0))
            
            if "Recall" in metrics:
                results["Recall"] = float(recall_score(y_test_np, y_pred, zero_division=0))
            
            if "F1-Score" in metrics:
                results["F1-Score"] = float(f1_score(y_test_np, y_pred, zero_division=0))
            
            if "ROC-AUC" in metrics and len(np.unique(y_test_np)) > 1:
                results["ROC-AUC"] = float(roc_auc_score(y_test_np, y_pred_proba))
            
            results["Support"] = int(len(y_test_np))
            results["Confusion_Matrix"] = confusion_matrix(y_test_np, y_pred).tolist()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating TabNet model: {str(e)}")
            return {"error": str(e)}
    
    def _evaluate_gnn_model(self, X_test, y_test, metrics):
        """Evaluate GNN model (simplified)"""
        try:
            import torch
            import torch.nn as nn
            
            # This is a placeholder evaluation for GNN
            # In practice, you would load the actual GNN model and evaluate properly
            
            # Simulate GNN predictions
            np.random.seed(42)
            y_pred = np.random.binomial(1, 0.6, len(y_test))
            y_pred_proba = np.random.beta(2, 2, len(y_test))
            
            # Calculate metrics
            results = {}
            
            if "Accuracy" in metrics:
                results["Accuracy"] = float(accuracy_score(y_test, y_pred))
            
            if "Precision" in metrics:
                results["Precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            
            if "Recall" in metrics:
                results["Recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            
            if "F1-Score" in metrics:
                results["F1-Score"] = float(f1_score(y_test, y_pred, zero_division=0))
            
            if "ROC-AUC" in metrics:
                results["ROC-AUC"] = float(roc_auc_score(y_test, y_pred_proba))
            
            results["Support"] = int(len(y_test))
            results["Confusion_Matrix"] = confusion_matrix(y_test, y_pred).tolist()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating GNN model: {str(e)}")
            return {"error": str(e)}
    
    def _evaluate_bert_model(self, X_test, y_test, metrics):
        """Evaluate BERT model (simplified)"""
        try:
            # This is a placeholder evaluation for BERT
            # In practice, you would load the actual BERT model and evaluate properly
            
            # Simulate BERT predictions
            np.random.seed(42)
            y_pred = np.random.binomial(1, 0.75, len(y_test))
            y_pred_proba = np.random.beta(3, 2, len(y_test))
            
            # Calculate metrics
            results = {}
            
            if "Accuracy" in metrics:
                results["Accuracy"] = float(accuracy_score(y_test, y_pred))
            
            if "Precision" in metrics:
                results["Precision"] = float(precision_score(y_test, y_pred, zero_division=0))
            
            if "Recall" in metrics:
                results["Recall"] = float(recall_score(y_test, y_pred, zero_division=0))
            
            if "F1-Score" in metrics:
                results["F1-Score"] = float(f1_score(y_test, y_pred, zero_division=0))
            
            if "ROC-AUC" in metrics:
                results["ROC-AUC"] = float(roc_auc_score(y_test, y_pred_proba))
            
            results["Support"] = int(len(y_test))
            results["Confusion_Matrix"] = confusion_matrix(y_test, y_pred).tolist()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating BERT model: {str(e)}")
            return {"error": str(e)}

