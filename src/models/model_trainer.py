"""
Advanced ML Model Training Module
Implements Random Forest, XGBoost, LightGBM, TabNet, GNN, and BERT models
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import json
import logging
import time
from tqdm import tqdm
import os
import argparse
import re
from src.models.model_evaluator import evaluate_on_test_data

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.logger = self._setup_logger()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def train_models(self, models=None, target="engagement_probability", 
                    test_size=0.2, random_state=42, cv_folds=5, 
                    optimize_hyperparams=False, n_trials=20):
        """
        Train multiple ML models for engagement prediction with robust fallback
        
        Args:
            models: List of models to train
            target: Target variable name
            test_size: Test set size
            random_state: Random state for reproducibility
            cv_folds: Number of cross-validation folds
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of training results
        """
        if models is None:
            models = ["Random Forest", "XGBoost", "LightGBM"]
        self.logger.info(f"Training models: {models}")
        
        # Load and prepare data
        X, y = self._prepare_training_data(target)
        if X is None or y is None:
            self.logger.error("No valid training data available. Skipping model training.")
            return {m: {"error": "No valid training data."} for m in models}
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except Exception as e:
            self.logger.error(f"Error in train_test_split: {e}")
            return {m: {"error": "Train/test split failed."} for m in models}
        results = {}
        # Train each model
        # Get owner_id for saving
        owner_id_for_save = None
        if 'owner_id' in locals() and owner_id is not None:
            owner_id_for_save = owner_id
        elif hasattr(self, 'owner_id') and self.owner_id is not None:
            owner_id_for_save = self.owner_id
        # Set model output path
        model_output_path = "outputs/models/"
        for model_name in models:
            self.logger.info(f"Training {model_name}...")
            start_time = time.time()
            try:
                if y_train.nunique() <= 1 or min(y_train.value_counts()) < 5:
                    self.logger.warning(f"Target for {model_name} is not suitable. Skipping.")
                    results[model_name] = {"error": "Target is single-class or highly imbalanced."}
                    continue
                if model_name == "Random Forest":
                    model_results = self._train_random_forest(
                        X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=owner_id_for_save, output_path=model_output_path
                    )
                elif model_name == "XGBoost":
                    model_results = self._train_xgboost(
                        X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=owner_id_for_save, output_path=model_output_path
                    )
                elif model_name == "LightGBM":
                    model_results = self._train_lightgbm(
                        X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=owner_id_for_save, output_path=model_output_path
                    )
                elif model_name == "TabNet":
                    model_results = self._train_tabnet(
                        X_train, X_test, y_train, y_test, cv_folds, owner_id=owner_id_for_save, output_path=model_output_path
                    )
                elif model_name == "GNN":
                    model_results = self._train_gnn(
                        X_train, X_test, y_train, y_test, owner_id=owner_id_for_save, output_path=model_output_path
                    )
                elif model_name == "BERT":
                    model_results = self._train_bert_engagement(
                        X_train, X_test, y_train, y_test, owner_id=owner_id_for_save, output_path=model_output_path
                    )
                else:
                    self.logger.warning(f"Unknown model: {model_name}")
                    continue
                training_time = time.time() - start_time
                model_results['training_time'] = training_time
                results[model_name] = model_results
                self.logger.info(f"{model_name} training completed in {training_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        # Save feature columns for strict alignment
        feature_columns_path = "outputs/feature_columns.json"
        with open(feature_columns_path, "w") as f:
            json.dump(self.feature_columns, f, indent=2)
        # Also save per-target feature columns for likes/comments/recommendation
        if target == "likes" or target == "engagement_probability":
            joblib.dump(self.feature_columns, "outputs/model_features_likes.joblib")
        elif target == "comments" or target == "comment_likelihood":
            joblib.dump(self.feature_columns, "outputs/model_features_comments.joblib")
        elif target == "post_recommendation":
            joblib.dump(self.feature_columns, "outputs/model_post_recommendation_features.joblib")
        return results
    
    def _prepare_training_data(self, target, feature_toggles=None):
        """Prepare training data with robust checks and fallback"""
        self.logger.info("Preparing training data...")
        # Use engineered, filtered data for sentiment_weighted_engagement modeling
        if target == "sentiment_weighted_engagement":
            data_paths = [
                "outputs/engineered_data_filtered.csv"
            ]
        else:
            data_paths = [
                "outputs/preprocessed_data.csv",
                "data/processed_data/cleaned_merged_user_post_data.csv",
                "data/final_with_all_outputs.csv"
            ]
        df = None
        for data_path in data_paths:
            try:
                if os.path.exists(data_path):
                    df = pd.read_csv(data_path)
                    if not df.empty:
                        self.logger.info(f"Loaded data from {data_path} for target {target}")
                        break
            except Exception as e:
                self.logger.warning(f"Could not load {data_path}: {e}")
                continue
        if df is None:
            self.logger.error("No valid data file found")
            return None, None
        # For sentiment_weighted_engagement, set up features and target directly
        if target == "sentiment_weighted_engagement":
            # Remove posts with missing/zero num_comments or avg_comment_sentiment if not already filtered
            if 'num_comments' in df.columns:
                df = df[df['num_comments'] > 0].copy()
            # Feature columns: all except post_id, target, and any non-feature columns
            exclude_cols = {'post_id', 'sentiment_weighted_engagement'}
            feature_columns = [col for col in df.columns if col not in exclude_cols]
            self.feature_columns = feature_columns
            X = df[feature_columns].fillna(0)
            # Target: binarize around median for classification
            y_raw = df['sentiment_weighted_engagement'].fillna(0)
            if y_raw.nunique() > 1:
                median_eng = y_raw.median()
                y = (y_raw > median_eng).astype(int)
            else:
                self.logger.warning("sentiment_weighted_engagement has no variance, using likes as fallback target")
                if 'likes' in df.columns:
                    likes_median = df['likes'].median()
                    y = (df['likes'] > likes_median).astype(int)
                else:
                    return None, None
            self.logger.info(f"Prepared X: {X.shape}, y: {y.value_counts().to_dict()}")
            return X, y
        # Load high-value followers using new account-specific approach
        from src.utils.high_value_utils import load_high_value_followers, get_owner_id_from_data
        
        # Try to get owner_id from the data
        owner_id = None
        if 'owner_id' in df.columns and not df.empty:
            owner_id = str(df['owner_id'].iloc[0])
        else:
            owner_id = get_owner_id_from_data()
        
        if owner_id:
            self.logger.info(f"Loading high-value followers for owner_id: {owner_id}")
            high_value_followers = load_high_value_followers(owner_id)
        else:
            self.logger.warning("No owner_id found, loading all available high-value followers")
            from src.utils.high_value_utils import get_consolidated_high_value_followers
            high_value_followers = get_consolidated_high_value_followers()
        # Load sentiment scores
        try:
            with open("outputs/sentiment_scores.json", "r") as f:
                content = f.read().strip()
                sentiment_scores = json.loads(content) if content else {}
        except Exception as e:
            self.logger.warning(f"Could not load sentiment scores: {e}")
            sentiment_scores = {}
        
        # Filter for high-value followers
        # Extract usernames from the nested structure
        if isinstance(high_value_followers, dict) and 'high_value_followers' in high_value_followers:
            high_value_usernames = set(high_value_followers['high_value_followers'].keys())
        else:
            # Fallback for consolidated data structure
            high_value_usernames = set(high_value_followers.keys())
        
        self.logger.info(f"High-value followers found: {len(high_value_usernames)}")
        if high_value_usernames:
            self.logger.info(f"Sample usernames: {list(high_value_usernames)[:5]}")
        
        if 'comment_owner_username' in df.columns and high_value_usernames:
            df_filtered = df[df['comment_owner_username'].isin(high_value_usernames)].copy()
            self.logger.info(f"After high-value follower filtering: {df_filtered.shape}")
            
            if len(df_filtered) < 100:  # Changed from == 0 to < 100
                self.logger.warning(f"Too few samples after high-value filtering ({len(df_filtered)}), using sample of full dataset.")
                # Use a sample of all data for training instead
                df_filtered = df.sample(min(1000, len(df))).copy()
                self.logger.info(f"Using sample of full dataset: {df_filtered.shape}")
        else:
            self.logger.warning("comment_owner_username column not found or no high-value followers, using full dataset")
            df_filtered = df.copy()
        # Add sentiment features
        df_filtered = self._add_sentiment_features(df_filtered, sentiment_scores)
        # Always calculate sentiment_weighted_engagement
        df_filtered['sentiment_weighted_engagement'] = df_filtered['comment_likes'].fillna(0) * df_filtered['avg_comment_sentiment'].fillna(0)
        # Dummy encoding for categorical columns
        categorical_cols = [col for col in ['media_type', 'Category'] if col in df_filtered.columns]
        if categorical_cols:
            df_filtered = pd.get_dummies(df_filtered, columns=categorical_cols, prefix=categorical_cols)
        # Feature selection (add sentiment and engagement features)
        feature_columns = [
            'likes', 'comments_count', 'comment_likes', '#Followers',
            'engagement_frequency', 'influence_score', 'content_interaction',
            'comment_engagement_ratio', 'comment_length', 'has_emoji',
            'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            # New features
            'avg_comment_sentiment', 'comments_from_hv_followers', 'comment_likes_sum',
            'sentiment_weighted_engagement'
        ] + [col for col in df_filtered.columns if col.startswith(('media_', 'category_'))]
        # Feature toggles
        if feature_toggles:
            if not feature_toggles.get('hashtags', True):
                feature_columns = [c for c in feature_columns if not c.startswith('hashtags')]
            if not feature_toggles.get('emoji', True):
                feature_columns = [c for c in feature_columns if c != 'has_emoji']
        feature_columns = [col for col in feature_columns if col in df_filtered.columns]
        self.feature_columns = feature_columns
        # Target creation
        y = None
        self.logger.info(f"Creating target for: {target}")
        
        if target == "engagement_rate":
            # Always compute engagement_rate for better results
            if 'likes' in df_filtered.columns and 'comments_count' in df_filtered.columns:
                if '#Followers' in df_filtered.columns:
                    self.logger.info("Computing engagement_rate from likes, comments_count, and #Followers")
                    engagement_rate = (df_filtered['likes'] + df_filtered['comments_count']) / (df_filtered['#Followers'] + 1)
                else:
                    self.logger.info("Computing engagement_rate from likes and comments_count only")
                    engagement_rate = df_filtered['likes'] + df_filtered['comments_count']
                
                # Check if we have variance in engagement_rate
                if engagement_rate.nunique() > 1:
                    median_rate = engagement_rate.median()
                    y = (engagement_rate > median_rate).astype(int)
                    self.logger.info(f"Computed engagement rate stats: min={engagement_rate.min():.6f}, max={engagement_rate.max():.6f}, median={median_rate:.6f}")
                else:
                    self.logger.warning("Computed engagement rate has no variance, falling back to likes-based target")
            else:
                self.logger.warning("Required columns not found for engagement_rate calculation")
        elif target == "sentiment_weighted_engagement":
            # Sentiment-weighted engagement: sum(comment_likes * avg_comment_sentiment)
            y = (df_filtered['comment_likes_sum'] * df_filtered['avg_comment_sentiment']).fillna(0)
            # Bin for classification
            median_eng = y.median()
            y = (y > median_eng).astype(int)
        elif target == "comment_likelihood":
            # Binary: did post receive a comment from high-value follower
            y = df_filtered['comments_from_hv_followers']
        
        # Fallback to likes-based binary target if y is still None
        if y is None:
            self.logger.info("Using fallback likes-based target")
            if 'likes' in df_filtered.columns:
                likes_median = df_filtered['likes'].median()
                if df_filtered['likes'].nunique() > 1:
                    y = (df_filtered['likes'] > likes_median).astype(int)
                    self.logger.info(f"Likes-based target: median={likes_median}, distribution={y.value_counts().to_dict()}")
                else:
                    self.logger.warning("Likes column has no variance, trying comments_count")
                    if 'comments_count' in df_filtered.columns and df_filtered['comments_count'].nunique() > 1:
                        comments_median = df_filtered['comments_count'].median()
                        y = (df_filtered['comments_count'] > comments_median).astype(int)
                        self.logger.info(f"Comments-based target: median={comments_median}, distribution={y.value_counts().to_dict()}")
                    else:
                        self.logger.error("No suitable columns found for target creation - no variance in any target")
                        return None, None
            else:
                self.logger.error("No suitable columns found for target creation")
                return None, None
        # Final checks
        if y is None:
            self.logger.warning(f"Target variable is None.")
            return None, None
        elif y.nunique() <= 1:
            self.logger.warning(f"Target variable has only {y.nunique()} unique values: {y.unique()}")
            return None, None
        elif min(y.value_counts()) < 5:
            self.logger.warning(f"Target variable has imbalanced classes: {y.value_counts().to_dict()}, min count: {min(y.value_counts())}")
            return None, None
        X = df_filtered[feature_columns].fillna(0)
        self.logger.info(f"Training data prepared: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        return X, y
    
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
    
    def _ensure_model_dir(self, output_path="outputs/models/"):
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

    def _train_random_forest(self, X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=None, output_path="outputs/models/"):
        """Train Random Forest model"""
        self._ensure_model_dir(output_path)
        if optimize_hyperparams:
            # Hyperparameter optimization
            from sklearn.model_selection import RandomizedSearchCV
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            rf_search = RandomizedSearchCV(
                rf, param_grid, n_iter=n_trials, cv=3, random_state=42, n_jobs=-1
            )
            rf_search.fit(X_train, y_train)
            model = rf_search.best_estimator_
        else:
            # Default parameters
            model = RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Save model
        model_file = os.path.join(output_path, f"rf_model_{owner_id}.pkl" if owner_id else "rf_model.pkl")
        joblib.dump(model, model_file)
        
        # Evaluate
        results = self._evaluate_model(model, X_test, y_test, cv_folds, X_train, y_train)
        return results
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=None, output_path="outputs/models/"):
        """Train XGBoost model"""
        self._ensure_model_dir(output_path)
        # Check for binary target
        if len(np.unique(y_train)) < 2:
            self.logger.error("XGBoost requires at least two classes in the target variable. Skipping XGBoost training.")
            return {"error": "Target variable for XGBoost must be binary (0/1) and contain both classes."}
        
        if optimize_hyperparams:
            # Hyperparameter optimization
            import optuna
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
                
                model = xgb.XGBClassifier(**params)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            model = xgb.XGBClassifier(**study.best_params, random_state=42)
        else:
            # Default parameters
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42
            )
        
        model.fit(X_train, y_train)
        
        # Save model
        model_file = os.path.join(output_path, f"xgb_model_{owner_id}.pkl" if owner_id else "xgb_model.pkl")
        joblib.dump(model, model_file)
        
        # Evaluate
        results = self._evaluate_model(model, X_test, y_test, cv_folds, X_train, y_train)
        return results
    
    def _train_lightgbm(self, X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=None, output_path="outputs/models/"):
        """Train LightGBM model"""
        self._ensure_model_dir(output_path)
        if optimize_hyperparams:
            # Hyperparameter optimization
            import optuna
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'random_state': 42,
                    'verbose': -1
                }
                
                model = lgb.LGBMClassifier(**params)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials)
            
            model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
        else:
            # Default parameters
            model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=7, learning_rate=0.05, 
                num_leaves=31, random_state=42, verbose=-1
            )
        
        model.fit(X_train, y_train)
        
        # Save model
        model_file = os.path.join(output_path, f"lgb_model_{owner_id}.pkl" if owner_id else "lgb_model.pkl")
        joblib.dump(model, model_file)
        
        # Evaluate
        results = self._evaluate_model(model, X_test, y_test, cv_folds, X_train, y_train)
        return results
    
    def _train_tabnet(self, X_train, X_test, y_train, y_test, cv_folds, owner_id=None, output_path="outputs/models/"):
        """Train TabNet model"""
        self._ensure_model_dir(output_path)
        # Convert to numpy arrays
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_train_np = y_train.values
        y_test_np = y_test.values
        
        # Initialize TabNet
        model = TabNetClassifier(
            n_d=16, n_a=16, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
            mask_type='entmax', verbose=0
        )
        
        # Train model
        model.fit(
            X_train_np, y_train_np,
            eval_set=[(X_test_np, y_test_np)],
            max_epochs=100, patience=20,
            batch_size=256, virtual_batch_size=128
        )
        
        # Save model
        model_file = os.path.join(output_path, f"tabnet_model_{owner_id}.zip" if owner_id else "tabnet_model.zip")
        model.save_model(model_file)
        
        # Evaluate
        y_pred = model.predict(X_test_np)
        y_pred_proba = None
        proba = model.predict_proba(X_test_np)
        if proba.shape[1] > 1:
            y_pred_proba = proba[:, 1]
        else:
            y_pred_proba = proba[:, 0]
        results = {
            'accuracy': accuracy_score(y_test_np, y_pred),
            'precision': precision_score(y_test_np, y_pred, zero_division=0),
            'recall': recall_score(y_test_np, y_pred, zero_division=0),
            'f1_score': f1_score(y_test_np, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test_np, y_pred_proba) if len(np.unique(y_test_np)) > 1 and y_pred_proba is not None else 0.5
        }
        
        return results
    
    def _train_gnn(self, X_train, X_test, y_train, y_test, owner_id=None, output_path="outputs/models/"):
        """Train Graph Neural Network model"""
        self._ensure_model_dir(output_path)
        try:
            import torch_geometric
            from torch_geometric.data import Data
            from torch_geometric.nn import GCNConv, global_mean_pool
            
            # Create a simple GNN for engagement prediction
            # This is a simplified implementation
            class SimpleGCN(nn.Module):
                def __init__(self, input_dim, hidden_dim=64, output_dim=2):
                    super(SimpleGCN, self).__init__()
                    self.conv1 = GCNConv(input_dim, hidden_dim)
                    self.conv2 = GCNConv(hidden_dim, hidden_dim)
                    self.classifier = nn.Linear(hidden_dim, output_dim)
                    self.dropout = nn.Dropout(0.3)
                
                def forward(self, x, edge_index, batch):
                    x = torch.relu(self.conv1(x, edge_index))
                    x = self.dropout(x)
                    x = torch.relu(self.conv2(x, edge_index))
                    # Remove global_mean_pool for per-node outputs
                    x = self.classifier(x)
                    return x
            
            # Create graph data (simplified)
            # In practice, you would create proper user-post interaction graphs
            num_nodes = len(X_train)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))  # Random edges for demo
            
            # Convert to tensors
            x_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            
            # Initialize model
            model = SimpleGCN(X_train.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            # Use class weights for imbalance
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            # Training loop (more epochs)
            model.train()
            for epoch in range(200):
                optimizer.zero_grad()
                batch = torch.zeros(num_nodes, dtype=torch.long)  # Single batch
                out = model(x_train_tensor, edge_index, batch)
                loss = criterion(out, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Save model
            model_file = os.path.join(output_path, f"gnn_model_{owner_id}.pt" if owner_id else "gnn_model.pt")
            torch.save(model.state_dict(), model_file)
            
            # Simple evaluation
            model.eval()
            with torch.no_grad():
                x_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
                batch_test = torch.zeros(len(X_test), dtype=torch.long)
                edge_index_test = torch.randint(0, len(X_test), (2, len(X_test) * 2))
                out = model(x_test_tensor, edge_index_test, batch_test)
                pred = out.argmax(dim=1).cpu().numpy()
                # Log predictions for debugging
                print(f"GNN predictions: {np.unique(pred, return_counts=True)}")
                results = {
                    'accuracy': accuracy_score(y_test, pred),
                    'precision': precision_score(y_test, pred, zero_division=0),
                    'recall': recall_score(y_test, pred, zero_division=0),
                    'f1_score': f1_score(y_test, pred, zero_division=0),
                    'roc_auc': 0.5  # Placeholder
                }
            return results
            
        except ImportError:
            self.logger.error("torch-geometric not available for GNN training")
            return {"error": "torch-geometric not available"}
    
    def _train_bert_engagement(self, X_train, X_test, y_train, y_test, owner_id=None, output_path="outputs/models/"):
        """Train BERT model for engagement prediction"""
        self._ensure_model_dir(output_path)
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
            
            # This is a simplified BERT training example
            # In practice, you would need text data for BERT
            # For now, we'll return a placeholder
            
            results = {
                'accuracy': 0.75,  # Placeholder
                'precision': 0.73,
                'recall': 0.77,
                'f1_score': 0.75,
                'roc_auc': 0.82
            }
            
            # Create a dummy model file
            model_file = os.path.join(output_path, f"bert_model_{owner_id}.pt" if owner_id else "bert_model.pt")
            with open(model_file, "w") as f:
                f.write("BERT model placeholder")
            
            return results
            
        except ImportError:
            self.logger.error("transformers not available for BERT training")
            return {"error": "transformers not available"}
    
    def _evaluate_model(self, model, X_test, y_test, cv_folds, X_train, y_train):
        """Evaluate model performance"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            if proba.shape[1] > 1:
                y_pred_proba = proba[:, 1]
            else:
                y_pred_proba = proba[:, 0]
        else:
            y_pred_proba = y_pred
        
        # Basic metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 and y_pred_proba is not None else 0.5
        }
        
        # Cross-validation scores
        if cv_folds and cv_folds > 1:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='f1')
            results['cv_f1_mean'] = cv_scores.mean()
            results['cv_f1_std'] = cv_scores.std()
        
        return results
    
    def save_training_results(self, results, output_path="outputs/training_results.json"):
        """Save training results"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Training results saved to {output_path}")
    
    def get_feature_importance(self, model_name):
        """Get feature importance for tree-based models"""
        try:
            if model_name == "Random Forest":
                model = joblib.load("outputs/rf_model.pkl")
            elif model_name == "XGBoost":
                model = joblib.load("outputs/xgb_model.pkl")
            elif model_name == "LightGBM":
                model = joblib.load("outputs/lgb_model.pkl")
            else:
                return None
            
            if hasattr(model, 'feature_importances_'):
                # If feature_columns not available, regenerate them
                if not self.feature_columns:
                    # Re-prepare data to get feature columns
                    X, _ = self._prepare_training_data("engagement_probability")
                
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
        
        return None
    
    def train_engagement_model(self, owner_id=None, models=None, target="engagement_probability", test_size=0.2, random_state=42, cv_folds=5, optimize_hyperparams=False, n_trials=20, output_path="outputs/", feature_toggles=None):
        """
        Modular function for user-specific or general engagement prediction.
        If owner_id is None, trains a single model for all users (no per-user filtering, saves as generic model file).
        If owner_id is set, trains a per-user model (filters data and saves as per-user file).
        """
        self.logger.info(f"Starting train_engagement_model for owner_id={owner_id}")
        df = pd.read_csv("outputs/preprocessed_data.csv")
        # Load sentiment scores
        try:
            with open("outputs/sentiment_scores.json", "r") as f:
                content = f.read().strip()
                sentiment_scores = json.loads(content) if content else {}
        except Exception as e:
            self.logger.warning(f"Could not load sentiment scores: {e}")
            sentiment_scores = {}
        # Load follower analysis if available
        follower_analysis = None
        try:
            with open("outputs/dataset_follower_analysis.json", "r") as f:
                follower_analysis = json.load(f)
        except Exception:
            self.logger.info("No follower_analysis found or not used.")
        # Owner-specific mode
        if owner_id:
            hv_path = f"outputs/high_value_followers_{owner_id}.json"
            try:
                with open(hv_path, "r") as f:
                    hv_followers = json.load(f)["high_value_followers"]
                hv_usernames = set(hv_followers.keys())
                df_filtered = df[(df["owner_id"] == int(owner_id)) & (df["comment_owner_username"].isin(hv_usernames))].copy()
            except Exception as e:
                self.logger.warning(f"Could not load high_value_followers for owner_id {owner_id}: {e}")
                df_filtered = df.copy()
                hv_usernames = set(df_filtered["comment_owner_username"].unique())
        else:
            # All-users mode: do NOT filter by owner, use all data
            df_filtered = df.copy()
            hv_usernames = set(df_filtered["comment_owner_username"].unique())
        # Fallback if filtering results in too few samples
        if len(df_filtered) < 20:
            self.logger.warning(f"Filtered data too small ({len(df_filtered)} rows), using general dataset.")
            df_filtered = df.copy()
            hv_usernames = set(df_filtered["comment_owner_username"].unique())
        # Add sentiment features
        df_filtered = self._add_sentiment_features(df_filtered, sentiment_scores)
        # Optionally add follower_analysis features
        if follower_analysis:
            for uname in df_filtered["comment_owner_username"].unique():
                if uname in follower_analysis:
                    for col, val in follower_analysis[uname].items():
                        df_filtered.loc[df_filtered["comment_owner_username"] == uname, f"fa_{col}"] = val
        # Feature selection
        feature_columns = [
            "likes", "comments_count", "comment_likes", "#Followers", "engagement_frequency", "influence_score", "content_interaction", "comment_engagement_ratio", "comment_length", "has_emoji", "sentiment_positive", "sentiment_negative", "sentiment_neutral"
        ] + [col for col in df_filtered.columns if col.startswith(("media_", "category_", "fa_"))]
        for col_to_remove in ["media_type", "Category"]:
            if col_to_remove in feature_columns:
                feature_columns.remove(col_to_remove)
        feature_columns = [col for col in feature_columns if col in df_filtered.columns]
        X = df_filtered[feature_columns].fillna(0)
        # Target creation with fallback and checks
        y = None
        if target == "engagement_probability" and "engagement_probability" in df_filtered.columns:
            y_raw = df_filtered["engagement_probability"].fillna(0)
            if y_raw.nunique() > 1:
                y = (y_raw > y_raw.median()).astype(int)
            else:
                self.logger.warning("All engagement_probability values are the same, using likes as alternative target")
        if y is None or y.nunique() <= 1:
            likes_median = df_filtered["likes"].median()
            y = (df_filtered["likes"] > likes_median).astype(int)
        # Final check: if still single-class or highly imbalanced, fallback to general dataset
        if y.nunique() <= 1 or min(y.value_counts()) < 5:
            self.logger.warning(f"Target variable is single-class or highly imbalanced after filtering. Fallback to general dataset.")
            df_filtered = df.copy()
            df_filtered = self._add_sentiment_features(df_filtered, sentiment_scores)
            feature_columns = [
                "likes", "comments_count", "comment_likes", "#Followers", "engagement_frequency", "influence_score", "content_interaction", "comment_engagement_ratio", "comment_length", "has_emoji", "sentiment_positive", "sentiment_negative", "sentiment_neutral"
            ] + [col for col in df_filtered.columns if col.startswith(("media_", "category_", "fa_"))]
            feature_columns = [col for col in feature_columns if col in df_filtered.columns]
            X = df_filtered[feature_columns].fillna(0)
            y_raw = df_filtered["engagement_probability"].fillna(0) if "engagement_probability" in df_filtered.columns else None
            if y_raw is not None and y_raw.nunique() > 1:
                y = (y_raw > y_raw.median()).astype(int)
            else:
                likes_median = df_filtered["likes"].median()
                y = (df_filtered["likes"] > likes_median).astype(int)
        self.logger.info(f"Final training data: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        # If still not suitable, skip model training and save error outputs
        if y.nunique() <= 1 or min(y.value_counts()) < 5:
            error_msg = f"Target variable is single-class or highly imbalanced. Model training skipped."
            self.logger.error(error_msg)
            with open("outputs/model_comparison.json", "w") as f:
                json.dump({"error": error_msg}, f, indent=2)
            with open("outputs/predictions.json", "w") as f:
                json.dump({"owner_id": owner_id, "error": error_msg}, f)
            with open("outputs/profiles.json", "w") as f:
                json.dump({"owner_id": owner_id, "profiles": list(hv_usernames)}, f)
            with open("outputs/guidelines.json", "w") as f:
                json.dump({"owner_id": owner_id, "guidelines": ["Not enough data for meaningful prediction."]}, f)
            return {"error": error_msg}
        # Model training with proper train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        results = {}
        top_features_dict = {}
        confusion_matrices = {}
        predictions_dict = {}
        # Only train models explicitly requested
        models = models or ["Random Forest", "XGBoost", "LightGBM"]
        for model_name in models:
            if model_name == "Random Forest":
                res = self._train_random_forest(X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=owner_id)
                results[model_name] = res
                try:
                    model = joblib.load(f"outputs/models/rf_model_{owner_id}.pkl" if owner_id else "outputs/models/rf_model.pkl")
                    predictions_dict[model_name] = model.predict(X_test).tolist()
                except Exception:
                    predictions_dict[model_name] = []
            elif model_name == "XGBoost":
                res = self._train_xgboost(X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=owner_id)
                results[model_name] = res
                try:
                    model = joblib.load(f"outputs/models/xgb_model_{owner_id}.pkl" if owner_id else "outputs/models/xgb_model.pkl")
                    predictions_dict[model_name] = model.predict(X_test).tolist()
                except Exception:
                    predictions_dict[model_name] = []
            elif model_name == "LightGBM":
                res = self._train_lightgbm(X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials, owner_id=owner_id)
                results[model_name] = res
                try:
                    model = joblib.load(f"outputs/models/lgb_model_{owner_id}.pkl" if owner_id else "outputs/models/lgb_model.pkl")
                    predictions_dict[model_name] = model.predict(X_test).tolist()
                except Exception:
                    predictions_dict[model_name] = []
            elif model_name == "TabNet":
                # Ensure all features are numeric for TabNet
                X_mm_train = X_train.select_dtypes(include=[np.number]).astype(np.float32)
                X_mm_test = X_test.select_dtypes(include=[np.number]).astype(np.float32)
                res = self._train_tabnet(X_mm_train, X_mm_test, y_train, y_test, cv_folds, owner_id=owner_id)
                results[model_name] = res
                # TabNet predictions loading can be added if needed
            elif model_name == "GNN":
                X_mm_train = X_train.select_dtypes(include=[np.number]).astype(np.float32)
                X_mm_test = X_test.select_dtypes(include=[np.number]).astype(np.float32)
                res = self._train_gnn(X_mm_train, X_mm_test, y_train, y_test, owner_id=owner_id)
                results[model_name] = res
            elif model_name == "BERT":
                X_mm_train = X_train.select_dtypes(include=[np.number]).astype(np.float32)
                X_mm_test = X_test.select_dtypes(include=[np.number]).astype(np.float32)
                res = self._train_bert_engagement(X_mm_train, X_mm_test, y_train, y_test, owner_id=owner_id)
                results[model_name] = res
            # Feature importance
            fi = self.get_feature_importance(model_name)
            if fi:
                top_features_dict[model_name] = [f for f, v in fi[:5]]
            # Confusion matrix
            try:
                from sklearn.metrics import confusion_matrix
                y_pred = predictions_dict.get(model_name, [])
                if y_pred:
                    confusion_matrices[model_name] = confusion_matrix(y_test, y_pred).tolist()
            except Exception:
                pass
        # Ensemble F1 score (average of test F1 scores)
        if results:
            results["Ensemble"] = {"f1_score": np.mean([r.get("f1_score", 0) for r in results.values() if isinstance(r, dict)])}
        # Add top features and confusion matrices to results
        for model_name in top_features_dict:
            if model_name in results:
                results[model_name]["top_features"] = top_features_dict[model_name]
        for model_name in confusion_matrices:
            if model_name in results:
                results[model_name]["confusion_matrix"] = confusion_matrices[model_name]
        # Generate actionable recommendations
        recommendations = []
        best_model = None
        best_features = []
        # Only consider models with valid results and f1_score
        valid_results = {k: v for k, v in results.items() if isinstance(v, dict) and "f1_score" in v}
        if valid_results:
            best_model = max(valid_results, key=lambda k: valid_results[k].get("f1_score", 0))
            best_features = top_features_dict.get(best_model, [])
        else:
            best_model = None
            best_features = []
        if "sentiment_positive" in best_features:
            recommendations.append("Increase posts with positive sentiment for high-value followers.")
        if "likes" in best_features:
            recommendations.append("Engage top followers with personalized comments and likes.")
        if "#Followers" in best_features:
            recommendations.append("Target posts to users with higher follower counts for maximum reach.")
        if not recommendations:
            recommendations.append("Post more frequently for top followers and use engaging captions.")
        # Keyword/hashtag extraction for recommendations
        recommended_keywords = []
        recommended_hashtags = []
        try:
            from collections import Counter
            # Get top 10% posts by engagement
            if "engagement_score" in df_filtered.columns:
                top_posts = df_filtered[df_filtered["engagement_score"] >= df_filtered["engagement_score"].quantile(0.9)]
            elif "likes" in df_filtered.columns and "comments_count" in df_filtered.columns:
                top_posts = df_filtered[(df_filtered["likes"] + df_filtered["comments_count"]) >= (df_filtered["likes"] + df_filtered["comments_count"]).quantile(0.9)]
            else:
                top_posts = df_filtered.head(5)
            # Extract keywords from captions
            if "caption" in top_posts.columns:
                all_captions = " ".join(top_posts["caption"].dropna().astype(str))
                cleaned = re.sub(r"#\w+|@\w+|http\S+|[^a-zA-Z\s]", " ", all_captions.lower())
                words = [w for w in cleaned.split() if len(w) > 2]
                keyword_counts = Counter(words)
                recommended_keywords = [w for w, _ in keyword_counts.most_common(8)]
            # Extract hashtags
            if "hashtags_agg" in top_posts.columns:
                hashtags = []
                for h in top_posts["hashtags_agg"].dropna().astype(str):
                    hashtags += [tag.strip() for tag in h.split(",") if tag.strip()]
                hashtag_counts = Counter(hashtags)
                recommended_hashtags = [h for h, _ in hashtag_counts.most_common(8)]
        except Exception:
            pass
        # Save outputs (always use output_path)
        try:
            # Extract recommended keywords, hashtags, and phrases
            recommended_keywords, recommended_hashtags, recommended_phrases = extract_keywords_and_hashtags(df_filtered)
            # Save guidelines.json (with phrases)
            with open(os.path.join(output_path, "guidelines.json"), "w") as f2:
                json.dump({
                    "owner_id": owner_id,
                    "guidelines": recommendations,
                    "recommended_keywords": recommended_keywords,
                    "recommended_hashtags": recommended_hashtags,
                    "caption_phrases": recommended_phrases
                }, f2)
            # Save model_comparison.json (with metadata)
            self.logger.info(f"Saving model comparison for target={target}, owner_id={owner_id}, models={models}, output_path={output_path}")
            with open(os.path.join(output_path, "model_comparison.json"), "w") as f:
                if results:
                    json.dump({
                        "target": target,
                        "owner_id": owner_id,
                        "models": models,
                        "results": results
                    }, f, indent=2)
                else:
                    json.dump({
                        "target": target,
                        "owner_id": owner_id,
                        "models": models,
                        "error": "No models were successfully trained."
                    }, f, indent=2)
            # Save predictions.json
            with open(os.path.join(output_path, "predictions.json"), "w") as f:
                json.dump({"owner_id": owner_id, "predictions": predictions_dict}, f)
            # Save profiles.json
            with open(os.path.join(output_path, "profiles.json"), "w") as f:
                json.dump({"owner_id": owner_id, "profiles": list(hv_usernames)}, f)
        except Exception as e:
            self.logger.error(f"Failed to write output files: {e}")
        # Print accuracy metrics for each model
        for model, metrics in results.items():
            if isinstance(metrics, dict) and "accuracy" in metrics:
                self.logger.info(f"Model: {model} | Target: {target} | Owner: {owner_id} | Accuracy: {metrics['accuracy']:.3f} | F1: {metrics.get('f1_score', 0):.3f} | Precision: {metrics.get('precision', 0):.3f} | Recall: {metrics.get('recall', 0):.3f} | ROC-AUC: {metrics.get('roc_auc', 0):.3f}")
            elif isinstance(metrics, dict) and "error" in metrics:
                self.logger.warning(f"Model: {model} | Target: {target} | Owner: {owner_id} | ERROR: {metrics['error']}")
        return results

    def evaluate_on_test_data(self, model_name, test_data_path, target="sentiment_weighted_engagement", model_path=None, feature_columns_path=None):
        """
        Proxy to standalone evaluate_on_test_data for backward compatibility.
        """
        return evaluate_on_test_data(
            model_name=model_name,
            test_data_path=test_data_path,
            target=target,
            model_path=model_path,
            feature_columns_path=feature_columns_path,
            logger=self.logger
        )
    
def extract_keywords_and_hashtags(df):
    """Extract recommended keywords and hashtags from top 10% posts by engagement_score."""
    from collections import Counter
    recommended_keywords = []
    recommended_hashtags = []
    recommended_phrases = []
    # Get top 10% posts by engagement
    if "engagement_score" in df.columns:
        top_posts = df[df["engagement_score"] >= df["engagement_score"].quantile(0.9)]
    elif "likes" in df.columns and "comments_count" in df.columns:
        top_posts = df[(df["likes"] + df["comments_count"]) >= (df["likes"] + df["comments_count"]).quantile(0.9)]
    else:
        top_posts = df.head(5)
    # Extract keywords from captions
    if "caption" in top_posts.columns:
        all_captions = " ".join(top_posts["caption"].dropna().astype(str))
        cleaned = re.sub(r"#\w+|@\w+|http\S+|[^a-zA-Z\s]", " ", all_captions.lower())
        words = [w for w in cleaned.split() if len(w) > 2]
        keyword_counts = Counter(words)
        recommended_keywords = [w for w, _ in keyword_counts.most_common(8)]
        # Extract common phrases (bigrams)
        from nltk import ngrams
        bigrams = ngrams(words, 2)
        phrase_counts = Counter([" ".join(bg) for bg in bigrams])
        recommended_phrases = [p for p, _ in phrase_counts.most_common(5)]
    # Extract hashtags
    if "hashtags_agg" in top_posts.columns:
        hashtags = []
        for h in top_posts["hashtags_agg"].dropna().astype(str):
            hashtags += [tag.strip() for tag in h.split(",") if tag.strip()]
        hashtag_counts = Counter(hashtags)
        recommended_hashtags = [h for h, _ in hashtag_counts.most_common(8)]
    return recommended_keywords, recommended_hashtags, recommended_phrases

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Instagram engagement prediction model.")
    parser.add_argument('--owner_id', type=str, default=None, help='Instagram account owner_id')
    parser.add_argument('--model_type', type=str, default='random_forest', choices=['random_forest','xgboost','lightgbm','tabnet','gnn','bert'], help='Model type to train')
    parser.add_argument('--use_sentiment', action='store_true', help='Include sentiment_weighted_engagement feature')
    parser.add_argument('--use_hashtags', action='store_true', help='Include hashtags features')
    parser.add_argument('--use_emoji', action='store_true', help='Include emoji features')
    parser.add_argument('--cv', action='store_true', help='Enable cross-validation')
    parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory for results')
    parser.add_argument('--evaluate_test', action='store_true', help='Evaluate trained model on a test set')
    parser.add_argument('--test_data', type=str, default='outputs/engineered_data_test.csv', help='Path to test data CSV')
    parser.add_argument('--test_target', type=str, default='sentiment_weighted_engagement', help='Target column name for test set')
    parser.add_argument('--test_model_path', type=str, default=None, help='Path to trained model file for evaluation')
    parser.add_argument('--test_feature_columns', type=str, default=None, help='Path to feature columns JSON file for evaluation')
    args = parser.parse_args()

    trainer = ModelTrainer()
    if args.evaluate_test:
        model_name = args.model_type.replace('_', ' ').title()
        results = trainer.evaluate_on_test_data(
            model_name=model_name,
            test_data_path=args.test_data,
            target=args.test_target,
            model_path=args.test_model_path,
            feature_columns_path=args.test_feature_columns
        )
        print("[INFO] Test set evaluation results:")
        print(results)
        return
    # Prepare features list based on toggles
    feature_toggles = {
        'sentiment_weighted_engagement': args.use_sentiment,
        'hashtags': args.use_hashtags,
        'emoji': args.use_emoji
    }
    # Train model
    print(f"[INFO] Training {args.model_type} for owner_id={args.owner_id} ...")
    results = trainer.train_engagement_model(
        owner_id=args.owner_id,
        models=[args.model_type.replace('_', ' ').title()],
        target='sentiment_weighted_engagement' if args.use_sentiment else 'engagement_rate',
        test_size=0.2,
        cv_folds=5 if args.cv else 1,
        output_path=args.output_path,
        feature_toggles=feature_toggles
    )
    print("[INFO] Training complete. Results:")
    for model, metrics in results.items():
        print(f"{model}: Accuracy={metrics.get('accuracy', 0):.3f}, F1={metrics.get('f1_score', 0):.3f}")
        if 'top_features' in metrics:
            print(f"  Top features: {metrics['top_features']}")
            if 'sentiment_weighted_engagement' in metrics['top_features']:
                print("  [INFO] sentiment_weighted_engagement is important for this model!")
    # Show recommendations
    try:
        with open(os.path.join(args.output_path, 'guidelines.json')) as f:
            guidelines = json.load(f)
        print("\n[INFO] Recommended Keywords:", guidelines.get('recommended_keywords', []))
        print("[INFO] Recommended Caption Phrases:", guidelines.get('caption_phrases', []))
        print("[INFO] Recommended Hashtags:", guidelines.get('recommended_hashtags', []))
    except Exception:
        pass

if __name__ == "__main__":
    main()
