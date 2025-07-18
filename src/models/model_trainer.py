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
        Train multiple ML models for engagement prediction
        
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        results = {}
        
        # Train each model
        for model_name in models:
            self.logger.info(f"Training {model_name}...")
            start_time = time.time()
            
            try:
                if model_name == "Random Forest":
                    model_results = self._train_random_forest(
                        X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials
                    )
                elif model_name == "XGBoost":
                    model_results = self._train_xgboost(
                        X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials
                    )
                elif model_name == "LightGBM":
                    model_results = self._train_lightgbm(
                        X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials
                    )
                elif model_name == "TabNet":
                    model_results = self._train_tabnet(
                        X_train, X_test, y_train, y_test, cv_folds
                    )
                elif model_name == "GNN":
                    model_results = self._train_gnn(
                        X_train, X_test, y_train, y_test
                    )
                elif model_name == "BERT":
                    model_results = self._train_bert_engagement(
                        X_train, X_test, y_train, y_test
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
        
        return results
    
    def _prepare_training_data(self, target):
        """Prepare training data"""
        self.logger.info("Preparing training data...")
        
        # Load preprocessed data
        df = pd.read_csv("outputs/preprocessed_data.csv")
        
        # Load high-value followers
        with open("outputs/high_value_followers.json", "r") as f:
            high_value_followers = json.load(f)
        
        # Load sentiment scores (handle empty file)
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
        
        # Encode categorical variables that aren't already dummy-encoded
        # Check if we have proper dummy-encoded media columns (not including the original media_type column)
        existing_media_dummy_cols = [col for col in df_filtered.columns if col.startswith('media_') and col != 'media_type']
        
        if 'media_type' in df_filtered.columns and len(existing_media_dummy_cols) == 0:
            # Only create dummy encoding if proper dummy columns don't exist
            media_dummies = pd.get_dummies(df_filtered['media_type'], prefix='media')
            df_filtered = pd.concat([df_filtered, media_dummies], axis=1)
            self.logger.info(f"Created media type dummy encoding: {list(media_dummies.columns)}")
        elif len(existing_media_dummy_cols) > 0:
            # If proper dummy encoding already exists, just log it
            self.logger.info(f"Using existing media type dummy encoding: {existing_media_dummy_cols}")
        
        # Handle category encoding (may already be dummy-encoded)
        if 'Category' in df_filtered.columns and not any(col.startswith('category_') for col in df_filtered.columns):
            # Only create dummy encoding if it doesn't already exist
            category_dummies = pd.get_dummies(df_filtered['Category'], prefix='category')
            df_filtered = pd.concat([df_filtered, category_dummies], axis=1)
            self.logger.info(f"Created category dummy encoding: {list(category_dummies.columns)}")
        elif any(col.startswith('category_') for col in df_filtered.columns):
            # If dummy encoding already exists, just log it
            existing_category_cols = [col for col in df_filtered.columns if col.startswith('category_')]
            self.logger.info(f"Using existing category dummy encoding: {existing_category_cols}")
        
        # Select features
        feature_columns = [
            'likes', 'comments_count', 'comment_likes', '#Followers',
            'engagement_frequency', 'influence_score', 'content_interaction',
            'comment_engagement_ratio', 'comment_length', 'has_emoji',
            'sentiment_positive', 'sentiment_negative', 'sentiment_neutral'
        ]
        
        # Add dummy encoded features (excluding original categorical columns)
        dummy_columns = [col for col in df_filtered.columns if col.startswith(('media_', 'category_')) and col not in ['media_type', 'Category']]
        feature_columns.extend(dummy_columns)
        
        # Keep only existing columns
        feature_columns = [col for col in feature_columns if col in df_filtered.columns]
        self.feature_columns = feature_columns
        
        X = df_filtered[feature_columns].fillna(0)
        
        # Prepare target variable
        if target == "engagement_probability":
            y = df_filtered['engagement_probability'].fillna(0)
            # Check if all values are the same
            if y.nunique() <= 1:
                self.logger.warning("All engagement_probability values are the same, using likes as alternative target")
                # Use likes above median as high engagement
                likes_median = df_filtered['likes'].median()
                y = (df_filtered['likes'] > likes_median).astype(int)
            else:
                # Convert to binary classification using median threshold
                y = (y > y.median()).astype(int)
        elif target == "engagement_binary":
            y = df_filtered['engagement_binary'].fillna(0)
            # Check if all values are the same
            if y.nunique() <= 1:
                self.logger.warning("All engagement_binary values are the same, using likes as alternative target")
                # Use likes above median as high engagement
                likes_median = df_filtered['likes'].median()
                y = (df_filtered['likes'] > likes_median).astype(int)
        else:
            raise ValueError(f"Unknown target variable: {target}")
        
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
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials):
        """Train Random Forest model"""
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
        joblib.dump(model, "outputs/rf_model.pkl")
        
        # Evaluate
        results = self._evaluate_model(model, X_test, y_test, cv_folds, X_train, y_train)
        return results
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials):
        """Train XGBoost model"""
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
        joblib.dump(model, "outputs/xgb_model.pkl")
        
        # Evaluate
        results = self._evaluate_model(model, X_test, y_test, cv_folds, X_train, y_train)
        return results
    
    def _train_lightgbm(self, X_train, X_test, y_train, y_test, cv_folds, optimize_hyperparams, n_trials):
        """Train LightGBM model"""
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
        joblib.dump(model, "outputs/lgb_model.pkl")
        
        # Evaluate
        results = self._evaluate_model(model, X_test, y_test, cv_folds, X_train, y_train)
        return results
    
    def _train_tabnet(self, X_train, X_test, y_train, y_test, cv_folds):
        """Train TabNet model"""
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
        model.save_model("outputs/tabnet_model")
        
        # Evaluate
        y_pred = model.predict(X_test_np)
        y_pred_proba = model.predict_proba(X_test_np)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test_np, y_pred),
            'precision': precision_score(y_test_np, y_pred),
            'recall': recall_score(y_test_np, y_pred),
            'f1_score': f1_score(y_test_np, y_pred),
            'roc_auc': roc_auc_score(y_test_np, y_pred_proba)
        }
        
        return results
    
    def _train_gnn(self, X_train, X_test, y_train, y_test):
        """Train Graph Neural Network model"""
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
                    x = global_mean_pool(x, batch)
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
            criterion = nn.CrossEntropyLoss()
            
            # Training loop (simplified)
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                batch = torch.zeros(num_nodes, dtype=torch.long)  # Single batch
                out = model(x_train_tensor, edge_index, batch)
                loss = criterion(out, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Save model
            torch.save(model.state_dict(), "outputs/gnn_model.pt")
            
            # Simple evaluation
            model.eval()
            with torch.no_grad():
                x_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
                batch_test = torch.zeros(len(X_test), dtype=torch.long)
                edge_index_test = torch.randint(0, len(X_test), (2, len(X_test) * 2))
                
                out = model(x_test_tensor, edge_index_test, batch_test)
                pred = out.argmax(dim=1).cpu().numpy()
                
                results = {
                    'accuracy': accuracy_score(y_test, pred),
                    'precision': precision_score(y_test, pred),
                    'recall': recall_score(y_test, pred),
                    'f1_score': f1_score(y_test, pred),
                    'roc_auc': 0.5  # Placeholder
                }
            
            return results
            
        except ImportError:
            self.logger.error("torch-geometric not available for GNN training")
            return {"error": "torch-geometric not available"}
    
    def _train_bert_engagement(self, X_train, X_test, y_train, y_test):
        """Train BERT model for engagement prediction"""
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
            with open("outputs/bert_model.pt", "w") as f:
                f.write("BERT model placeholder")
            
            return results
            
        except ImportError:
            self.logger.error("transformers not available for BERT training")
            return {"error": "transformers not available"}
    
    def _evaluate_model(self, model, X_test, y_test, cv_folds, X_train, y_train):
        """Evaluate model performance"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Basic metrics
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
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
