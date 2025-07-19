"""
Advanced Boosting Models for Instagram Engagement Prediction
Implements XGBoost, LightGBM, and CatBoost with hyperparameter optimization
"""

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")


def calculate_advanced_metrics(y_true, y_pred, model_name="Advanced Model"):
    """Calculate comprehensive evaluation metrics for advanced models."""
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Advanced metrics
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def explained_variance_score(y_true, y_pred):
        var_y = np.var(y_true)
        return 1 - np.var(y_true - y_pred) / var_y if var_y != 0 else 0
    
    mape = mean_absolute_percentage_error(y_true, y_pred)
    explained_var = explained_variance_score(y_true, y_pred)
    
    # Accuracy within bounds
    error_bounds = [0.1, 0.2, 0.3]
    accuracy_within_bounds = {}
    
    for bound in error_bounds:
        relative_error = np.abs((y_true - y_pred) / (y_true + 1e-8))
        accuracy = np.mean(relative_error <= bound) * 100
        accuracy_within_bounds[f'accuracy_within_{int(bound*100)}%'] = accuracy
    
    return {
        'Model_Name': model_name,
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2_Score': float(r2),
        'MAPE': float(mape),
        'Explained_Variance': float(explained_var),
        'Timestamp': datetime.now().isoformat(),
        **accuracy_within_bounds
    }


class AdvancedBoostingPredictor:
    """Advanced boosting models with hyperparameter optimization."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.evaluation_results = []
    
    def prepare_advanced_features(self, df):
        """Create advanced feature engineering for boosting models."""
        df_enhanced = df.copy()
        
        # User-specific advanced features
        if '#Followers' in df.columns:
            df_enhanced['follower_tier'] = pd.cut(
                df['#Followers'], 
                bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
                labels=['nano', 'micro', 'mid', 'macro', 'mega']
            )
            
            # High-value follower approach
            df_enhanced['is_high_value_influencer'] = (
                df['#Followers'] > df['#Followers'].quantile(0.8)
            ).astype(int)
            
            df_enhanced['follower_engagement_ratio'] = (
                df['likes'] + df.get('comments_count', df.get('Comments', 0))
            ) / (df['#Followers'] + 1)
        
        # Content quality metrics
        if 'Caption' in df.columns:
            df_enhanced['content_quality_score'] = (
                df['Caption'].str.len() * 0.3 +
                df['Caption'].str.count('!') * 2 +
                df['Caption'].str.count('?') * 1.5 +
                df.get('num_hashtags', 0) * 0.5
            )
            
            # Sentiment-based features
            if 'caption_sentiment_vader' in df.columns:
                df_enhanced['sentiment_engagement_interaction'] = (
                    df['caption_sentiment_vader'] * df['engagement_rate']
                )
        
        # Temporal features
        if 'hour_of_day' in df.columns:
            df_enhanced['is_peak_time'] = df['hour_of_day'].isin([18, 19, 20, 21]).astype(int)
            df_enhanced['is_weekend'] = df.get('day_of_week', 0).isin([5, 6]).astype(int)
        
        # Cross-feature interactions
        numerical_cols = df_enhanced.select_dtypes(include=[np.number]).columns
        for i, col1 in enumerate(numerical_cols[:5]):  # Limit to avoid explosion
            for col2 in numerical_cols[i+1:6]:
                if col1 != col2:
                    df_enhanced[f'{col1}_x_{col2}'] = df_enhanced[col1] * df_enhanced[col2]
        
        return df_enhanced
    
    def train_xgboost_model(self, X, y, optimize_hyperparams=True):
        """Train XGBoost model with optional hyperparameter optimization."""
        if not XGBOOST_AVAILABLE:
            return None, None
        
        print("Training XGBoost model...")
        
        if optimize_hyperparams:
            # Hyperparameter optimization
            param_grid = {
                'n_estimators': [200, 500, 800],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            print(f"Best XGBoost params: {grid_search.best_params_}")
        else:
            # Use default optimized parameters
            best_model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            best_model.fit(X, y)
        
        # Predictions and evaluation
        y_pred = best_model.predict(X)
        metrics = calculate_advanced_metrics(y, y_pred, "XGBoost")
        
        # Feature importance
        feature_importance = dict(zip(X.columns, best_model.feature_importances_))
        metrics['feature_importance'] = feature_importance
        
        return best_model, metrics
    
    def train_lightgbm_model(self, X, y, optimize_hyperparams=True):
        """Train LightGBM model with optional hyperparameter optimization."""
        if not LIGHTGBM_AVAILABLE:
            return None, None
        
        print("Training LightGBM model...")
        
        if optimize_hyperparams:
            param_grid = {
                'n_estimators': [200, 500, 800],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'num_leaves': [31, 63, 127]
            }
            
            lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
            grid_search = GridSearchCV(
                lgb_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            print(f"Best LightGBM params: {grid_search.best_params_}")
        else:
            best_model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                num_leaves=63,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            best_model.fit(X, y)
        
        y_pred = best_model.predict(X)
        metrics = calculate_advanced_metrics(y, y_pred, "LightGBM")
        
        # Feature importance
        feature_importance = dict(zip(X.columns, best_model.feature_importances_))
        metrics['feature_importance'] = feature_importance
        
        return best_model, metrics
    
    def train_catboost_model(self, X, y, optimize_hyperparams=True):
        """Train CatBoost model with optional hyperparameter optimization."""
        if not CATBOOST_AVAILABLE:
            return None, None
        
        print("Training CatBoost model...")
        
        if optimize_hyperparams:
            param_grid = {
                'iterations': [200, 500, 800],
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5]
            }
            
            cat_model = cb.CatBoostRegressor(random_state=42, verbose=False)
            grid_search = GridSearchCV(
                cat_model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
            print(f"Best CatBoost params: {grid_search.best_params_}")
        else:
            best_model = cb.CatBoostRegressor(
                iterations=500,
                depth=8,
                learning_rate=0.05,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False
            )
            best_model.fit(X, y)
        
        y_pred = best_model.predict(X)
        metrics = calculate_advanced_metrics(y, y_pred, "CatBoost")
        
        # Feature importance
        feature_importance = dict(zip(X.columns, best_model.feature_importances_))
        metrics['feature_importance'] = feature_importance
        
        return best_model, metrics
    
    def train_advanced_ensemble(self, X, y):
        """Train ensemble of all available boosting models."""
        from sklearn.ensemble import VotingRegressor
        
        print("Training Advanced Ensemble...")
        
        # Collect available models
        ensemble_models = []
        
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )
            ensemble_models.append(('xgboost', xgb_model))
        
        if LIGHTGBM_AVAILABLE:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, num_leaves=63,
                random_state=42, n_jobs=-1, verbose=-1
            )
            ensemble_models.append(('lightgbm', lgb_model))
        
        if CATBOOST_AVAILABLE:
            cat_model = cb.CatBoostRegressor(
                iterations=300, depth=8, learning_rate=0.05,
                l2_leaf_reg=3, random_state=42, verbose=False
            )
            ensemble_models.append(('catboost', cat_model))
        
        # Add traditional models as baseline
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        )
        ensemble_models.append(('random_forest', rf_model))
        
        gb_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42
        )
        ensemble_models.append(('gradient_boosting', gb_model))
        
        if len(ensemble_models) >= 2:
            # Create voting ensemble
            ensemble = VotingRegressor(estimators=ensemble_models, n_jobs=-1)
            ensemble.fit(X, y)
            
            y_pred = ensemble.predict(X)
            metrics = calculate_advanced_metrics(y, y_pred, "Advanced_Ensemble")
            
            return ensemble, metrics
        else:
            print("Not enough models available for ensemble")
            return None, None
    
    def train_all_advanced_models(self, df, target_column='engagement_rate', optimize=True):
        """Train all available advanced models."""
        print(f"Training advanced models for target: {target_column}")
        
        # Prepare enhanced features
        df_enhanced = self.prepare_advanced_features(df)
        
        # Feature selection
        feature_cols = [
            'likes', 'shares', 'caption_length', 'num_hashtags', 'engagement_rate',
            '#Followers', '#Followees', '#Posts', 'follower_tier', 'is_high_value_influencer',
            'follower_engagement_ratio', 'content_quality_score', 'is_peak_time'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df_enhanced.columns]
        
        # Add interaction features
        interaction_features = [col for col in df_enhanced.columns if '_x_' in col]
        available_features.extend(interaction_features[:10])  # Limit to top 10
        
        # Prepare data
        X = df_enhanced[available_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Handle categorical features
        categorical_features = ['follower_tier']
        for cat_feature in categorical_features:
            if cat_feature in X.columns:
                le = LabelEncoder()
                X[cat_feature] = le.fit_transform(X[cat_feature].astype(str))
        
        # Target variable
        if target_column in df_enhanced.columns:
            y = df_enhanced[target_column].apply(pd.to_numeric, errors='coerce').fillna(0)
        else:
            print(f"Target column {target_column} not found. Using engagement_rate.")
            y = df_enhanced['engagement_rate']
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features for better performance
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), 
            columns=X_train.columns, 
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), 
            columns=X_test.columns, 
            index=X_test.index
        )
        
        self.scalers[target_column] = scaler
        
        # Train models
        models_trained = {}
        
        # XGBoost
        xgb_model, xgb_metrics = self.train_xgboost_model(X_train_scaled, y_train, optimize)
        if xgb_model is not None:
            models_trained['XGBoost'] = xgb_model
            self.evaluation_results.append(xgb_metrics)
        
        # LightGBM
        lgb_model, lgb_metrics = self.train_lightgbm_model(X_train_scaled, y_train, optimize)
        if lgb_model is not None:
            models_trained['LightGBM'] = lgb_model
            self.evaluation_results.append(lgb_metrics)
        
        # CatBoost
        cat_model, cat_metrics = self.train_catboost_model(X_train_scaled, y_train, optimize)
        if cat_model is not None:
            models_trained['CatBoost'] = cat_model
            self.evaluation_results.append(cat_metrics)
        
        # Advanced Ensemble
        ensemble_model, ensemble_metrics = self.train_advanced_ensemble(X_train_scaled, y_train)
        if ensemble_model is not None:
            models_trained['Advanced_Ensemble'] = ensemble_model
            self.evaluation_results.append(ensemble_metrics)
        
        # Save models
        for model_name, model in models_trained.items():
            model_path = f'outputs/advanced_model_{model_name.lower()}_{target_column}.joblib'
            joblib.dump(model, model_path)
            print(f"Saved {model_name} model to {model_path}")
        
        # Save scaler
        scaler_path = f'outputs/advanced_scaler_{target_column}.joblib'
        joblib.dump(scaler, scaler_path)
        
        # Save evaluation results
        results_path = f'outputs/advanced_model_evaluation_{target_column}.json'
        with open(results_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        self.models = models_trained
        
        print(f"\nAdvanced Models Training Complete!")
        print(f"Trained {len(models_trained)} models")
        print(f"Best performing model: {max(self.evaluation_results, key=lambda x: x['R2_Score'])['Model_Name']}")
        
        return models_trained, self.evaluation_results
    
    def predict_with_model(self, model_name, input_data, target_column='engagement_rate'):
        """Make predictions with a specific advanced model."""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        # Scale input data
        if target_column in self.scalers:
            input_scaled = self.scalers[target_column].transform(input_data)
            input_scaled = pd.DataFrame(input_scaled, columns=input_data.columns)
        else:
            input_scaled = input_data
        
        # Make prediction
        prediction = self.models[model_name].predict(input_scaled)
        return prediction
    
    def get_feature_importance(self, model_name, top_k=10):
        """Get feature importance for interpretability."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, model.feature_importances_))
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            return sorted_importance[:top_k]
        else:
            return None


def run_advanced_boosting_analysis(df):
    """Main function to run advanced boosting analysis."""
    predictor = AdvancedBoostingPredictor()
    
    # Train models for different targets
    targets = ['engagement_rate']
    if 'likes' in df.columns:
        targets.append('likes')
    if 'Comments' in df.columns or 'comments_count' in df.columns:
        targets.append('Comments' if 'Comments' in df.columns else 'comments_count')
    
    all_results = {}
    
    for target in targets:
        print(f"\n{'='*50}")
        print(f"Training advanced models for: {target}")
        print(f"{'='*50}")
        
        models, results = predictor.train_all_advanced_models(df, target, optimize=False)
        all_results[target] = {
            'models': models,
            'results': results
        }
    
    return all_results


if __name__ == "__main__":
    # Test with sample data
    print("Advanced Boosting Models - Test Run")
    
    # Create sample data
    sample_data = {
        'likes': np.random.randint(10, 1000, 100),
        'shares': np.random.randint(1, 50, 100),
        'caption_length': np.random.randint(10, 500, 100),
        'num_hashtags': np.random.randint(1, 30, 100),
        '#Followers': np.random.randint(1000, 100000, 100),
        '#Posts': np.random.randint(10, 1000, 100),
        'engagement_rate': np.random.uniform(0.01, 0.1, 100)
    }
    
    df = pd.DataFrame(sample_data)
    results = run_advanced_boosting_analysis(df)
    
    print("\nAdvanced Boosting Analysis Complete!")
    for target, target_results in results.items():
        print(f"\nTarget: {target}")
        for result in target_results['results']:
            print(f"  {result['Model_Name']}: R² = {result['R2_Score']:.4f}")
