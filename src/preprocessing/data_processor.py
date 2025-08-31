"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import json
import logging

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def process_data(self, df, handle_missing="impute_median", normalize=True, cap_outliers=True):
        """
        Main data processing pipeline
        
        Args:
            df: Raw Instagram dataset
            handle_missing: How to handle missing values
            normalize: Whether to normalize numerical features
            cap_outliers: Whether to cap outliers in numerical features
            
        Returns:
            Processed DataFrame
        """
        self.logger.info("Starting data preprocessing...")
        
        # Validate required columns
        required_cols = ['media_type', 'Category', 'likes', 'comments_count', 
                        'comment_text', 'comment_owner_username', 'comment_likes', '#Followers']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # 1. Handle missing values
        processed_df = self._handle_missing_values(processed_df, method=handle_missing)

        # 2. Remove/cap outliers
        if cap_outliers:
            processed_df = self._cap_outliers(processed_df)

        # 3. Feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # 4. Encode categorical variables
        processed_df = self._encode_categorical(processed_df)
        
        # 5. Normalize numerical features
        if normalize:
            processed_df = self._normalize_features(processed_df)
        
        # 6. Create target variable
        processed_df = self._create_target_variable(processed_df)
        
        self.logger.info(f"Data preprocessing completed. Shape: {processed_df.shape}")
        return processed_df
    
    def _cap_outliers(self, df):
        """Cap outliers in likes, comments_count, #Followers using 1st/99th percentiles"""
        self.logger.info("Capping outliers...")
        for col in ['likes', 'comments_count', '#Followers']:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = np.clip(df[col], lower, upper)
        return df

    def _handle_missing_values(self, df, method="impute_median"):
        """Handle missing values in the dataset"""
        self.logger.info(f"Handling missing values using method: {method}")
        
        # Drop rows with missing comment_text (critical for sentiment analysis)
        if 'comment_text' in df.columns:
            df = df.dropna(subset=['comment_text'])
        # Impute numerical columns with median
        numerical_cols = ['likes', 'comments_count', 'comment_likes', '#Followers']
        for col in numerical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        # Handle text columns
        if 'comment_text' in df.columns:
            df['comment_text'] = df['comment_text'].fillna("")
        return df
    
    def _engineer_features(self, df):
        """Engineer additional features"""
        self.logger.info("Engineering features...")
        
        # Standardize media_type
        if 'media_type' in df.columns:
            df['media_type'] = df['media_type'].replace({
                'image': 'photo', 'Image': 'photo', 'PHOTO': 'photo', 'Photo': 'photo', 'pic': 'photo', 'picture': 'photo'
            }).str.lower()
        
        # Validate uniqueness of owner_id and post_id
        if 'owner_id' in df.columns and 'post_id' in df.columns:
            df = df.drop_duplicates(subset=['owner_id', 'post_id'])
        
        # Engagement rate
        if 'likes' in df.columns and '#Followers' in df.columns:
            df['engagement_rate'] = df['likes'] / (df['#Followers'] + 1)
        
        # Engagement Frequency: Comments per post for each user
        if 'comment_owner_username' in df.columns:
            user_engagement = df.groupby('comment_owner_username').size()
            df['engagement_frequency'] = df['comment_owner_username'].map(user_engagement)
        
        # Influence Score: Normalize #Followers
        if '#Followers' in df.columns:
            df['influence_score'] = MinMaxScaler().fit_transform(df[['#Followers']])
        
        # Content interaction score
        if all(col in df.columns for col in ['likes', 'comments_count']):
            df['content_interaction'] = df['likes'] + df['comments_count']
        
        # Comment engagement ratio
        if all(col in df.columns for col in ['comment_likes', 'comments_count']):
            df['comment_engagement_ratio'] = df['comment_likes'] / (df['comments_count'] + 1)
        
        # Text features
        if 'comment_text' in df.columns:
            df['comment_length'] = df['comment_text'].str.len()
            df['has_emoji'] = df['comment_text'].str.contains(r'[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\U0001F1E0-\U0001F1FF]', regex=True)
        
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical variables"""
        self.logger.info("Encoding categorical variables...")
        
        # One-hot encode media_type
        if 'media_type' in df.columns:
            media_dummies = pd.get_dummies(df['media_type'], prefix='media')
            df = pd.concat([df, media_dummies], axis=1)
        
        # One-hot encode Category
        if 'Category' in df.columns:
            category_dummies = pd.get_dummies(df['Category'], prefix='category')
            df = pd.concat([df, category_dummies], axis=1)
        
        # Label encode comment_owner_username
        if 'comment_owner_username' in df.columns:
            le = LabelEncoder()
            df['user_id_encoded'] = le.fit_transform(df['comment_owner_username'].astype(str))
            self.label_encoders['comment_owner_username'] = le
        
        return df
    
    def _normalize_features(self, df):
        """Normalize numerical features"""
        self.logger.info("Normalizing numerical features...")
        
        # Features to normalize
        numerical_features = [
            'likes', 'comments_count', 'comment_likes', '#Followers',
            'engagement_frequency', 'content_interaction', 'comment_engagement_ratio',
            'comment_length'
        ]
        
        # Only normalize existing columns
        cols_to_normalize = [col for col in numerical_features if col in df.columns]
        
        if cols_to_normalize:
            df[cols_to_normalize] = self.scaler.fit_transform(df[cols_to_normalize])
        
        return df
    
    def _create_target_variable(self, df):
        """Create target variable for engagement prediction"""
        self.logger.info("Creating target variable...")
        
        # Binary engagement label (1 if user engaged, 0 otherwise)
        if all(col in df.columns for col in ['comment_likes', 'comments_count']):
            # User engaged if they commented or their comment received likes
            df['engagement_binary'] = ((df['comment_likes'] > 0) | 
                                     (df['comments_count'] > 0)).astype(int)
        
        # Engagement probability (normalized engagement score)
        if 'content_interaction' in df.columns:
            df['engagement_probability'] = MinMaxScaler().fit_transform(
                df[['content_interaction']]
            )
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns for ML models"""
        # Exclude non-feature columns
        exclude_cols = [
            'post_id', 'owner_id', 'timestamp', 'comment_text', 
            'comment_owner_username', 'media_type', 'Category',
            'engagement_binary', 'engagement_probability'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def save_preprocessing_info(self, output_path="outputs/preprocessing_info.json"):
        """Save preprocessing information for later use"""
        info = {
            "scaler_params": {
                "feature_range": self.scaler.feature_range,
                "data_min": self.scaler.data_min_.tolist() if hasattr(self.scaler, 'data_min_') else None,
                "data_max": self.scaler.data_max_.tolist() if hasattr(self.scaler, 'data_max_') else None
            },
            "label_encoders": {
                name: list(encoder.classes_) 
                for name, encoder in self.label_encoders.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        self.logger.info(f"Preprocessing info saved to {output_path}")
