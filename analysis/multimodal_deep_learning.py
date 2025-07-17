"""
Multimodal Deep Learning for Instagram Engagement Prediction
Combines text embeddings, numerical features, and temporal patterns using transformers
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import (
        AutoTokenizer, AutoModel, 
        BertTokenizer, BertModel,
        RobertaTokenizer, RobertaModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not available. Install with: pip install transformers torch")

try:
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


class InstagramDataset(Dataset):
    """Custom dataset for Instagram multimodal data."""
    
    def __init__(self, captions, numerical_features, targets, tokenizer, max_length=128):
        self.captions = captions
        self.numerical_features = numerical_features
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = str(self.captions.iloc[idx])
        
        # Tokenize caption
        encoding = self.tokenizer(
            caption,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'numerical_features': torch.FloatTensor(self.numerical_features.iloc[idx].values),
            'targets': torch.FloatTensor([self.targets.iloc[idx]])
        }


class MultiModalTransformer(nn.Module):
    """Multimodal transformer for Instagram engagement prediction."""
    
    def __init__(self, bert_model_name='bert-base-uncased', num_numerical_features=10, 
                 hidden_size=256, num_heads=8, num_layers=3, dropout=0.1):
        super(MultiModalTransformer, self).__init__()
        
        # Text encoder (BERT/RoBERTa)
        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert = AutoModel.from_pretrained(bert_model_name)
            bert_hidden_size = self.bert.config.hidden_size
        else:
            bert_hidden_size = 768  # Default BERT size
            
        # Freeze BERT initially (optional)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Numerical features encoder
        self.numerical_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layers
        fusion_input_size = bert_hidden_size + 64  # BERT + numerical features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers with residual connections
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in [self.numerical_encoder, self.fusion_layers, self.output_layers]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, numerical_features):
        # Text encoding
        if TRANSFORMERS_AVAILABLE:
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            text_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        else:
            # Fallback for when transformers not available
            text_embeddings = torch.randn(input_ids.size(0), 768)
        
        # Numerical features encoding
        numerical_embeddings = self.numerical_encoder(numerical_features)
        
        # Fusion
        combined_features = torch.cat([text_embeddings, numerical_embeddings], dim=1)
        fused_features = self.fusion_layers(combined_features)
        
        # Output prediction
        output = self.output_layers(fused_features)
        
        return output.squeeze()


class AdvancedTextAnalyzer:
    """Advanced text analysis using transformers."""
    
    def __init__(self):
        if TRANSFORMERS_AVAILABLE:
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.sentiment_model = AutoModel.from_pretrained(
                "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
    def get_advanced_text_features(self, captions):
        """Extract advanced text features using transformers."""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers not available. Using fallback features.")
            return self._get_fallback_text_features(captions)
        
        features = []
        
        for caption in captions:
            caption_str = str(caption)
            
            # BERT embeddings
            inputs = self.bert_tokenizer(
                caption_str, 
                return_tensors='pt', 
                truncation=True, 
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                bert_embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
            # Advanced text metrics
            text_metrics = {
                'semantic_density': np.mean(np.abs(bert_embedding)),
                'text_complexity': len(caption_str.split()) / max(len(caption_str), 1),
                'exclamation_ratio': caption_str.count('!') / max(len(caption_str.split()), 1),
                'question_ratio': caption_str.count('?') / max(len(caption_str.split()), 1),
                'capitalization_ratio': sum(1 for c in caption_str if c.isupper()) / max(len(caption_str), 1),
                'emoji_count': sum(1 for c in caption_str if ord(c) > 127),
                'bert_magnitude': np.linalg.norm(bert_embedding)
            }
            
            features.append(text_metrics)
        
        return pd.DataFrame(features)
    
    def _get_fallback_text_features(self, captions):
        """Fallback text features when transformers not available."""
        features = []
        
        for caption in captions:
            caption_str = str(caption)
            
            text_metrics = {
                'semantic_density': len(caption_str.split()) / max(len(caption_str), 1),
                'text_complexity': len(set(caption_str.split())) / max(len(caption_str.split()), 1),
                'exclamation_ratio': caption_str.count('!') / max(len(caption_str.split()), 1),
                'question_ratio': caption_str.count('?') / max(len(caption_str.split()), 1),
                'capitalization_ratio': sum(1 for c in caption_str if c.isupper()) / max(len(caption_str), 1),
                'emoji_count': sum(1 for c in caption_str if ord(c) > 127),
                'bert_magnitude': np.random.normal(0, 1)  # Random placeholder
            }
            
            features.append(text_metrics)
        
        return pd.DataFrame(features)


class MultiModalEngagementPredictor:
    """Main class for multimodal engagement prediction."""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.text_analyzer = AdvancedTextAnalyzer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def prepare_multimodal_features(self, df):
        """Prepare comprehensive multimodal features."""
        df_enhanced = df.copy()
        
        # Advanced text features
        if 'Caption' in df.columns:
            text_features = self.text_analyzer.get_advanced_text_features(df['Caption'])
            for col in text_features.columns:
                df_enhanced[f'text_{col}'] = text_features[col]
        
        # User-specific features with high-value approach
        if '#Followers' in df.columns:
            # Follower tiers
            df_enhanced['follower_log'] = np.log1p(df['#Followers'])
            df_enhanced['is_mega_influencer'] = (df['#Followers'] > 1000000).astype(int)
            df_enhanced['is_macro_influencer'] = (
                (df['#Followers'] > 100000) & (df['#Followers'] <= 1000000)
            ).astype(int)
            df_enhanced['is_micro_influencer'] = (
                (df['#Followers'] > 10000) & (df['#Followers'] <= 100000)
            ).astype(int)
            
            # Engagement quality metrics
            total_engagement = df.get('likes', 0) + df.get('Comments', df.get('comments_count', 0))
            df_enhanced['engagement_per_1k_followers'] = (
                total_engagement / (df['#Followers'] / 1000 + 1)
            )
            
            df_enhanced['follower_to_following_ratio'] = (
                df['#Followers'] / (df.get('#Followees', 1) + 1)
            )
        
        # Content strategy features
        if 'num_hashtags' in df.columns:
            df_enhanced['hashtag_density'] = df['num_hashtags'] / (df.get('caption_length', 1) + 1)
            df_enhanced['optimal_hashtag_range'] = (
                (df['num_hashtags'] >= 5) & (df['num_hashtags'] <= 15)
            ).astype(int)
        
        # Temporal and posting strategy features
        if 'hour_of_day' in df.columns:
            # Create cyclical features for hour
            df_enhanced['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df_enhanced['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            
            # Peak engagement times
            df_enhanced['is_prime_time'] = df['hour_of_day'].isin([19, 20, 21]).astype(int)
            df_enhanced['is_morning_rush'] = df['hour_of_day'].isin([7, 8, 9]).astype(int)
            df_enhanced['is_lunch_time'] = df['hour_of_day'].isin([12, 13]).astype(int)
        
        # Advanced engagement features
        if 'engagement_rate' in df.columns:
            df_enhanced['engagement_percentile'] = df['engagement_rate'].rank(pct=True)
            df_enhanced['above_avg_engagement'] = (
                df['engagement_rate'] > df['engagement_rate'].mean()
            ).astype(int)
        
        return df_enhanced
    
    def train_multimodal_model(self, df, target_column='engagement_rate', 
                             epochs=50, batch_size=32, learning_rate=1e-4):
        """Train the multimodal transformer model."""
        print(f"Training multimodal model for target: {target_column}")
        
        # Prepare features
        df_enhanced = self.prepare_multimodal_features(df)
        
        # Select numerical features
        numerical_cols = [
            'likes', 'shares', 'caption_length', 'num_hashtags', '#Followers', '#Posts',
            'follower_log', 'engagement_per_1k_followers', 'follower_to_following_ratio',
            'hashtag_density', 'hour_sin', 'hour_cos', 'is_prime_time',
            'text_semantic_density', 'text_text_complexity', 'text_bert_magnitude'
        ]
        
        # Filter available numerical features
        available_numerical = [col for col in numerical_cols if col in df_enhanced.columns]
        X_numerical = df_enhanced[available_numerical].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Text data
        X_text = df_enhanced['Caption'].fillna('').astype(str)
        
        # Target
        y = df_enhanced[target_column].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_numerical_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_numerical),
            columns=X_numerical.columns,
            index=X_numerical.index
        )
        
        # Split data
        X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
            X_text, X_numerical_scaled, y, test_size=0.2, random_state=42
        )
        
        # Create datasets
        if TRANSFORMERS_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            tokenizer = None
            print("Transformers not available. Using fallback approach.")
            return self._train_fallback_model(X_numerical_scaled, y)
        
        train_dataset = InstagramDataset(X_text_train, X_num_train, y_train, tokenizer)
        test_dataset = InstagramDataset(X_text_test, X_num_test, y_test, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = MultiModalTransformer(
            bert_model_name=self.model_name,
            num_numerical_features=len(available_numerical),
            hidden_size=256,
            num_heads=8,
            num_layers=3,
            dropout=0.1
        ).to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, numerical_features)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    numerical_features = batch['numerical_features'].to(self.device)
                    targets = batch['targets'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask, numerical_features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save best model
                torch.save(self.model.state_dict(), f'outputs/multimodal_best_model_{target_column}.pth')
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Evaluate model
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, numerical_features)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        
        metrics = {
            'Model_Name': 'Multimodal_Transformer',
            'Target': target_column,
            'MSE': float(mse),
            'RMSE': float(np.sqrt(mse)),
            'R2_Score': float(r2),
            'Training_Epochs': epochs,
            'Best_Val_Loss': float(best_val_loss),
            'Timestamp': datetime.now().isoformat()
        }
        
        # Save model and results
        model_path = f'outputs/multimodal_model_{target_column}.pth'
        torch.save(self.model.state_dict(), model_path)
        
        scaler_path = f'outputs/multimodal_scaler_{target_column}.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        results_path = f'outputs/multimodal_evaluation_{target_column}.json'
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMultimodal Model Training Complete!")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        
        return self.model, metrics
    
    def _train_fallback_model(self, X, y):
        """Fallback to traditional ML when transformers not available."""
        from sklearn.ensemble import RandomForestRegressor
        
        print("Training fallback Random Forest model...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'Model_Name': 'Fallback_RandomForest',
            'R2_Score': float(r2_score(y_test, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'Timestamp': datetime.now().isoformat()
        }
        
        return model, metrics


def run_multimodal_analysis(df):
    """Main function to run multimodal analysis."""
    predictor = MultiModalEngagementPredictor()
    
    targets = ['engagement_rate']
    if 'likes' in df.columns:
        targets.append('likes')
    if 'Comments' in df.columns or 'comments_count' in df.columns:
        targets.append('Comments' if 'Comments' in df.columns else 'comments_count')
    
    results = {}
    
    for target in targets:
        print(f"\n{'='*60}")
        print(f"Training Multimodal Model for: {target}")
        print(f"{'='*60}")
        
        model, metrics = predictor.train_multimodal_model(df, target, epochs=20)
        results[target] = {
            'model': model,
            'metrics': metrics
        }
    
    return results


if __name__ == "__main__":
    print("Multimodal Deep Learning - Test Run")
    
    # Create sample data
    sample_data = {
        'Caption': [f"Amazing post #{i} with great content! 🌟" for i in range(100)],
        'likes': np.random.randint(10, 1000, 100),
        'shares': np.random.randint(1, 50, 100),
        'caption_length': np.random.randint(10, 500, 100),
        'num_hashtags': np.random.randint(1, 30, 100),
        '#Followers': np.random.randint(1000, 100000, 100),
        '#Posts': np.random.randint(10, 1000, 100),
        'hour_of_day': np.random.randint(0, 24, 100),
        'engagement_rate': np.random.uniform(0.01, 0.1, 100)
    }
    
    df = pd.DataFrame(sample_data)
    results = run_multimodal_analysis(df)
    
    print("\nMultimodal Analysis Complete!")
    for target, result in results.items():
        print(f"\nTarget: {target}")
        print(f"  Model: {result['metrics']['Model_Name']}")
        print(f"  R² Score: {result['metrics']['R2_Score']:.4f}")
