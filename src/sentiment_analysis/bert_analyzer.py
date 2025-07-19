"""
BERT Sentiment Analysis Module
Implements BERT-based sentiment analysis for comment text with fine-tuning support
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, pipeline,
    Trainer, TrainingArguments
)
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import logging
import warnings
import os
warnings.filterwarnings('ignore')

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis fine-tuning"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BERTSentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased", learning_rate=2e-5, 
                 batch_size=16, epochs=3, use_fine_tuning=False):
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        
        # Hyperparameters as specified
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_fine_tuning = use_fine_tuning
        
        # Memory management
        self.chunk_size = 5000  # Process large datasets in chunks
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def analyze_sentiment(self, df, model_name=None, 
                         batch_size=None, max_length=128, use_gpu=True):
        """
        Analyze sentiment of comment texts using BERT with enhanced error handling
        
        Args:
            df: Preprocessed Instagram dataset
            model_name: BERT model to use (defaults to instance model_name)
            batch_size: Batch size for processing (defaults to instance batch_size)
            max_length: Maximum sequence length
            use_gpu: Whether to use GPU if available
            
        Returns:
            Dictionary of sentiment scores per comment
        """
        # Use instance defaults if not provided
        model_name = model_name or self.model_name
        batch_size = batch_size or self.batch_size
        
        self.logger.info(f"Starting sentiment analysis with {model_name}")
        self.logger.info(f"Hyperparameters: LR={self.learning_rate}, Batch={batch_size}, Epochs={self.epochs}")
        
        # Validate required columns
        required_cols = ['comment_text', 'comment_owner_username']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
        
        # Handle empty dataset
        if df.empty or df['comment_text'].dropna().empty:
            self.logger.warning("No comment text found in dataset")
            return {}
        
        # Load model and tokenizer
        self._load_model(model_name, use_gpu)
        
        # Extract and validate comment texts
        comments_df = df.dropna(subset=['comment_text']).copy()
        self.logger.info(f"Processing {len(comments_df)} comments with valid comment_text")
        
        # Check for large dataset and use chunked processing if needed
        if len(comments_df) > self.chunk_size:
            self.logger.info(f"Large dataset detected ({len(comments_df)} comments), using chunked processing")
            return self._process_large_dataset(comments_df, batch_size, max_length)
        else:
            return self._process_standard_dataset(comments_df, batch_size, max_length)
    
    def _process_large_dataset(self, df, batch_size, max_length):
        """Process large datasets in chunks to avoid memory issues"""
        all_results = {}
        
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i+self.chunk_size]
            self.logger.info(f"Processing chunk {i//self.chunk_size + 1}/{(len(df)-1)//self.chunk_size + 1}")
            
            try:
                chunk_results = self._process_standard_dataset(chunk, batch_size, max_length)
                all_results.update(chunk_results)
                
                # Clear GPU cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                self.logger.error(f"Error processing chunk {i//self.chunk_size + 1}: {str(e)}")
                continue
        
        return all_results
    
    def _process_standard_dataset(self, df, batch_size, max_length):
        """Process standard-sized dataset"""
        # Extract unique comment texts for efficiency
        unique_comments = df['comment_text'].unique().tolist()
        self.logger.info(f"Analyzing sentiment for {len(unique_comments)} unique comments")
        
        # Process comments in batches
        sentiment_scores = self._process_comments_batch(
            unique_comments, batch_size, max_length
        )
        
        # Map back to original dataframe
        comment_to_sentiment = dict(zip(unique_comments, sentiment_scores))
        
        # Create final sentiment mapping with enhanced output format
        final_sentiment_scores = {}
        for idx, row in df.iterrows():
            if pd.notna(row['comment_text']) and row['comment_text'] in comment_to_sentiment:
                # Create unique key using post_id and comment_owner_username
                post_id = str(row.get('post_id', 'unknown'))
                comment_username = str(row.get('comment_owner_username', f'user_{idx}'))
                comment_key = f"post_{post_id}_comment_{comment_username}_{idx}"
                
                sentiment_data = comment_to_sentiment[row['comment_text']]
                
                # Enhanced output format with metadata
                # Get model name safely (avoid JSON serialization issues)
                try:
                    if hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model, 'name_or_path'):
                        model_name = self.pipeline.model.name_or_path
                    elif hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model.config, 'name_or_path'):
                        model_name = self.pipeline.model.config.name_or_path
                    else:
                        model_name = self.model_name  # Use the specified model name
                except:
                    model_name = self.model_name  # Use the specified model name
                
                final_sentiment_scores[comment_key] = {
                    'post_id': row.get('post_id', 'unknown'),
                    'post_owner_username': row.get('username', 'unknown'),  # Add post owner username
                    'comment_owner_username': row.get('comment_owner_username', 'unknown'),
                    'comment_text': row['comment_text'][:100] + '...' if len(str(row['comment_text'])) > 100 else str(row['comment_text']),
                    'comment_full_text': str(row['comment_text']),  # Keep full text for analysis
                    'sentiment': sentiment_data['sentiment'],
                    'confidence': sentiment_data['confidence'],
                    'positive': sentiment_data.get('positive', 0.0),
                    'negative': sentiment_data.get('negative', 0.0),
                    'neutral': sentiment_data.get('neutral', 0.0),
                    'model_used': model_name,
                    'timestamp': pd.Timestamp.now().isoformat(),
                    # Additional metadata
                    'comment_likes': row.get('comment_likes', 0),
                    'comment_timestamp': row.get('comment_timestamp', 0),
                    'post_likes': row.get('likes', 0),
                    'post_comments_count': row.get('comments_count', 0),
                    'engagement_rate': row.get('engagement_rate', 0.0)
                }
        
        self.logger.info(f"Sentiment analysis completed for {len(final_sentiment_scores)} comments")
        return final_sentiment_scores
    
    def analyze_batch(self, data_input, save_to_file=True, output_file="outputs/sentiment_scores.json"):
        """
        Direct batch analysis of comments using configured BERT model
        
        Args:
            data_input: Either DataFrame with full comment data or list of comment texts
            save_to_file: Whether to save results to JSON file
            output_file: Output file path
            
        Returns:
            Dictionary of sentiment scores per comment
        """
        
        # Handle different input types
        if isinstance(data_input, pd.DataFrame):
            # Full dataframe with all metadata
            df = data_input.copy()
            self.logger.info(f"Starting batch analysis of {len(df):,} comments from DataFrame")
        elif isinstance(data_input, list):
            # List of comment texts - create minimal dataframe
            self.logger.info(f"Starting batch analysis of {len(data_input):,} comment texts")
            df = pd.DataFrame({
                'comment_text': data_input,
                'comment_owner_username': [f'user_{i}' for i in range(len(data_input))],
                'post_id': ['unknown'] * len(data_input)
            })
        else:
            raise ValueError("data_input must be either a pandas DataFrame or list of strings")
        
        self.logger.info(f"Configuration: {self.model_name}, LR={self.learning_rate}, Batch={self.batch_size}, Epochs={self.epochs}")
        
        # Analyze sentiment using the configured model
        results = self.analyze_sentiment(df)
        
        # Create metadata
        metadata = {
            'model_used': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'max_length': 128,
            'total_comments': len(df),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'fine_tuning_enabled': self.use_fine_tuning,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }
        
        # Save results
        if save_to_file:
            final_results = {
                'sentiment_scores': results,
                'metadata': metadata
            }
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            self.logger.info(f"Results saved to {output_file}")
        
        return results
    
    def fine_tune_model(self, train_texts, train_labels, model_name="distilbert-base-uncased", 
                       max_length=128, output_dir="./sentiment_model"):
        """
        Fine-tune BERT model with specified hyperparameters
        
        Args:
            train_texts: List of training texts
            train_labels: List of training labels (0: negative, 1: neutral, 2: positive)
            model_name: Base model to fine-tune
            max_length: Maximum sequence length
            output_dir: Directory to save fine-tuned model
        """
        if not self.use_fine_tuning:
            self.logger.warning("Fine-tuning not enabled. Set use_fine_tuning=True")
            return
            
        self.logger.info(f"Starting fine-tuning of {model_name}")
        self.logger.info(f"Hyperparameters: LR={self.learning_rate}, Batch={self.batch_size}, Epochs={self.epochs}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=3  # negative, neutral, positive
        )
        
        # Create dataset
        dataset = SentimentDataset(train_texts, train_labels, tokenizer, max_length)
        
        # Training arguments with specified hyperparameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",  # No evaluation dataset provided
            save_total_limit=2,
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
            warmup_steps=100,
            weight_decay=0.01,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )
        
        # Train the model
        try:
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            self.logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            
            # Update current model to use fine-tuned version
            self.tokenizer = tokenizer
            self.model = model
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {str(e)}")
            raise
    def _load_model(self, model_name, use_gpu):
        """Load BERT model and tokenizer with enhanced error handling"""
        self.logger.info(f"Loading model: {model_name}")
        
        # Determine device
        device_id = 0 if use_gpu and torch.cuda.is_available() else -1
        
        try:
            # For sentiment analysis, use a pre-trained sentiment model if available
            if 'sentiment' in model_name.lower() or 'twitter' in model_name.lower():
                # Use specialized sentiment model
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=device_id,
                    return_all_scores=True
                )
                self.logger.info(f"Loaded specialized sentiment model: {model_name}")
                
            else:
                # For general BERT models, try to load tokenizer first
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.logger.info(f"Loaded tokenizer for {model_name}")
                    
                    # Use the specified model for classification
                    self.pipeline = pipeline(
                        "sentiment-analysis",
                        model=model_name,
                        device=device_id,
                        return_all_scores=True
                    )
                    self.logger.info(f"Using {model_name} model for classification")
                    
                except Exception as tokenizer_error:
                    self.logger.warning(f"Could not load tokenizer for {model_name}: {tokenizer_error}")
                    # Fall through to default model
                    raise tokenizer_error
            
            # Test the pipeline with a sample text
            test_result = self.pipeline(["This is a test sentence."])
            self.logger.info("Model loaded and tested successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {str(e)}")
            self.logger.info("Falling back to default model...")
            
            # Fallback to a reliable model
            try:
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,  # Use the specified model as fallback too
                    device=-1,  # Use CPU as fallback
                    return_all_scores=True
                )
                self.logger.info(f"Successfully loaded fallback model: {self.model_name}")
            except Exception as fallback_error:
                self.logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError("Could not load any sentiment analysis model")
    
    def _process_comments_batch(self, comments, batch_size, max_length):
        """Process comments in batches with enhanced error handling"""
        sentiment_results = []
        failed_count = 0
        
        # Process in batches
        for i in tqdm(range(0, len(comments), batch_size), desc="Processing sentiment"):
            batch = comments[i:i + batch_size]
            
            # Clean and prepare texts
            cleaned_batch = []
            for text in batch:
                cleaned_text = self._clean_text(text, max_length)
                if cleaned_text:  # Only process non-empty texts
                    cleaned_batch.append(cleaned_text)
                else:
                    # Add default neutral sentiment for empty/invalid text
                    sentiment_results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.5,
                        'positive': 0.33,
                        'negative': 0.33,
                        'neutral': 0.34
                    })
                    failed_count += 1
            
            if not cleaned_batch:
                continue
                
            try:
                # Get sentiment predictions
                batch_results = self.pipeline(cleaned_batch)
                
                # Process results
                for result in batch_results:
                    sentiment_score = self._process_sentiment_result(result)
                    sentiment_results.append(sentiment_score)
                    
            except Exception as e:
                self.logger.warning(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                failed_count += len(cleaned_batch)
                
                # Add default neutral sentiment for failed batch
                for _ in cleaned_batch:
                    sentiment_results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.5,
                        'positive': 0.33,
                        'negative': 0.33,
                        'neutral': 0.34
                    })
                
                # Clear GPU cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if failed_count > 0:
            self.logger.warning(f"Failed to process {failed_count} out of {len(comments)} comments")
        
        return sentiment_results
    
    def _clean_text(self, text, max_length):
        """Clean and preprocess comment text"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Skip very short texts (likely noise)
        if len(text) < 3:
            return ""
        
        # Truncate if too long (leave room for special tokens)
        if len(text) > max_length - 10:
            text = text[:max_length-13] + "..."
        
        return text
    
    def _process_sentiment_result(self, result):
        """Process sentiment analysis result"""
        # Handle different result formats
        if isinstance(result, list):
            # Multi-label result (return_all_scores=True)
            sentiments = {item['label'].lower(): item['score'] for item in result}
            
            # Normalize label names
            label_mapping = {
                'positive': 'positive',
                'pos': 'positive',
                'label_2': 'positive',
                'negative': 'negative', 
                'neg': 'negative',
                'label_0': 'negative',
                'neutral': 'neutral',
                'neu': 'neutral',
                'label_1': 'neutral'
            }
            
            normalized_sentiments = {}
            for label, score in sentiments.items():
                mapped_label = label_mapping.get(label, label)
                normalized_sentiments[mapped_label] = score
            
            # Determine dominant sentiment
            dominant_sentiment = max(normalized_sentiments, key=normalized_sentiments.get)
            confidence = normalized_sentiments[dominant_sentiment]
            
            return {
                'sentiment': dominant_sentiment,
                'confidence': confidence,
                'positive': normalized_sentiments.get('positive', 0.0),
                'negative': normalized_sentiments.get('negative', 0.0),
                'neutral': normalized_sentiments.get('neutral', 0.0)
            }
        
        else:
            # Single result format
            sentiment = result['label'].lower()
            confidence = result['score']
            
            # Map to standard format
            if sentiment in ['positive', 'pos']:
                return {
                    'sentiment': 'positive',
                    'confidence': confidence,
                    'positive': confidence,
                    'negative': (1 - confidence) / 2,
                    'neutral': (1 - confidence) / 2
                }
            elif sentiment in ['negative', 'neg']:
                return {
                    'sentiment': 'negative',
                    'confidence': confidence,
                    'positive': (1 - confidence) / 2,
                    'negative': confidence,
                    'neutral': (1 - confidence) / 2
                }
            else:
                return {
                    'sentiment': 'neutral',
                    'confidence': confidence,
                    'positive': (1 - confidence) / 2,
                    'negative': (1 - confidence) / 2,
                    'neutral': confidence
                }
    
    def aggregate_user_sentiment(self, sentiment_scores, df):
        """Aggregate sentiment scores by user"""
        self.logger.info("Aggregating sentiment scores by user...")
        
        # Create mapping from comment to user
        user_sentiments = {}
        
        for comment_key, sentiment_data in sentiment_scores.items():
            # Extract username from comment key
            try:
                username = comment_key.split('_')[-1]
                
                if username not in user_sentiments:
                    user_sentiments[username] = {
                        'sentiments': [],
                        'confidences': [],
                        'positive_scores': [],
                        'negative_scores': [],
                        'neutral_scores': []
                    }
                
                user_sentiments[username]['sentiments'].append(sentiment_data['sentiment'])
                user_sentiments[username]['confidences'].append(sentiment_data['confidence'])
                user_sentiments[username]['positive_scores'].append(sentiment_data['positive'])
                user_sentiments[username]['negative_scores'].append(sentiment_data['negative'])
                user_sentiments[username]['neutral_scores'].append(sentiment_data['neutral'])
                
            except Exception as e:
                self.logger.warning(f"Error processing comment key {comment_key}: {str(e)}")
        
        # Calculate aggregated metrics
        user_sentiment_profiles = {}
        for username, data in user_sentiments.items():
            if data['sentiments']:
                # Calculate dominant sentiment
                sentiment_counts = pd.Series(data['sentiments']).value_counts()
                dominant_sentiment = sentiment_counts.index[0]
                
                # Calculate average scores
                avg_positive = np.mean(data['positive_scores'])
                avg_negative = np.mean(data['negative_scores'])
                avg_neutral = np.mean(data['neutral_scores'])
                avg_confidence = np.mean(data['confidences'])
                
                # Calculate sentiment diversity (entropy)
                sentiment_probs = sentiment_counts / len(data['sentiments'])
                sentiment_diversity = -sum(p * np.log(p + 1e-8) for p in sentiment_probs)
                
                user_sentiment_profiles[username] = {
                    'dominant_sentiment': dominant_sentiment,
                    'avg_confidence': float(avg_confidence),
                    'avg_positive': float(avg_positive),
                    'avg_negative': float(avg_negative),
                    'avg_neutral': float(avg_neutral),
                    'sentiment_diversity': float(sentiment_diversity),
                    'total_comments': len(data['sentiments']),
                    'sentiment_distribution': sentiment_counts.to_dict()
                }
        
        return user_sentiment_profiles
    
    def save_sentiment_results(self, sentiment_scores, user_profiles=None, 
                              output_path="outputs/sentiment_analysis_results.json"):
        """Save sentiment analysis results with enhanced metadata"""
        
        # Calculate comprehensive statistics
        sentiments = [score['sentiment'] for score in sentiment_scores.values()]
        confidences = [score['confidence'] for score in sentiment_scores.values()]
        
        sentiment_dist = pd.Series(sentiments).value_counts(normalize=True)
        
        # Get model name safely for metadata
        try:
            if hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model, 'name_or_path'):
                model_name = self.pipeline.model.name_or_path
            elif hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model.config, 'name_or_path'):
                model_name = self.pipeline.model.config.name_or_path
            else:
                model_name = self.model_name  # Use the specified model name
        except:
            model_name = self.model_name  # Use the specified model name
        
        results = {
            "sentiment_scores": sentiment_scores,
            "metadata": {
                "total_comments_analyzed": len(sentiment_scores),
                "model_used": model_name,
                "hyperparameters": {
                    "learning_rate": float(self.learning_rate),
                    "batch_size": self.batch_size,
                    "epochs": self.epochs
                },
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "device_used": str(self.device),
                "fine_tuning_enabled": self.use_fine_tuning
            },
            "analysis_summary": {
                "sentiment_distribution": sentiment_dist.to_dict(),
                "average_confidence": float(np.mean(confidences)),
                "confidence_std": float(np.std(confidences)),
                "high_confidence_count": int(sum(1 for c in confidences if c > 0.8)),
                "low_confidence_count": int(sum(1 for c in confidences if c < 0.6))
            }
        }
        
        if user_profiles:
            results["user_sentiment_profiles"] = user_profiles
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Enhanced sentiment analysis results saved to {output_path}")
        return results
    
    def get_sentiment_statistics(self, sentiment_scores):
        """Get overall sentiment statistics"""
        sentiments = [score['sentiment'] for score in sentiment_scores.values()]
        confidences = [score['confidence'] for score in sentiment_scores.values()]
        
        sentiment_dist = pd.Series(sentiments).value_counts(normalize=True)
        
        stats = {
            "sentiment_distribution": sentiment_dist.to_dict(),
            "average_confidence": float(np.mean(confidences)),
            "confidence_std": float(np.std(confidences)),
            "total_comments": len(sentiments)
        }
        
        return stats
