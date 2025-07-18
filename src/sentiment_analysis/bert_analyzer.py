"""
BERT Sentiment Analysis Module
Implements BERT-based sentiment analysis for comment text
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm
import json
import logging
import warnings
warnings.filterwarnings('ignore')

class BERTSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def analyze_sentiment(self, df, model_name="distilbert-base-uncased", 
                         batch_size=16, max_length=128, use_gpu=True):
        """
        Analyze sentiment of comment texts using BERT
        
        Args:
            df: Preprocessed Instagram dataset
            model_name: BERT model to use
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            use_gpu: Whether to use GPU if available
            
        Returns:
            Dictionary of sentiment scores per comment
        """
        self.logger.info(f"Starting sentiment analysis with {model_name}")
        
        # Load model and tokenizer
        self._load_model(model_name, use_gpu)
        
        # Extract comment texts
        comments = df['comment_text'].dropna().unique().tolist()
        self.logger.info(f"Analyzing sentiment for {len(comments)} unique comments")
        
        # Process comments in batches
        sentiment_scores = self._process_comments_batch(
            comments, batch_size, max_length
        )
        
        # Map back to original dataframe
        comment_to_sentiment = dict(zip(comments, sentiment_scores))
        
        # Create final sentiment mapping
        final_sentiment_scores = {}
        for idx, row in df.iterrows():
            if pd.notna(row['comment_text']) and row['comment_text'] in comment_to_sentiment:
                comment_key = f"comment_{idx}_{row['comment_owner_username']}"
                final_sentiment_scores[comment_key] = comment_to_sentiment[row['comment_text']]
        
        self.logger.info(f"Sentiment analysis completed for {len(final_sentiment_scores)} comments")
        return final_sentiment_scores
    
    def _load_model(self, model_name, use_gpu):
        """Load BERT model and tokenizer"""
        self.logger.info(f"Loading model: {model_name}")
        
        try:
            # For sentiment analysis, use a pre-trained sentiment model if available
            if 'sentiment' in model_name.lower() or 'twitter' in model_name.lower():
                # Use specialized sentiment model
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    device=0 if use_gpu and torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
            else:
                # Use general BERT model with sentiment classification head
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # For general BERT models, we'll use a simple classification approach
                # This is a simplified version - in practice, you might want to fine-tune
                self.pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if use_gpu and torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            # Fallback to a simple model
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=-1,  # Use CPU as fallback
                return_all_scores=True
            )
    
    def _process_comments_batch(self, comments, batch_size, max_length):
        """Process comments in batches"""
        sentiment_results = []
        
        # Process in batches
        for i in tqdm(range(0, len(comments), batch_size), desc="Processing sentiment"):
            batch = comments[i:i + batch_size]
            
            # Clean and prepare texts
            cleaned_batch = [self._clean_text(text, max_length) for text in batch]
            
            try:
                # Get sentiment predictions
                batch_results = self.pipeline(cleaned_batch)
                
                # Process results
                for result in batch_results:
                    sentiment_score = self._process_sentiment_result(result)
                    sentiment_results.append(sentiment_score)
                    
            except Exception as e:
                self.logger.warning(f"Error processing batch: {str(e)}")
                # Add default neutral sentiment for failed batch
                for _ in batch:
                    sentiment_results.append({
                        'sentiment': 'neutral',
                        'confidence': 0.5,
                        'positive': 0.33,
                        'negative': 0.33,
                        'neutral': 0.34
                    })
        
        return sentiment_results
    
    def _clean_text(self, text, max_length):
        """Clean and truncate text"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length-3] + "..."
        
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
        """Save sentiment analysis results"""
        results = {
            "comment_sentiment_scores": sentiment_scores,
            "analysis_summary": {
                "total_comments_analyzed": len(sentiment_scores),
                "model_used": getattr(self.pipeline, 'model', 'unknown'),
                "avg_confidence": np.mean([score['confidence'] for score in sentiment_scores.values()])
            }
        }
        
        if user_profiles:
            results["user_sentiment_profiles"] = user_profiles
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Sentiment analysis results saved to {output_path}")
    
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
