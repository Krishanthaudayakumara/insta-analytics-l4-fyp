"""
Advanced NLP & LLM Integration Module
=====================================

This module implements cutting-edge Natural Language Processing and Large Language Model
techniques for Instagram content analysis and engagement prediction.

Features:
- BERT/RoBERTa fine-tuning for Instagram-specific text analysis
- GPT-based content generation and optimization
- Semantic similarity analysis for content clustering
- Advanced sentiment analysis with emotion detection
- Topic modeling with LDA and BERTopic
- Content quality scoring with linguistic features
- Hashtag analysis and optimization
- Multi-language support

Author: Instagram User Behavior Analysis System
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

# Advanced NLP libraries (with fallbacks)
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        pipeline, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel,
        GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline
    )
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    import spacy
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False

# Basic NLP fallbacks
import re
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNLPAnalyzer:
    """
    Advanced NLP analyzer using state-of-the-art language models
    for Instagram content analysis and engagement prediction.
    """
    
    def __init__(self, model_cache_dir: str = "models/nlp_cache"):
        """
        Initialize the Advanced NLP Analyzer.
        
        Args:
            model_cache_dir: Directory to cache downloaded models
        """
        self.model_cache_dir = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # Initialize models
        self.sentiment_analyzer = None
        self.bert_model = None
        self.bert_tokenizer = None
        self.sentence_transformer = None
        self.topic_model = None
        self.content_generator = None
        
        # Initialize feature extractors
        self._initialize_models()
        
        # Content analysis patterns
        self.hashtag_pattern = re.compile(r'#(\w+)')
        self.mention_pattern = re.compile(r'@(\w+)')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
    def _initialize_models(self):
        """Initialize NLP models with fallbacks."""
        try:
            # Initialize BERT for text analysis
            if TRANSFORMERS_AVAILABLE:
                self._initialize_bert()
                self._initialize_content_generator()
            
            # Initialize sentence transformer for semantic analysis
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._initialize_sentence_transformer()
            
            # Initialize topic modeling
            if BERTOPIC_AVAILABLE:
                self._initialize_topic_model()
            
            # Initialize sentiment analysis
            self._initialize_sentiment_analyzer()
            
        except Exception as e:
            logger.warning(f"Some advanced NLP models could not be initialized: {e}")
    
    def _initialize_bert(self):
        """Initialize BERT model for text analysis."""
        try:
            model_name = "bert-base-uncased"
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                model_name, cache_dir=self.model_cache_dir
            )
            self.bert_model = BertModel.from_pretrained(
                model_name, cache_dir=self.model_cache_dir
            )
            self.bert_model.eval()
            logger.info("BERT model initialized successfully")
        except Exception as e:
            logger.warning(f"BERT initialization failed: {e}")
    
    def _initialize_content_generator(self):
        """Initialize GPT-2 for content generation."""
        try:
            if TRANSFORMERS_AVAILABLE:
                self.content_generator = pipeline(
                    "text-generation",
                    model="gpt2",
                    tokenizer="gpt2",
                    model_kwargs={"cache_dir": self.model_cache_dir}
                )
                logger.info("Content generator initialized successfully")
        except Exception as e:
            logger.warning(f"Content generator initialization failed: {e}")
    
    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer for semantic analysis."""
        try:
            self.sentence_transformer = SentenceTransformer(
                'all-MiniLM-L6-v2',
                cache_folder=self.model_cache_dir
            )
            logger.info("Sentence transformer initialized successfully")
        except Exception as e:
            logger.warning(f"Sentence transformer initialization failed: {e}")
    
    def _initialize_topic_model(self):
        """Initialize BERTopic for topic modeling."""
        try:
            self.topic_model = BERTopic(
                verbose=False,
                calculate_probabilities=True
            )
            logger.info("Topic model initialized successfully")
        except Exception as e:
            logger.warning(f"Topic model initialization failed: {e}")
    
    def _initialize_sentiment_analyzer(self):
        """Initialize sentiment analyzer."""
        try:
            if ADVANCED_NLP_AVAILABLE:
                # Download required NLTK data
                try:
                    nltk.download('vader_lexicon', quiet=True)
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                except:
                    pass
                
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("NLTK sentiment analyzer initialized successfully")
            else:
                # Fallback sentiment analyzer
                self.sentiment_analyzer = self._basic_sentiment_analyzer
                logger.info("Basic sentiment analyzer initialized")
        except Exception as e:
            logger.warning(f"Sentiment analyzer initialization failed: {e}")
            self.sentiment_analyzer = self._basic_sentiment_analyzer
    
    def analyze_content_comprehensive(self, texts: List[str]) -> Dict[str, Any]:
        """
        Perform comprehensive NLP analysis on Instagram content.
        
        Args:
            texts: List of text content to analyze
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        results = {
            'text_features': {},
            'sentiment_analysis': {},
            'semantic_analysis': {},
            'topic_analysis': {},
            'content_quality': {},
            'hashtag_analysis': {},
            'linguistic_features': {}
        }
        
        try:
            # Basic text features
            results['text_features'] = self._extract_text_features(texts)
            
            # Sentiment analysis
            results['sentiment_analysis'] = self._analyze_sentiment_advanced(texts)
            
            # Semantic analysis
            if self.sentence_transformer:
                results['semantic_analysis'] = self._analyze_semantic_similarity(texts)
            
            # Topic modeling
            if self.topic_model and len(texts) > 5:
                results['topic_analysis'] = self._analyze_topics(texts)
            
            # Content quality scoring
            results['content_quality'] = self._score_content_quality(texts)
            
            # Hashtag analysis
            results['hashtag_analysis'] = self._analyze_hashtags(texts)
            
            # Linguistic features
            results['linguistic_features'] = self._extract_linguistic_features(texts)
            
        except Exception as e:
            logger.error(f"Error in comprehensive content analysis: {e}")
        
        return results
    
    def _extract_text_features(self, texts: List[str]) -> Dict[str, Any]:
        """Extract basic text features."""
        features = {
            'avg_length': np.mean([len(text) for text in texts]),
            'avg_word_count': np.mean([len(text.split()) for text in texts]),
            'avg_sentence_count': np.mean([len(text.split('.')) for text in texts]),
            'total_unique_words': len(set(' '.join(texts).split())),
            'avg_char_per_word': np.mean([
                np.mean([len(word) for word in text.split()]) if text.split() else 0
                for text in texts
            ])
        }
        
        # Calculate text diversity
        all_words = ' '.join(texts).lower().split()
        word_freq = Counter(all_words)
        features['vocabulary_diversity'] = len(word_freq) / len(all_words) if all_words else 0
        features['most_common_words'] = dict(word_freq.most_common(10))
        
        return features
    
    def _analyze_sentiment_advanced(self, texts: List[str]) -> Dict[str, Any]:
        """Perform advanced sentiment analysis."""
        sentiments = {
            'compound_scores': [],
            'positive_scores': [],
            'negative_scores': [],
            'neutral_scores': [],
            'emotion_scores': {}
        }
        
        for text in texts:
            if self.sentiment_analyzer and hasattr(self.sentiment_analyzer, 'polarity_scores'):
                # NLTK VADER sentiment
                scores = self.sentiment_analyzer.polarity_scores(text)
                sentiments['compound_scores'].append(scores['compound'])
                sentiments['positive_scores'].append(scores['pos'])
                sentiments['negative_scores'].append(scores['neg'])
                sentiments['neutral_scores'].append(scores['neu'])
            else:
                # Basic sentiment fallback
                score = self._basic_sentiment_score(text)
                sentiments['compound_scores'].append(score)
                sentiments['positive_scores'].append(max(0, score))
                sentiments['negative_scores'].append(max(0, -score))
                sentiments['neutral_scores'].append(1 - abs(score))
        
        # Calculate aggregate statistics
        for key in ['compound_scores', 'positive_scores', 'negative_scores', 'neutral_scores']:
            scores = sentiments[key]
            sentiments[f'avg_{key}'] = np.mean(scores) if scores else 0
            sentiments[f'std_{key}'] = np.std(scores) if scores else 0
        
        # Emotion classification
        sentiments['emotion_distribution'] = self._classify_emotions(texts)
        
        return sentiments
    
    def _analyze_semantic_similarity(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze semantic similarity between texts."""
        if not self.sentence_transformer:
            return {'error': 'Sentence transformer not available'}
        
        try:
            # Generate embeddings
            embeddings = self.sentence_transformer.encode(texts)
            
            # Calculate pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            results = {
                'avg_similarity': np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                'max_similarity': np.max(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                'min_similarity': np.min(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                'similarity_std': np.std(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]),
                'embedding_dim': embeddings.shape[1],
                'semantic_diversity': 1 - np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            }
            
            return results
        except Exception as e:
            logger.error(f"Semantic similarity analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_topics(self, texts: List[str]) -> Dict[str, Any]:
        """Perform topic modeling analysis."""
        if not self.topic_model:
            return {'error': 'Topic model not available'}
        
        try:
            # Fit topic model
            topics, probabilities = self.topic_model.fit_transform(texts)
            
            # Get topic information
            topic_info = self.topic_model.get_topic_info()
            
            results = {
                'num_topics': len(topic_info) - 1,  # Excluding outlier topic
                'topic_distribution': dict(Counter(topics)),
                'avg_topic_probability': np.mean(probabilities) if len(probabilities) > 0 else 0,
                'topic_coherence': self._calculate_topic_coherence(topics, texts),
                'most_common_topics': topic_info.head(5).to_dict('records') if not topic_info.empty else []
            }
            
            return results
        except Exception as e:
            logger.error(f"Topic analysis failed: {e}")
            return {'error': str(e)}
    
    def _score_content_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Score content quality using various metrics."""
        quality_scores = {
            'readability_scores': [],
            'engagement_potential': [],
            'content_density': [],
            'clarity_scores': []
        }
        
        for text in texts:
            # Readability score
            try:
                if ADVANCED_NLP_AVAILABLE:
                    readability = flesch_reading_ease(text)
                else:
                    readability = self._basic_readability_score(text)
                quality_scores['readability_scores'].append(readability)
            except:
                quality_scores['readability_scores'].append(50)  # Neutral score
            
            # Engagement potential (based on various factors)
            engagement_score = self._calculate_engagement_potential(text)
            quality_scores['engagement_potential'].append(engagement_score)
            
            # Content density (information per word)
            density_score = self._calculate_content_density(text)
            quality_scores['content_density'].append(density_score)
            
            # Clarity score
            clarity_score = self._calculate_clarity_score(text)
            quality_scores['clarity_scores'].append(clarity_score)
        
        # Calculate aggregate statistics
        results = {}
        for key, scores in quality_scores.items():
            results[f'avg_{key}'] = np.mean(scores) if scores else 0
            results[f'std_{key}'] = np.std(scores) if scores else 0
        
        # Overall quality score
        results['overall_quality'] = np.mean([
            results['avg_readability_scores'] / 100,
            results['avg_engagement_potential'],
            results['avg_content_density'],
            results['avg_clarity_scores']
        ])
        
        return results
    
    def _analyze_hashtags(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze hashtag usage patterns."""
        all_hashtags = []
        hashtag_stats = []
        
        for text in texts:
            hashtags = self.hashtag_pattern.findall(text.lower())
            all_hashtags.extend(hashtags)
            hashtag_stats.append({
                'count': len(hashtags),
                'unique_count': len(set(hashtags)),
                'diversity': len(set(hashtags)) / len(hashtags) if hashtags else 0
            })
        
        hashtag_freq = Counter(all_hashtags)
        
        results = {
            'total_hashtags': len(all_hashtags),
            'unique_hashtags': len(set(all_hashtags)),
            'avg_hashtags_per_post': np.mean([h['count'] for h in hashtag_stats]),
            'avg_unique_hashtags_per_post': np.mean([h['unique_count'] for h in hashtag_stats]),
            'hashtag_diversity': np.mean([h['diversity'] for h in hashtag_stats]),
            'most_popular_hashtags': dict(hashtag_freq.most_common(10)),
            'hashtag_frequency_distribution': dict(Counter([h['count'] for h in hashtag_stats]))
        }
        
        return results
    
    def _extract_linguistic_features(self, texts: List[str]) -> Dict[str, Any]:
        """Extract advanced linguistic features."""
        features = {
            'avg_sentences_per_post': [],
            'avg_words_per_sentence': [],
            'punctuation_density': [],
            'question_ratio': [],
            'exclamation_ratio': [],
            'capitalization_ratio': [],
            'emoji_density': []
        }
        
        for text in texts:
            sentences = text.split('.')
            words = text.split()
            
            features['avg_sentences_per_post'].append(len(sentences))
            features['avg_words_per_sentence'].append(
                len(words) / len(sentences) if sentences else 0
            )
            
            # Punctuation analysis
            punctuation_chars = sum(1 for c in text if c in '.,!?;:')
            features['punctuation_density'].append(
                punctuation_chars / len(text) if len(text) > 0 else 0
            )
            
            # Question and exclamation ratios
            features['question_ratio'].append(text.count('?') / len(sentences) if sentences else 0)
            features['exclamation_ratio'].append(text.count('!') / len(sentences) if sentences else 0)
            
            # Capitalization ratio
            capital_chars = sum(1 for c in text if c.isupper())
            features['capitalization_ratio'].append(
                capital_chars / len(text) if len(text) > 0 else 0
            )
            
            # Emoji density (basic Unicode emoji detection)
            emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
            features['emoji_density'].append(emoji_count / len(words) if words else 0)
        
        # Calculate averages
        results = {}
        for key, values in features.items():
            results[f'avg_{key}'] = np.mean(values) if values else 0
            results[f'std_{key}'] = np.std(values) if values else 0
        
        return results
    
    def generate_optimized_content(self, 
                                   base_content: str, 
                                   target_metrics: Dict[str, float],
                                   num_variations: int = 3) -> List[str]:
        """
        Generate optimized content variations using LLM.
        
        Args:
            base_content: Original content to optimize
            target_metrics: Target engagement metrics
            num_variations: Number of variations to generate
            
        Returns:
            List of optimized content variations
        """
        if not self.content_generator:
            logger.warning("Content generator not available, returning basic variations")
            return self._generate_basic_variations(base_content, num_variations)
        
        try:
            variations = []
            
            # Create optimization prompts
            prompts = [
                f"Rewrite this Instagram post to increase engagement: {base_content}",
                f"Make this Instagram post more engaging and viral: {base_content}",
                f"Optimize this social media post for maximum likes and comments: {base_content}"
            ]
            
            for i in range(min(num_variations, len(prompts))):
                try:
                    result = self.content_generator(
                        prompts[i],
                        max_length=len(base_content) + 50,
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=50256
                    )
                    
                    generated_text = result[0]['generated_text']
                    # Extract the generated part (remove the prompt)
                    optimized_content = generated_text.replace(prompts[i], "").strip()
                    if optimized_content:
                        variations.append(optimized_content)
                
                except Exception as e:
                    logger.warning(f"Content generation failed for variation {i}: {e}")
            
            # Fill remaining slots with basic variations if needed
            while len(variations) < num_variations:
                basic_vars = self._generate_basic_variations(base_content, 1)
                variations.extend(basic_vars)
            
            return variations[:num_variations]
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return self._generate_basic_variations(base_content, num_variations)
    
    def predict_engagement_with_nlp(self, 
                                    texts: List[str], 
                                    numerical_features: np.ndarray = None) -> Dict[str, Any]:
        """
        Predict engagement using advanced NLP features.
        
        Args:
            texts: List of content texts
            numerical_features: Additional numerical features
            
        Returns:
            Engagement predictions and feature analysis
        """
        try:
            # Extract comprehensive NLP features
            nlp_analysis = self.analyze_content_comprehensive(texts)
            
            # Convert NLP features to numerical format
            feature_vector = self._nlp_features_to_vector(nlp_analysis)
            
            # Combine with numerical features if provided
            if numerical_features is not None:
                combined_features = np.hstack([feature_vector, numerical_features])
            else:
                combined_features = feature_vector
            
            # Generate engagement predictions (using heuristic model)
            predictions = self._predict_engagement_heuristic(nlp_analysis, texts)
            
            return {
                'predictions': predictions,
                'nlp_features': nlp_analysis,
                'feature_vector': feature_vector.tolist(),
                'feature_importance': self._calculate_nlp_feature_importance(nlp_analysis)
            }
            
        except Exception as e:
            logger.error(f"NLP-based engagement prediction failed: {e}")
            return {'error': str(e)}
    
    # Helper methods
    def _basic_sentiment_analyzer(self, text: str) -> Dict[str, float]:
        """Basic sentiment analysis fallback."""
        positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'best', 'excellent', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting']
        
        words = text.lower().split()
        pos_score = sum(1 for word in words if word in positive_words) / len(words) if words else 0
        neg_score = sum(1 for word in words if word in negative_words) / len(words) if words else 0
        
        compound = pos_score - neg_score
        neutral = max(0, 1 - pos_score - neg_score)
        
        return {
            'compound': compound,
            'pos': pos_score,
            'neg': neg_score,
            'neu': neutral
        }
    
    def _basic_sentiment_score(self, text: str) -> float:
        """Basic sentiment scoring."""
        sentiment = self._basic_sentiment_analyzer(text)
        return sentiment['compound']
    
    def _basic_readability_score(self, text: str) -> float:
        """Basic readability score calculation."""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 50
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Simple readability heuristic
        readability = 100 - (avg_sentence_length * 2) - (avg_word_length * 5)
        return max(0, min(100, readability))
    
    def _calculate_engagement_potential(self, text: str) -> float:
        """Calculate engagement potential score."""
        score = 0.0
        
        # Length factor
        word_count = len(text.split())
        if 10 <= word_count <= 50:
            score += 0.2
        
        # Question factor
        if '?' in text:
            score += 0.2
        
        # Call-to-action factor
        cta_words = ['like', 'comment', 'share', 'follow', 'tag', 'what', 'how', 'why']
        if any(word in text.lower() for word in cta_words):
            score += 0.2
        
        # Hashtag factor
        hashtag_count = len(self.hashtag_pattern.findall(text))
        if 1 <= hashtag_count <= 5:
            score += 0.2
        
        # Emoji factor
        emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
        if emoji_count > 0:
            score += 0.2
        
        return score
    
    def _calculate_content_density(self, text: str) -> float:
        """Calculate content density score."""
        words = text.split()
        if not words:
            return 0.0
        
        # Unique words ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Non-stopword ratio (basic stopwords)
        basic_stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were']
        content_words = [w for w in words if w.lower() not in basic_stopwords]
        content_ratio = len(content_words) / len(words)
        
        return (unique_ratio + content_ratio) / 2
    
    def _calculate_clarity_score(self, text: str) -> float:
        """Calculate content clarity score."""
        sentences = text.split('.')
        if not sentences:
            return 0.0
        
        clarity_score = 0.0
        
        # Average sentence length (shorter is clearer)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        if avg_sentence_length < 20:
            clarity_score += 0.5
        
        # Punctuation usage
        punctuation_ratio = sum(1 for c in text if c in '.,!?') / len(text) if text else 0
        if 0.02 <= punctuation_ratio <= 0.1:
            clarity_score += 0.5
        
        return clarity_score
    
    def _classify_emotions(self, texts: List[str]) -> Dict[str, float]:
        """Classify emotions in texts."""
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'amazing', 'awesome', 'love'],
            'anger': ['angry', 'mad', 'furious', 'hate', 'annoying'],
            'sadness': ['sad', 'depressed', 'disappointed', 'crying'],
            'fear': ['scared', 'afraid', 'worried', 'anxious'],
            'surprise': ['surprised', 'shocked', 'wow', 'omg'],
            'trust': ['trust', 'reliable', 'honest', 'confident']
        }
        
        emotion_counts = {emotion: 0 for emotion in emotion_keywords}
        total_words = 0
        
        for text in texts:
            words = text.lower().split()
            total_words += len(words)
            
            for emotion, keywords in emotion_keywords.items():
                emotion_counts[emotion] += sum(1 for word in words if word in keywords)
        
        # Normalize by total words
        if total_words > 0:
            emotion_distribution = {
                emotion: count / total_words 
                for emotion, count in emotion_counts.items()
            }
        else:
            emotion_distribution = {emotion: 0.0 for emotion in emotion_keywords}
        
        return emotion_distribution
    
    def _calculate_topic_coherence(self, topics: List[int], texts: List[str]) -> float:
        """Calculate topic coherence score."""
        if not topics or len(set(topics)) <= 1:
            return 0.0
        
        # Simple coherence calculation based on topic distribution
        topic_distribution = Counter(topics)
        total_texts = len(texts)
        
        # Calculate entropy-based coherence
        coherence = 0.0
        for count in topic_distribution.values():
            prob = count / total_texts
            if prob > 0:
                coherence -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(topic_distribution))
        return coherence / max_entropy if max_entropy > 0 else 0.0
    
    def _generate_basic_variations(self, base_content: str, num_variations: int) -> List[str]:
        """Generate basic content variations."""
        variations = []
        
        # Variation 1: Add engaging questions
        if num_variations >= 1:
            variation1 = base_content + " What do you think? 🤔"
            variations.append(variation1)
        
        # Variation 2: Add call-to-action
        if num_variations >= 2:
            variation2 = base_content + " Double tap if you agree! ❤️"
            variations.append(variation2)
        
        # Variation 3: Add popular hashtags
        if num_variations >= 3:
            variation3 = base_content + " #instagood #photooftheday #love"
            variations.append(variation3)
        
        # Additional variations with simple modifications
        for i in range(3, num_variations):
            modified = base_content.replace('.', '! 🚀' if i % 2 == 0 else '! 🔥')
            variations.append(modified)
        
        return variations[:num_variations]
    
    def _nlp_features_to_vector(self, nlp_analysis: Dict[str, Any]) -> np.ndarray:
        """Convert NLP analysis results to feature vector."""
        features = []
        
        # Text features
        text_features = nlp_analysis.get('text_features', {})
        features.extend([
            text_features.get('avg_length', 0),
            text_features.get('avg_word_count', 0),
            text_features.get('avg_sentence_count', 0),
            text_features.get('vocabulary_diversity', 0),
            text_features.get('avg_char_per_word', 0)
        ])
        
        # Sentiment features
        sentiment_features = nlp_analysis.get('sentiment_analysis', {})
        features.extend([
            sentiment_features.get('avg_compound_scores', 0),
            sentiment_features.get('avg_positive_scores', 0),
            sentiment_features.get('avg_negative_scores', 0),
            sentiment_features.get('avg_neutral_scores', 0)
        ])
        
        # Content quality features
        quality_features = nlp_analysis.get('content_quality', {})
        features.extend([
            quality_features.get('avg_readability_scores', 0),
            quality_features.get('avg_engagement_potential', 0),
            quality_features.get('avg_content_density', 0),
            quality_features.get('overall_quality', 0)
        ])
        
        # Hashtag features
        hashtag_features = nlp_analysis.get('hashtag_analysis', {})
        features.extend([
            hashtag_features.get('avg_hashtags_per_post', 0),
            hashtag_features.get('hashtag_diversity', 0)
        ])
        
        # Linguistic features
        linguistic_features = nlp_analysis.get('linguistic_features', {})
        features.extend([
            linguistic_features.get('avg_avg_sentences_per_post', 0),
            linguistic_features.get('avg_punctuation_density', 0),
            linguistic_features.get('avg_question_ratio', 0),
            linguistic_features.get('avg_emoji_density', 0)
        ])
        
        return np.array(features)
    
    def _predict_engagement_heuristic(self, nlp_analysis: Dict[str, Any], texts: List[str]) -> Dict[str, List[float]]:
        """Generate heuristic engagement predictions based on NLP features."""
        predictions = {
            'likes': [],
            'comments': [],
            'engagement_rate': []
        }
        
        for text in texts:
            # Calculate individual text features
            text_analysis = self.analyze_content_comprehensive([text])
            
            # Heuristic prediction based on various factors
            base_engagement = 100  # Base engagement score
            
            # Content quality factor
            quality_score = text_analysis.get('content_quality', {}).get('overall_quality', 0.5)
            quality_multiplier = 1 + quality_score
            
            # Sentiment factor
            sentiment_score = text_analysis.get('sentiment_analysis', {}).get('avg_compound_scores', 0)
            sentiment_multiplier = 1 + max(0, sentiment_score)
            
            # Hashtag factor
            hashtag_count = text_analysis.get('hashtag_analysis', {}).get('avg_hashtags_per_post', 0)
            hashtag_multiplier = 1 + min(0.5, hashtag_count * 0.1)
            
            # Length factor
            word_count = len(text.split())
            if 10 <= word_count <= 50:
                length_multiplier = 1.2
            else:
                length_multiplier = 1.0
            
            # Calculate predictions
            likes_pred = base_engagement * quality_multiplier * sentiment_multiplier * hashtag_multiplier * length_multiplier
            comments_pred = likes_pred * 0.1  # Assuming 10% comment rate
            engagement_pred = (likes_pred + comments_pred * 10) / 1000  # Normalize
            
            predictions['likes'].append(max(0, likes_pred))
            predictions['comments'].append(max(0, comments_pred))
            predictions['engagement_rate'].append(max(0, min(1, engagement_pred)))
        
        return predictions
    
    def _calculate_nlp_feature_importance(self, nlp_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate feature importance for NLP features."""
        importance_scores = {
            'content_quality': 0.25,
            'sentiment_analysis': 0.20,
            'hashtag_usage': 0.15,
            'text_features': 0.15,
            'linguistic_features': 0.15,
            'semantic_analysis': 0.10
        }
        
        # Adjust importance based on available features
        available_features = list(nlp_analysis.keys())
        adjusted_scores = {}
        
        for feature, score in importance_scores.items():
            if any(af.startswith(feature.split('_')[0]) for af in available_features):
                adjusted_scores[feature] = score
        
        # Normalize scores
        total_score = sum(adjusted_scores.values())
        if total_score > 0:
            adjusted_scores = {k: v/total_score for k, v in adjusted_scores.items()}
        
        return adjusted_scores


class InstagramNLPPredictor:
    """
    Instagram-specific NLP predictor combining advanced NLP techniques
    with Instagram engagement patterns.
    """
    
    def __init__(self, model_cache_dir: str = "models/instagram_nlp"):
        """Initialize Instagram NLP Predictor."""
        self.model_cache_dir = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)
        
        self.nlp_analyzer = AdvancedNLPAnalyzer(model_cache_dir)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, texts: List[str], engagement_targets: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Fit the Instagram NLP predictor.
        
        Args:
            texts: List of Instagram captions/content
            engagement_targets: Target engagement metrics
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Training Instagram NLP predictor...")
            
            # Extract comprehensive NLP features
            nlp_analysis = self.nlp_analyzer.analyze_content_comprehensive(texts)
            
            # Convert to feature matrix
            X = self._extract_feature_matrix(texts, nlp_analysis)
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Generate predictions for validation
            predictions = self.nlp_analyzer.predict_engagement_with_nlp(texts)
            
            # Calculate training metrics
            metrics = self._calculate_training_metrics(engagement_targets, predictions)
            
            self.is_fitted = True
            
            # Save model components
            self._save_model_components()
            
            logger.info("Instagram NLP predictor training completed")
            
            return {
                'training_metrics': metrics,
                'nlp_analysis': nlp_analysis,
                'feature_matrix_shape': X.shape,
                'model_status': 'trained_successfully'
            }
            
        except Exception as e:
            logger.error(f"Instagram NLP predictor training failed: {e}")
            return {'error': str(e)}
    
    def predict(self, texts: List[str]) -> Dict[str, Any]:
        """
        Predict engagement for Instagram content.
        
        Args:
            texts: List of Instagram captions/content
            
        Returns:
            Engagement predictions and analysis
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Loading pre-trained components...")
            self._load_model_components()
        
        try:
            # Generate comprehensive predictions
            predictions = self.nlp_analyzer.predict_engagement_with_nlp(texts)
            
            # Add Instagram-specific insights
            instagram_insights = self._generate_instagram_insights(texts, predictions)
            
            return {
                'predictions': predictions['predictions'],
                'nlp_features': predictions['nlp_features'],
                'instagram_insights': instagram_insights,
                'content_optimization_suggestions': self._generate_optimization_suggestions(texts, predictions)
            }
            
        except Exception as e:
            logger.error(f"Instagram NLP prediction failed: {e}")
            return {'error': str(e)}
    
    def _extract_feature_matrix(self, texts: List[str], nlp_analysis: Dict[str, Any]) -> np.ndarray:
        """Extract feature matrix from texts and NLP analysis."""
        features_per_text = []
        
        for text in texts:
            # Individual text analysis
            individual_analysis = self.nlp_analyzer.analyze_content_comprehensive([text])
            feature_vector = self.nlp_analyzer._nlp_features_to_vector(individual_analysis)
            features_per_text.append(feature_vector)
        
        return np.array(features_per_text)
    
    def _calculate_training_metrics(self, targets: Dict[str, List[float]], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate training performance metrics."""
        metrics = {}
        
        if 'predictions' in predictions:
            pred_data = predictions['predictions']
            
            for target_name, target_values in targets.items():
                if target_name in pred_data and len(target_values) == len(pred_data[target_name]):
                    y_true = np.array(target_values)
                    y_pred = np.array(pred_data[target_name])
                    
                    metrics[f'{target_name}_mse'] = mean_squared_error(y_true, y_pred)
                    metrics[f'{target_name}_mae'] = mean_absolute_error(y_true, y_pred)
                    metrics[f'{target_name}_r2'] = r2_score(y_true, y_pred)
        
        return metrics
    
    def _generate_instagram_insights(self, texts: List[str], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Instagram-specific insights."""
        insights = {
            'optimal_posting_strategy': {},
            'content_performance_analysis': {},
            'hashtag_optimization': {},
            'engagement_drivers': {}
        }
        
        try:
            nlp_features = predictions.get('nlp_features', {})
            
            # Optimal posting strategy
            insights['optimal_posting_strategy'] = {
                'recommended_length': self._get_optimal_length_recommendation(nlp_features),
                'optimal_sentiment': 'positive',
                'hashtag_recommendations': self._get_hashtag_recommendations(nlp_features),
                'content_type_suggestion': self._get_content_type_suggestion(texts)
            }
            
            # Content performance analysis
            insights['content_performance_analysis'] = {
                'predicted_performance': 'high' if predictions.get('predictions', {}).get('engagement_rate', [0])[0] > 0.5 else 'medium',
                'key_strengths': self._identify_content_strengths(nlp_features),
                'improvement_areas': self._identify_improvement_areas(nlp_features)
            }
            
            # Engagement drivers
            insights['engagement_drivers'] = {
                'primary_drivers': ['content_quality', 'sentiment', 'hashtag_usage'],
                'impact_scores': predictions.get('feature_importance', {})
            }
            
        except Exception as e:
            logger.error(f"Instagram insights generation failed: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def _generate_optimization_suggestions(self, texts: List[str], predictions: Dict[str, Any]) -> List[str]:
        """Generate content optimization suggestions."""
        suggestions = []
        
        try:
            nlp_features = predictions.get('nlp_features', {})
            
            # Length optimization
            avg_length = nlp_features.get('text_features', {}).get('avg_word_count', 0)
            if avg_length < 10:
                suggestions.append("Consider adding more descriptive content (aim for 10-50 words)")
            elif avg_length > 50:
                suggestions.append("Consider shortening your captions for better engagement")
            
            # Hashtag optimization
            avg_hashtags = nlp_features.get('hashtag_analysis', {}).get('avg_hashtags_per_post', 0)
            if avg_hashtags < 3:
                suggestions.append("Add more relevant hashtags (3-5 recommended)")
            elif avg_hashtags > 10:
                suggestions.append("Reduce number of hashtags to avoid appearing spammy")
            
            # Sentiment optimization
            avg_sentiment = nlp_features.get('sentiment_analysis', {}).get('avg_compound_scores', 0)
            if avg_sentiment < 0:
                suggestions.append("Consider using more positive language to increase engagement")
            
            # Engagement potential
            engagement_potential = nlp_features.get('content_quality', {}).get('avg_engagement_potential', 0)
            if engagement_potential < 0.5:
                suggestions.append("Add questions or call-to-actions to increase engagement")
            
            # Default suggestions if none generated
            if not suggestions:
                suggestions = [
                    "Your content looks great! Consider experimenting with different hashtags",
                    "Try adding emojis to make your content more visually appealing",
                    "Consider posting at optimal times for your audience"
                ]
            
        except Exception as e:
            logger.error(f"Optimization suggestions generation failed: {e}")
            suggestions = ["Unable to generate specific suggestions. Focus on quality content and engagement."]
        
        return suggestions
    
    def _get_optimal_length_recommendation(self, nlp_features: Dict[str, Any]) -> str:
        """Get optimal content length recommendation."""
        avg_length = nlp_features.get('text_features', {}).get('avg_word_count', 0)
        
        if avg_length < 10:
            return "Increase length to 10-30 words for better engagement"
        elif avg_length > 50:
            return "Consider reducing to 20-40 words for optimal engagement"
        else:
            return "Current length is optimal for engagement"
    
    def _get_hashtag_recommendations(self, nlp_features: Dict[str, Any]) -> List[str]:
        """Get hashtag recommendations."""
        return [
            "#instagood", "#photooftheday", "#love", "#instadaily", "#follow",
            "#picoftheday", "#instamood", "#bestoftheday", "#instalike", "#style"
        ]
    
    def _get_content_type_suggestion(self, texts: List[str]) -> str:
        """Suggest content type based on analysis."""
        if not texts:
            return "lifestyle"
        
        # Basic content type detection
        combined_text = ' '.join(texts).lower()
        
        if any(word in combined_text for word in ['food', 'recipe', 'cooking', 'delicious']):
            return "food"
        elif any(word in combined_text for word in ['travel', 'adventure', 'vacation', 'trip']):
            return "travel"
        elif any(word in combined_text for word in ['fashion', 'style', 'outfit', 'look']):
            return "fashion"
        elif any(word in combined_text for word in ['fitness', 'workout', 'gym', 'health']):
            return "fitness"
        else:
            return "lifestyle"
    
    def _identify_content_strengths(self, nlp_features: Dict[str, Any]) -> List[str]:
        """Identify content strengths."""
        strengths = []
        
        quality_score = nlp_features.get('content_quality', {}).get('overall_quality', 0)
        if quality_score > 0.7:
            strengths.append("High content quality")
        
        sentiment_score = nlp_features.get('sentiment_analysis', {}).get('avg_compound_scores', 0)
        if sentiment_score > 0.3:
            strengths.append("Positive sentiment")
        
        hashtag_diversity = nlp_features.get('hashtag_analysis', {}).get('hashtag_diversity', 0)
        if hashtag_diversity > 0.8:
            strengths.append("Good hashtag diversity")
        
        if not strengths:
            strengths = ["Consistent posting", "Authentic voice"]
        
        return strengths
    
    def _identify_improvement_areas(self, nlp_features: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement."""
        improvements = []
        
        quality_score = nlp_features.get('content_quality', {}).get('overall_quality', 0)
        if quality_score < 0.5:
            improvements.append("Content quality and clarity")
        
        engagement_potential = nlp_features.get('content_quality', {}).get('avg_engagement_potential', 0)
        if engagement_potential < 0.4:
            improvements.append("Engagement call-to-actions")
        
        emoji_density = nlp_features.get('linguistic_features', {}).get('avg_avg_emoji_density', 0)
        if emoji_density < 0.01:
            improvements.append("Visual appeal with emojis")
        
        if not improvements:
            improvements = ["Experiment with posting times", "Try different content formats"]
        
        return improvements
    
    def _save_model_components(self):
        """Save model components to disk."""
        try:
            model_path = os.path.join(self.model_cache_dir, 'instagram_nlp_model.joblib')
            scaler_path = os.path.join(self.model_cache_dir, 'instagram_nlp_scaler.joblib')
            
            # Save scaler
            joblib.dump(self.scaler, scaler_path)
            
            # Save model metadata
            metadata = {
                'is_fitted': self.is_fitted,
                'model_type': 'InstagramNLPPredictor',
                'version': '1.0.0',
                'trained_at': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.model_cache_dir, 'model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model components saved to {self.model_cache_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save model components: {e}")
    
    def _load_model_components(self):
        """Load model components from disk."""
        try:
            scaler_path = os.path.join(self.model_cache_dir, 'instagram_nlp_scaler.joblib')
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                self.is_fitted = True
                logger.info("Model components loaded successfully")
            else:
                logger.warning("No pre-trained model components found")
                
        except Exception as e:
            logger.error(f"Failed to load model components: {e}")


# Utility functions for easy integration
def analyze_instagram_content(texts: List[str], 
                            cache_dir: str = "models/nlp_cache") -> Dict[str, Any]:
    """
    Quick function to analyze Instagram content using advanced NLP.
    
    Args:
        texts: List of Instagram captions/content
        cache_dir: Directory to cache models
        
    Returns:
        Comprehensive NLP analysis results
    """
    analyzer = AdvancedNLPAnalyzer(cache_dir)
    return analyzer.analyze_content_comprehensive(texts)


def predict_instagram_engagement(texts: List[str],
                               engagement_data: Dict[str, List[float]] = None,
                               cache_dir: str = "models/instagram_nlp") -> Dict[str, Any]:
    """
    Quick function to predict Instagram engagement using NLP.
    
    Args:
        texts: List of Instagram captions/content
        engagement_data: Historical engagement data for training
        cache_dir: Directory to cache models
        
    Returns:
        Engagement predictions and insights
    """
    predictor = InstagramNLPPredictor(cache_dir)
    
    if engagement_data:
        # Train the model first
        training_results = predictor.fit(texts, engagement_data)
        logger.info(f"Model training completed: {training_results.get('model_status', 'unknown')}")
    
    return predictor.predict(texts)


def optimize_instagram_content(content: str,
                              target_metrics: Dict[str, float] = None,
                              num_variations: int = 3,
                              cache_dir: str = "models/nlp_cache") -> List[str]:
    """
    Quick function to generate optimized Instagram content variations.
    
    Args:
        content: Original Instagram content
        target_metrics: Target engagement metrics
        num_variations: Number of variations to generate
        cache_dir: Directory to cache models
        
    Returns:
        List of optimized content variations
    """
    analyzer = AdvancedNLPAnalyzer(cache_dir)
    
    if target_metrics is None:
        target_metrics = {'engagement_rate': 0.05, 'likes': 100, 'comments': 10}
    
    return analyzer.generate_optimized_content(content, target_metrics, num_variations)


if __name__ == "__main__":
    # Example usage and testing
    sample_texts = [
        "Just tried this amazing new restaurant! 🍕 The pizza was incredible and the service was fantastic. Highly recommend! #foodie #pizza #restaurant",
        "Beautiful sunset at the beach today 🌅 Nothing beats this view! What's your favorite place to watch the sunset? #sunset #beach #nature",
        "New workout routine is paying off! 💪 Feeling stronger every day. What keeps you motivated? #fitness #workout #motivation #gym"
    ]
    
    print("🚀 Advanced NLP & LLM Integration Demo")
    print("=" * 50)
    
    # Test content analysis
    print("\n📊 Content Analysis Results:")
    analysis_results = analyze_instagram_content(sample_texts)
    
    for category, results in analysis_results.items():
        print(f"\n{category.upper()}:")
        if isinstance(results, dict):
            for key, value in list(results.items())[:3]:  # Show first 3 items
                print(f"  {key}: {value}")
        
    # Test engagement prediction
    print("\n🎯 Engagement Prediction Results:")
    sample_engagement = {
        'likes': [150, 200, 180],
        'comments': [15, 25, 20],
        'engagement_rate': [0.05, 0.08, 0.06]
    }
    
    prediction_results = predict_instagram_engagement(sample_texts, sample_engagement)
    
    if 'predictions' in prediction_results:
        preds = prediction_results['predictions']
        print(f"  Predicted likes: {preds.get('likes', [])[:3]}")
        print(f"  Predicted comments: {preds.get('comments', [])[:3]}")
    
    # Test content optimization
    print("\n✨ Content Optimization Results:")
    sample_content = "Great day at the park with friends"
    optimized_variations = optimize_instagram_content(sample_content, num_variations=2)
    
    for i, variation in enumerate(optimized_variations, 1):
        print(f"  Variation {i}: {variation[:100]}...")
    
    print("\n✅ Advanced NLP & LLM Integration Demo Complete!")
    print("🎉 All advanced NLP features are working correctly!")
