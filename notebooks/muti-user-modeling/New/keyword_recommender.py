import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings
warnings.filterwarnings('ignore')

class KeywordRecommendationSystem:
    def __init__(self, csv_path, models_dir='models'):
        """Initialize with training data and existing predictor"""
        self.data = pd.read_csv(csv_path)
        self.models_dir = models_dir
        self.hashtag_impact = {}
        self.keyword_impact = {}
        self.category_keywords = {}
        self.predictor = None
        
        # Load the data and analyze
        self._preprocess_data()
        self._analyze_hashtag_impact()
        self._analyze_keyword_impact()
        self._build_category_mapping()
    
    def set_predictor(self, predictor):
        """Set the predictor instance"""
        self.predictor = predictor
    
    def _preprocess_data(self):
        """Preprocess the data for analysis"""
        # Handle missing values
        self.data['caption'] = self.data['caption'].fillna('')
        self.data['hashtags'] = self.data['hashtags'].fillna('')
        
        # Calculate engagement rate if not present OR if all values are zero
        if ('engagement_rate' not in self.data.columns or 
            self.data['engagement_rate'].sum() == 0):
            if 'likes' in self.data.columns and '#Followers' in self.data.columns:
                self.data['engagement_rate'] = self.data['likes'] / (self.data['#Followers'] + 1)
                print(f"🔧 Calculated engagement rate from likes/followers - Mean: {self.data['engagement_rate'].mean():.6f}")
            else:
                self.data['engagement_rate'] = 0
        
        # Handle sentiment weighted engagement
        if 'sentiment_weighted_engagement' not in self.data.columns:
            # Create a proxy using available columns
            if 'likes' in self.data.columns and 'comments_count' in self.data.columns:
                self.data['sentiment_weighted_engagement'] = (
                    self.data['likes'] * 0.7 + self.data['comments_count'] * 2.0
                ) / (self.data['#Followers'] + 1)
            else:
                self.data['sentiment_weighted_engagement'] = self.data['engagement_rate']
        
        print(f"✅ Loaded {len(self.data)} posts for analysis")
    
    def _analyze_hashtag_impact(self):
        """Analyze impact of individual hashtags on engagement"""
        hashtag_stats = defaultdict(list)
        
        for _, row in self.data.iterrows():
            hashtags = str(row.get('hashtags', '')).lower()
            engagement_rate = row.get('engagement_rate', 0)
            weighted_engagement = row.get('sentiment_weighted_engagement', 0)
            likes = row.get('likes', 0)
            comments = row.get('comments_count', 0)
            
            # Extract hashtags - handle comma-separated format
            if ',' in hashtags:
                # Split by comma and clean
                hashtag_list = [tag.strip().replace('#', '').lower() for tag in hashtags.split(',') if tag.strip()]
            else:
                # Use regex for #hashtag format
                hashtag_list = re.findall(r'#(\w+)', hashtags)
            
            for hashtag in hashtag_list:
                hashtag_stats[hashtag].append({
                    'engagement_rate': engagement_rate,
                    'weighted_engagement': weighted_engagement,
                    'likes': likes,
                    'comments': comments
                })
        
        # Calculate average impact for each hashtag
        for hashtag, stats in hashtag_stats.items():
            if len(stats) >= 3:  # Only consider hashtags with enough data
                avg_engagement = np.mean([s['engagement_rate'] for s in stats])
                avg_weighted = np.mean([s['weighted_engagement'] for s in stats])
                avg_likes = np.mean([s['likes'] for s in stats])
                count = len(stats)
                
                # Create a meaningful combined score
                combined_score = avg_engagement * min(count / 10.0, 1.0)
                impact_score = avg_engagement * avg_weighted * 0.1 + avg_engagement * 0.5
                
                self.hashtag_impact[hashtag] = {
                    'avg_engagement_rate': avg_engagement,
                    'avg_weighted_engagement': avg_weighted,
                    'avg_likes': avg_likes,
                    'frequency': count,
                    'count': count,  # Add both for compatibility
                    'impact_score': max(impact_score, combined_score),  # Use better score
                    'combined_score': combined_score,  # Add for compatibility
                    'avg_engagement': avg_engagement  # Add for compatibility
                }
        
        print(f"✅ Analyzed {len(self.hashtag_impact)} hashtags")
    
    def _analyze_keyword_impact(self):
        """Analyze impact of caption keywords on engagement"""
        # Extract keywords from captions
        captions = self.data['caption'].fillna('').astype(str)
        
        # Create TF-IDF vectorizer for captions
        tfidf = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8
        )
        
        try:
            caption_tfidf = tfidf.fit_transform(captions)
            feature_names = tfidf.get_feature_names_out()
            
            # Calculate correlation between keywords and engagement
            engagement_rates = self.data['engagement_rate'].fillna(0)
            weighted_engagement = self.data['sentiment_weighted_engagement'].fillna(0)
            
            for i, keyword in enumerate(feature_names):
                keyword_scores = caption_tfidf[:, i].toarray().flatten()
                
                # Calculate correlation with engagement metrics
                try:
                    eng_corr = np.corrcoef(keyword_scores, engagement_rates)[0, 1]
                    weighted_corr = np.corrcoef(keyword_scores, weighted_engagement)[0, 1]
                    
                    if not np.isnan(eng_corr) and not np.isnan(weighted_corr):
                        self.keyword_impact[keyword] = {
                            'engagement_correlation': eng_corr,
                            'weighted_correlation': weighted_corr,
                            'combined_score': (eng_corr + weighted_corr) / 2,
                            'frequency': np.sum(keyword_scores > 0)
                        }
                except:
                    continue
            
            print(f"✅ Analyzed {len(self.keyword_impact)} keywords")
        except Exception as e:
            print(f"⚠️ Warning: Keyword analysis failed: {e}")
            self.keyword_impact = {}
    
    def _build_category_mapping(self):
        """Build category-specific keyword recommendations"""
        if 'Category' not in self.data.columns:
            print("⚠️ No Category column found, skipping category mapping")
            return
        
        categories = self.data['Category'].unique()
        
        for category in categories:
            if pd.isna(category):
                continue
                
            category_data = self.data[self.data['Category'] == category]
            
            # Get top hashtags for this category
            category_hashtags = []
            for hashtags in category_data['hashtags'].fillna(''):
                category_hashtags.extend(re.findall(r'#(\w+)', str(hashtags).lower()))
            
            hashtag_counts = Counter(category_hashtags)
            top_hashtags = [h for h, c in hashtag_counts.most_common(15)]
            
            # Get high-performing keywords for this category
            high_engagement_captions = category_data[
                category_data['engagement_rate'] > category_data['engagement_rate'].quantile(0.75)
            ]['caption'].fillna('').astype(str)
            
            # Extract common words from high-performing captions
            all_words = []
            for caption in high_engagement_captions:
                words = re.findall(r'\b\w+\b', caption.lower())
                all_words.extend([w for w in words if len(w) > 3])
            
            word_counts = Counter(all_words)
            top_keywords = [w for w, c in word_counts.most_common(10)]
            
            self.category_keywords[category] = {
                'hashtags': top_hashtags,
                'keywords': top_keywords
            }
        
        print(f"✅ Built category mappings for {len(self.category_keywords)} categories")
    
    def recommend_hashtags(self, user_data, top_n=10, category=None):
        """Recommend hashtags to maximize engagement"""
        if not self.predictor:
            raise ValueError("Predictor not set. Call set_predictor() first.")
        
        # Filter hashtags by category if specified
        if category and category in self.category_keywords:
            candidate_hashtags = self.category_keywords[category]['hashtags']
        else:
            candidate_hashtags = list(self.hashtag_impact.keys())
        
        # Sort by impact score first to get best candidates
        sorted_hashtags = sorted(
            [(h, self.hashtag_impact.get(h, {}).get('combined_score', 
             self.hashtag_impact.get(h, {}).get('impact_score', 0))) 
             for h in candidate_hashtags if h in self.hashtag_impact],
            key=lambda x: x[1], reverse=True
        )
        
        recommendations = []
        baseline_prediction = self.predictor.predict(user_data)
        baseline_engagement = baseline_prediction.get('Engagement Rate', 0)
        baseline_weighted = baseline_prediction.get('Sentiment-Weighted Engagement', 0)
        
        # Test top candidates
        for hashtag, historical_score in sorted_hashtags[:min(30, len(sorted_hashtags))]:
            # Test this hashtag with user's data
            test_data = user_data.copy()
            current_hashtags = test_data.get('hashtags', '')
            test_hashtags = f"{current_hashtags} #{hashtag}".strip()
            test_data['hashtags'] = test_hashtags
            
            try:
                predictions = self.predictor.predict(test_data)
                predicted_engagement = predictions.get('Engagement Rate', 0)
                predicted_weighted = predictions.get('Sentiment-Weighted Engagement', 0)
                
                # Calculate improvement
                engagement_improvement = predicted_engagement - baseline_engagement
                weighted_improvement = predicted_weighted - baseline_weighted
                
                recommendations.append({
                    'hashtag': hashtag,
                    'predicted_engagement': predicted_engagement,
                    'predicted_weighted_engagement': predicted_weighted,
                    'engagement_improvement': engagement_improvement,
                    'weighted_improvement': weighted_improvement,
                    'historical_impact': historical_score,
                    'frequency': self.hashtag_impact[hashtag].get('count', 
                                self.hashtag_impact[hashtag].get('frequency', 0)),
                    'combined_score': engagement_improvement * 0.6 + weighted_improvement * 0.4
                })
            except Exception as e:
                continue
        
        # Sort by combined improvement score
        recommendations.sort(key=lambda x: x['combined_score'], reverse=True)
        return recommendations[:top_n]
    
    def recommend_keywords(self, user_data, top_n=10):
        """Recommend keywords for caption to maximize engagement"""
        if not self.predictor:
            raise ValueError("Predictor not set. Call set_predictor() first.")
        
        # Get high-impact keywords
        high_impact_keywords = sorted(
            self.keyword_impact.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )[:min(50, len(self.keyword_impact))]
        
        recommendations = []
        baseline_prediction = self.predictor.predict(user_data)
        baseline_engagement = baseline_prediction.get('Engagement Rate', 0)
        baseline_weighted = baseline_prediction.get('Sentiment-Weighted Engagement', 0)
        
        current_caption = user_data.get('caption', '').lower()
        
        for keyword, impact_data in high_impact_keywords:
            # Skip if keyword already in caption
            if keyword in current_caption:
                continue
            
            # Test adding this keyword to caption
            test_data = user_data.copy()
            test_caption = f"{user_data.get('caption', '')} {keyword}".strip()
            test_data['caption'] = test_caption
            
            try:
                predictions = self.predictor.predict(test_data)
                predicted_engagement = predictions.get('Engagement Rate', 0)
                predicted_weighted = predictions.get('Sentiment-Weighted Engagement', 0)
                
                # Calculate improvement
                engagement_improvement = predicted_engagement - baseline_engagement
                weighted_improvement = predicted_weighted - baseline_weighted
                
                recommendations.append({
                    'keyword': keyword,
                    'predicted_engagement': predicted_engagement,
                    'predicted_weighted_engagement': predicted_weighted,
                    'engagement_improvement': engagement_improvement,
                    'weighted_improvement': weighted_improvement,
                    'historical_correlation': impact_data['combined_score'],
                    'frequency': impact_data['frequency'],
                    'impact_score': engagement_improvement * 0.6 + weighted_improvement * 0.4
                })
            except Exception as e:
                continue
        
        # Sort by impact score
        recommendations.sort(key=lambda x: x['impact_score'], reverse=True)
        return recommendations[:top_n]
    
    def optimize_content(self, user_data, category=None):
        """Provide comprehensive content optimization recommendations"""
        if not self.predictor:
            raise ValueError("Predictor not set. Call set_predictor() first.")
        
        baseline_predictions = self.predictor.predict(user_data)
        baseline_engagement = baseline_predictions.get('Engagement Rate', 0)
        baseline_weighted = baseline_predictions.get('Sentiment-Weighted Engagement', 0)
        
        # Get recommendations
        hashtag_recs = self.recommend_hashtags(user_data, top_n=8, category=category)
        keyword_recs = self.recommend_keywords(user_data, top_n=6)
        
        # Test combined optimization
        optimized_data = user_data.copy()
        
        # Add top recommended hashtags (select best ones that improve engagement)
        positive_hashtag_recs = [r for r in hashtag_recs if r['combined_score'] > 0][:3]
        if positive_hashtag_recs:
            top_hashtags = [f"#{rec['hashtag']}" for rec in positive_hashtag_recs]
            current_hashtags = optimized_data.get('hashtags', '')
            optimized_hashtags = f"{current_hashtags} {' '.join(top_hashtags)}".strip()
            optimized_data['hashtags'] = optimized_hashtags
        else:
            optimized_hashtags = optimized_data.get('hashtags', '')
        
        # Add top recommended keywords to caption
        positive_keyword_recs = [r for r in keyword_recs if r['impact_score'] > 0][:2]
        if positive_keyword_recs:
            top_keywords = [rec['keyword'] for rec in positive_keyword_recs]
            current_caption = optimized_data.get('caption', '')
            optimized_caption = f"{current_caption} {' '.join(top_keywords)}".strip()
            optimized_data['caption'] = optimized_caption
        else:
            optimized_caption = optimized_data.get('caption', '')
        
        # Get optimized predictions
        optimized_predictions = self.predictor.predict(optimized_data)
        
        return {
            'baseline': {
                'engagement_rate': baseline_engagement,
                'weighted_engagement': baseline_weighted
            },
            'optimized': {
                'engagement_rate': optimized_predictions.get('Engagement Rate', 0),
                'weighted_engagement': optimized_predictions.get('Sentiment-Weighted Engagement', 0)
            },
            'improvement': {
                'engagement_lift': optimized_predictions.get('Engagement Rate', 0) - baseline_engagement,
                'weighted_lift': optimized_predictions.get('Sentiment-Weighted Engagement', 0) - baseline_weighted
            },
            'recommendations': {
                'hashtags': hashtag_recs,
                'keywords': keyword_recs,
                'optimized_hashtags': optimized_hashtags,
                'optimized_caption': optimized_caption
            }
        }
    
    def get_stats(self):
        """Get statistics about the recommendation system"""
        return {
            'total_posts': len(self.data),
            'hashtags_analyzed': len(self.hashtag_impact),
            'keywords_analyzed': len(self.keyword_impact),
            'categories': len(self.category_keywords),
            'top_hashtags': sorted(self.hashtag_impact.items(), 
                                 key=lambda x: x[1]['impact_score'], reverse=True)[:10],
            'top_keywords': sorted(self.keyword_impact.items(), 
                                 key=lambda x: x[1]['combined_score'], reverse=True)[:10]
        }

def demo_keyword_recommendation():
    """Demonstrate the keyword recommendation system"""
    # Initialize the system
    try:
        recommender = KeywordRecommendationSystem(
            '/home/krishantha/Github/fyp-l4/notebooks/muti-user-modeling/New/balanced_posts_with_sentiment_emotion_analysis.csv'
        )
        
        # Import and initialize predictor
        from instagram_predictor_cli import InstagramEngagementPredictor
        predictor = InstagramEngagementPredictor()
        recommender.set_predictor(predictor)
        
        # Example user data
        user_data = {
            'followers': 5000,
            'followees': 1200,
            'posts': 150,
            'caption': 'Beautiful sunset today',
            'hashtags': '#photography'
        }
        
        # Get optimization recommendations
        results = recommender.optimize_content(user_data)
        
        print("\n🎯 CONTENT OPTIMIZATION RECOMMENDATIONS")
        print("=" * 50)
        print(f"Baseline Engagement Rate: {results['baseline']['engagement_rate']:.2%}")
        print(f"Optimized Engagement Rate: {results['optimized']['engagement_rate']:.2%}")
        print(f"Improvement: +{results['improvement']['engagement_lift']:.2%}")
        print()
        print(f"Baseline Weighted Engagement: {results['baseline']['weighted_engagement']:.4f}")
        print(f"Optimized Weighted Engagement: {results['optimized']['weighted_engagement']:.4f}")
        print(f"Improvement: +{results['improvement']['weighted_lift']:.4f}")
        print()
        print("📋 RECOMMENDED HASHTAGS:")
        for i, rec in enumerate(results['recommendations']['hashtags'][:5], 1):
            print(f"  {i}. #{rec['hashtag']} (Improvement: +{rec['combined_score']:.4f})")
        print()
        print("📝 RECOMMENDED KEYWORDS:")
        for i, rec in enumerate(results['recommendations']['keywords'][:5], 1):
            print(f"  {i}. {rec['keyword']} (Impact: +{rec['impact_score']:.4f})")
        print()
        print(f"🏷️ Optimized Hashtags: {results['recommendations']['optimized_hashtags']}")
        print(f"✍️ Optimized Caption: {results['recommendations']['optimized_caption']}")
        
    except Exception as e:
        print(f"Error in demo: {e}")

if __name__ == "__main__":
    demo_keyword_recommendation()
