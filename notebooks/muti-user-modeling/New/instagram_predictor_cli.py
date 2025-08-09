import argparse
import pickle
import joblib
import numpy as np
import pandas as pd
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class InstagramEngagementPredictor:
    def __init__(self, models_dir='models'):
        """Initialize the predictor with saved models"""
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        self.tfidf = None
        self.metadata = None
        self.sentiment_analyzer = None
        
        self.load_models()
        self.load_sentiment_analyzer()
    
    def load_models(self):
        """Load all saved models and preprocessing objects"""
        try:
            print("Loading models...")
            
            # Load individual models
            self.models['engagement'] = joblib.load(f'{self.models_dir}/engagement_model.pkl')
            self.models['comments'] = joblib.load(f'{self.models_dir}/comments_model.pkl')
            self.models['sentiment'] = joblib.load(f'{self.models_dir}/sentiment_model.pkl')
            self.models['weighted'] = joblib.load(f'{self.models_dir}/weighted_model.pkl')
            
            # Load preprocessing objects
            self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')
            self.tfidf = joblib.load(f'{self.models_dir}/tfidf_vectorizer.pkl')
            
            # Load metadata
            with open(f'{self.models_dir}/metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def load_sentiment_analyzer(self):
        """Load sentiment analysis pipeline"""
        try:
            print("Loading sentiment analyzer...")
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Try offline mode first
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512
            )
            print("✅ Sentiment analyzer loaded!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load sentiment analyzer: {e}")
            print("Using fallback sentiment analysis...")
            self.sentiment_analyzer = None
    
    def get_sentiment_features(self, text):
        """Extract sentiment features from text"""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {'caption_sentiment_score': 0.0}
        
        try:
            result = self.sentiment_analyzer(text, truncation=True, max_length=512)[0]
            score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
            return {'caption_sentiment_score': score}
        except Exception as e:
            print(f"Warning: Error analyzing text: {e}")
            return {'caption_sentiment_score': 0.0}
    
    def preprocess_input(self, user_data):
        """Preprocess user input into feature vector"""
        try:
            # Get sentiment features
            sentiment_features = self.get_sentiment_features(user_data.get('caption', ''))
            
            # Process TF-IDF features
            user_caption = user_data.get('caption', '')
            if user_caption and len(self.metadata['tfidf_columns']) > 0:
                try:
                    user_tfidf = self.tfidf.transform([user_caption])
                    user_tfidf_dense = user_tfidf.toarray()[0]
                except Exception as e:
                    print(f"Warning: TF-IDF transform error: {e}")
                    user_tfidf_dense = np.zeros(len(self.metadata['tfidf_columns']))
            else:
                user_tfidf_dense = np.zeros(len(self.metadata['tfidf_columns']))
            
            # Create feature vector
            user_features = []
            
            for feature in self.metadata['feature_names']:
                if feature == 'caption_sentiment_score':
                    user_features.append(sentiment_features['caption_sentiment_score'])
                elif feature == '#Followers':
                    user_features.append(user_data.get('followers', user_data.get('#Followers', 0)))
                elif feature == '#Followees':
                    user_features.append(user_data.get('followees', user_data.get('#Followees', 0)))
                elif feature == '#Posts':
                    user_features.append(user_data.get('posts', user_data.get('#Posts', 0)))
                elif feature == 'caption_length':
                    user_features.append(user_data.get('caption_length', len(user_data.get('caption', ''))))
                elif feature == 'num_hashtags':
                    hashtags = user_data.get('hashtags', '')
                    num_hashtags = len(hashtags.split('#')) - 1 if hashtags else 0
                    user_features.append(user_data.get('num_hashtags', num_hashtags))
                elif feature.startswith('tfidf_'):
                    try:
                        tfidf_idx = self.metadata['tfidf_columns'].index(feature)
                        if tfidf_idx < len(user_tfidf_dense):
                            user_features.append(user_tfidf_dense[tfidf_idx])
                        else:
                            user_features.append(0.0)
                    except (ValueError, IndexError):
                        user_features.append(0.0)
                else:
                    user_features.append(0.0)
            
            # Convert to numpy array and ensure correct shape
            user_features = np.array(user_features).reshape(1, -1)
            
            # Handle feature count mismatch
            if user_features.shape[1] != self.scaler.n_features_in_:
                if user_features.shape[1] < self.scaler.n_features_in_:
                    padding = np.zeros((1, self.scaler.n_features_in_ - user_features.shape[1]))
                    user_features = np.hstack([user_features, padding])
                else:
                    user_features = user_features[:, :self.scaler.n_features_in_]
            
            # Scale features
            user_features_scaled = self.scaler.transform(user_features)
            
            return user_features_scaled
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise
    
    def predict(self, user_data):
        """Make predictions for user data"""
        try:
            # Preprocess input
            user_features_scaled = self.preprocess_input(user_data)
            
            # Make predictions
            predictions = {}
            
            if 'engagement' in self.models:
                engagement_pred = self.models['engagement'].predict(user_features_scaled)[0]
                predictions['Engagement Rate'] = max(0, engagement_pred)
            
            if 'comments' in self.models:
                comments_pred = self.models['comments'].predict(user_features_scaled)[0]
                predictions['Comment Count'] = max(0, int(comments_pred))
            
            if 'sentiment' in self.models:
                sentiment_pred = self.models['sentiment'].predict(user_features_scaled)[0]
                predictions['Comment Sentiment'] = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}.get(sentiment_pred, 'Neutral')
            
            if 'weighted' in self.models:
                weighted_pred = self.models['weighted'].predict(user_features_scaled)[0]
                predictions['Sentiment-Weighted Engagement'] = max(0, weighted_pred)
            
            return predictions
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return {
                'Engagement Rate': 0.0,
                'Comment Count': 0,
                'Comment Sentiment': 'Neutral',
                'Sentiment-Weighted Engagement': 0.0,
                'Error': str(e)
            }

def main():
    # Parse arguments first
    parser = argparse.ArgumentParser(description='Instagram Engagement Predictor CLI')
    parser.add_argument('--followers', type=int, default=1000, help='Number of followers')
    parser.add_argument('--followees', type=int, default=500, help='Number of following')
    parser.add_argument('--posts', type=int, default=100, help='Number of posts')
    parser.add_argument('--caption', type=str, default='', help='Post caption')
    parser.add_argument('--hashtags', type=str, default='', help='Hashtags (space or # separated)')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory containing saved models')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--recommend', action='store_true', help='Get keyword/hashtag recommendations')
    parser.add_argument('--optimize', action='store_true', help='Optimize content for maximum engagement')
    parser.add_argument('--data-file', type=str, default='balanced_posts_with_sentiment_emotion_analysis.csv', help='CSV file for training recommendations')
    
    args = parser.parse_args()
    
    # Show usage instructions if no arguments provided
    import sys
    if len(sys.argv) == 1:
        print("\n🎯 INSTAGRAM ENGAGEMENT PREDICTOR")
        print("=" * 50)
        print("📖 Usage:")
        print("  Interactive Mode: python instagram_predictor_cli.py --interactive")
        print("  Command Line:     python instagram_predictor_cli.py --followers 5000 --caption 'Hello!'")
        print("  Recommendations:  python instagram_predictor_cli.py --followers 5000 --caption 'Hello!' --recommend")
        print("  Optimization:     python instagram_predictor_cli.py --followers 5000 --caption 'Hello!' --optimize")
        print("  Help:            python instagram_predictor_cli.py --help")
        print("\n💡 For best experience, use: python instagram_predictor_cli.py --interactive")
        print("=" * 50)
        return

    # Initialize predictor
    try:
        print("🔄 Loading models...")
        predictor = InstagramEngagementPredictor(args.models_dir)
        
        # Initialize keyword recommender if needed
        recommender = None
        if args.recommend or args.optimize or args.interactive:
            try:
                from keyword_recommender import KeywordRecommendationSystem
                print("📚 Loading recommendation system...")
                recommender = KeywordRecommendationSystem(args.data_file, args.models_dir)
                recommender.set_predictor(predictor)
                print("✅ Recommendation system ready!")
            except Exception as e:
                print(f"⚠️ Warning: Could not load recommendation system: {e}")
                print("📝 Proceeding without recommendations...")
                
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return
    
    if args.interactive:        # Interactive mode
        print("\n🎮 INSTAGRAM ENGAGEMENT PREDICTOR - INTERACTIVE MODE")
        print("=" * 60)
        print("💡 Tip: Press Enter to use default values, type 'quit' to exit")
        print("📝 Example hashtags: '#fashion #style #ootd' or 'fashion style ootd'")
        if recommender:
            print("🎯 NEW: Get AI-powered content optimization recommendations!")
        print()
        
        while True:
            try:
                print("\n" + "-" * 50)
                print("📊 Enter your Instagram post details:")
                print("-" * 50)
                
                # Get followers with validation
                while True:
                    try:
                        followers_input = input("👥 Number of followers (default 1000): ").strip()
                        followers = int(followers_input) if followers_input else 1000
                        if followers < 0:
                            print("❌ Please enter a positive number")
                            continue
                        break
                    except ValueError:
                        print("❌ Please enter a valid number")
                
                # Get followees with validation
                while True:
                    try:
                        followees_input = input("➕ Number of following (default 500): ").strip()
                        followees = int(followees_input) if followees_input else 500
                        if followees < 0:
                            print("❌ Please enter a positive number")
                            continue
                        break
                    except ValueError:
                        print("❌ Please enter a valid number")
                
                # Get posts with validation
                while True:
                    try:
                        posts_input = input("📸 Number of posts (default 100): ").strip()
                        posts = int(posts_input) if posts_input else 100
                        if posts < 0:
                            print("❌ Please enter a positive number")
                            continue
                        break
                    except ValueError:
                        print("❌ Please enter a valid number")
                
                # Get caption
                caption = input("✍️  Caption (or 'quit' to exit): ").strip()
                if caption.lower() == 'quit':
                    print("\n👋 Thank you for using Instagram Engagement Predictor!")
                    break
                
                # Get hashtags
                hashtags = input("🏷️  Hashtags (with or without #): ").strip()
                
                # Process hashtags - ensure they start with #
                if hashtags and not hashtags.startswith('#'):
                    # If user entered space-separated words, convert to hashtags
                    hashtag_words = hashtags.split()
                    hashtags = ' '.join([f'#{word}' if not word.startswith('#') else word for word in hashtag_words])
                
                user_data = {
                    'followers': followers,
                    'followees': followees,
                    'posts': posts,
                    'caption': caption,
                    'hashtags': hashtags
                }
                
                print(f"\n🔄 Processing your data...")
                print(f"   👥 Followers: {followers:,}")
                print(f"   ➕ Following: {followees:,}")
                print(f"   📸 Posts: {posts:,}")
                print(f"   ✍️  Caption: '{caption}'")
                print(f"   🏷️  Hashtags: '{hashtags}'")
                
                # Ask if user wants recommendations
                if recommender:
                    want_recommendations = input("\n🔍 Get keyword/hashtag recommendations? (y/n, default=n): ").strip().lower()
                    
                    if want_recommendations in ['y', 'yes']:
                        print("\n🔄 Analyzing content and generating recommendations...")
                        
                        try:
                            results = recommender.optimize_content(user_data)
                            
                            print("\n" + "🎯" * 25)
                            print("📈 CONTENT OPTIMIZATION RECOMMENDATIONS")
                            print("🎯" * 25)
                            
                            # Show improvements
                            baseline_eng = results['baseline']['engagement_rate']
                            optimized_eng = results['optimized']['engagement_rate']
                            improvement = results['improvement']['engagement_lift']
                            
                            print(f"📊 Current Predicted Engagement: {baseline_eng:.2%}")
                            print(f"🚀 Optimized Engagement: {optimized_eng:.2%}")
                            print(f"📈 Potential Improvement: +{improvement:.2%}")
                            
                            baseline_weighted = results['baseline']['weighted_engagement']
                            optimized_weighted = results['optimized']['weighted_engagement']
                            weighted_improvement = results['improvement']['weighted_lift']
                            
                            print(f"⚖️ Current Weighted Engagement: {baseline_weighted:.4f}")
                            print(f"🚀 Optimized Weighted Engagement: {optimized_weighted:.4f}")
                            print(f"📈 Weighted Improvement: +{weighted_improvement:.4f}")
                            
                            print(f"\n🏷️ RECOMMENDED HASHTAGS:")
                            for i, rec in enumerate(results['recommendations']['hashtags'][:5], 1):
                                if rec['combined_score'] > 0:
                                    print(f"  {i}. #{rec['hashtag']} (Improvement: +{rec['combined_score']:.4f})")
                            
                            print(f"\n📝 RECOMMENDED KEYWORDS:")
                            for i, rec in enumerate(results['recommendations']['keywords'][:5], 1):
                                if rec['impact_score'] > 0:
                                    print(f"  {i}. {rec['keyword']} (Impact: +{rec['impact_score']:.4f})")
                            
                            print(f"\n✨ OPTIMIZED CONTENT:")
                            print(f"Caption: {results['recommendations']['optimized_caption']}")
                            print(f"Hashtags: {results['recommendations']['optimized_hashtags']}")
                            
                            # Ask if user wants to use optimized content
                            use_optimized = input("\n🔄 Use optimized content for prediction? (y/n, default=n): ").strip().lower()
                            if use_optimized in ['y', 'yes']:
                                user_data['caption'] = results['recommendations']['optimized_caption']
                                user_data['hashtags'] = results['recommendations']['optimized_hashtags']
                                print("✅ Using optimized content!")
                        
                        except Exception as e:
                            print(f"❌ Error generating recommendations: {e}")
                
                # Make prediction
                predictions = predictor.predict(user_data)
                
                # Display results with enhanced formatting
                print("\n" + "🎯" * 25)
                print("📊 PREDICTED ENGAGEMENT METRICS")
                print("🎯" * 25)
                
                # Calculate estimated likes based on engagement rate and followers
                engagement_rate = predictions.get('Engagement Rate', 0)
                estimated_likes = int(followers * engagement_rate)
                
                print(f"📈 Engagement Rate............ {engagement_rate:.2%}")
                print(f"❤️  Estimated Likes........... {estimated_likes:,}")
                print(f"💬 Comment Count.............. {predictions.get('Comment Count', 0):,} comments")
                
                sentiment = predictions.get('Comment Sentiment', 'Neutral')
                emoji_map = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😔'}
                emoji = emoji_map.get(sentiment, '🤔')
                print(f"🎭 Comment Sentiment.......... {sentiment} {emoji}")
                
                weighted_engagement = predictions.get('Sentiment-Weighted Engagement', 0)
                print(f"⚖️  Weighted Engagement....... {weighted_engagement:.4f}")
                
                # Add performance insights
                print(f"\n💡 INSIGHTS:")
                if engagement_rate > 0.05:
                    print("✨ High engagement expected! Great content strategy.")
                elif engagement_rate > 0.02:
                    print("👍 Good engagement expected. Keep up the good work!")
                elif engagement_rate > 0.01:
                    print("📈 Moderate engagement. Consider optimizing your content.")
                else:
                    print("� Room for improvement. Try more engaging captions and hashtags.")
                
                if predictions.get('Comment Count', 0) > 50:
                    print("💬 High comment activity predicted - great for community building!")
                
                if 'Error' in predictions:
                    print(f"\n⚠️  Error: {predictions['Error']}")
                
                print("🎯" * 25)
                  # Ask if user wants to continue
                continue_choice = input("\n🔄 Would you like to predict another post? (y/n, default=y): ").strip().lower()
                if continue_choice in ['n', 'no', 'quit', 'exit']:
                    print("\n👋 Thank you for using Instagram Engagement Predictor!")
                    break
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    else:
        # Command line mode
        user_data = {
            'followers': args.followers,
            'followees': args.followees,
            'posts': args.posts,
            'caption': args.caption,
            'hashtags': args.hashtags
        }
        
        print(f"\n🔮 Making prediction for:")
        print(f"Followers: {args.followers:,}")
        print(f"Following: {args.followees:,}")
        print(f"Posts: {args.posts:,}")
        print(f"Caption: '{args.caption}'")
        print(f"Hashtags: '{args.hashtags}'")
        
        # Check if user wants recommendations
        if args.recommend or args.optimize:
            if recommender:
                print("\n🔄 Generating recommendations...")
                try:
                    results = recommender.optimize_content(user_data)
                    
                    print("\n" + "🎯" * 25)
                    print("📈 CONTENT OPTIMIZATION RECOMMENDATIONS")
                    print("🎯" * 25)
                    
                    baseline_eng = results['baseline']['engagement_rate']
                    optimized_eng = results['optimized']['engagement_rate']
                    improvement = results['improvement']['engagement_lift']
                    
                    print(f"📊 Current Predicted Engagement: {baseline_eng:.2%}")
                    print(f"🚀 Optimized Engagement: {optimized_eng:.2%}")
                    print(f"📈 Potential Improvement: +{improvement:.2%}")
                    
                    baseline_weighted = results['baseline']['weighted_engagement']
                    optimized_weighted = results['optimized']['weighted_engagement']
                    weighted_improvement = results['improvement']['weighted_lift']
                    
                    print(f"⚖️ Current Weighted Engagement: {baseline_weighted:.4f}")
                    print(f"🚀 Optimized Weighted Engagement: {optimized_weighted:.4f}")
                    print(f"📈 Weighted Improvement: +{weighted_improvement:.4f}")
                    
                    print(f"\n🏷️ RECOMMENDED HASHTAGS:")
                    for i, rec in enumerate(results['recommendations']['hashtags'][:5], 1):
                        if rec.get('combined_score', 0) > 0:
                            print(f"  {i}. #{rec['hashtag']} (Improvement: +{rec['combined_score']:.4f})")
                    
                    print(f"\n📝 RECOMMENDED KEYWORDS:")
                    for i, rec in enumerate(results['recommendations']['keywords'][:5], 1):
                        if rec.get('impact_score', 0) > 0:
                            print(f"  {i}. {rec['keyword']} (Impact: +{rec['impact_score']:.4f})")
                    
                    print(f"\n✨ OPTIMIZED CONTENT:")
                    print(f"Caption: {results['recommendations']['optimized_caption']}")
                    print(f"Hashtags: {results['recommendations']['optimized_hashtags']}")
                    
                    # Use optimized content for final prediction if optimize flag is set
                    if args.optimize:
                        user_data['caption'] = results['recommendations']['optimized_caption']
                        user_data['hashtags'] = results['recommendations']['optimized_hashtags']
                        print("\n✅ Using optimized content for final prediction!")
                        
                except Exception as e:
                    print(f"❌ Error generating recommendations: {e}")
            else:
                print("⚠️ Recommendation system not available")
        
        # Make prediction
        predictions = predictor.predict(user_data)
        
        # Display results
        print("\n" + "="*50)
        print("📊 PREDICTED ENGAGEMENT METRICS")
        print("="*50)
        
        for metric, value in predictions.items():
            if metric != 'Error':
                if metric == 'Engagement Rate':
                    print(f"📈 {metric:.<30} {value:.2%}")
                elif metric == 'Comment Count':
                    print(f"💬 {metric:.<30} {value:,} comments")
                elif metric == 'Comment Sentiment':
                    emoji_map = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😔'}
                    emoji = emoji_map.get(value, '🤔')
                    print(f"🎭 {metric:.<30} {value} {emoji}")
                elif metric == 'Sentiment-Weighted Engagement':
                    print(f"⚖️ {metric:.<30} {value:.2f}")
                else:
                    print(f"📋 {metric:.<30} {value}")
        
        if 'Error' in predictions:
            print(f"\n⚠️ Error: {predictions['Error']}")

if __name__ == "__main__":
    main()
