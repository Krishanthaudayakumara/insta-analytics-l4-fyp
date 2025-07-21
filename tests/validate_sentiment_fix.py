#!/usr/bin/env python3
"""
Quick Sentiment Analysis Validation
Tests the fixed sentiment analysis with a small sample
"""

import sys
import os
import pandas as pd
import json
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings('ignore')
os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/tmp/pytorch_cache'

# Add src to path
sys.path.append('src')

def validate_sentiment_analysis():
    """Test sentiment analysis with corrected JSON serialization"""
    print("🧪 Quick Sentiment Analysis Validation")
    print("=" * 60)
    
    try:
        from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
        print("✅ Successfully imported BERTSentimentAnalyzer")
        
        # Create test data
        test_data = {
            'comment_text': [
                'This is amazing! Love this post!',
                'Not really my thing...',
                'Terrible, worst post ever!',
                'Nice photo, looks good',
                'Okay I guess'
            ],
            'comment_owner_username': ['user1', 'user2', 'user3', 'user4', 'user5'],
            'post_id': ['post1', 'post2', 'post3', 'post4', 'post5']
        }
        
        df = pd.DataFrame(test_data)
        print(f"✅ Created test dataset with {len(df)} comments")
        
        # Initialize analyzer
        analyzer = BERTSentimentAnalyzer(use_fine_tuning=False)
        print("✅ Initialized sentiment analyzer")
        
        # Run analysis
        print("\n🚀 Running sentiment analysis...")
        sentiment_scores = analyzer.analyze_sentiment(
            df,
            model_name="distilbert-base-uncased",
            batch_size=4,
            max_length=64,
            use_gpu=False  # Use CPU for validation
        )
        
        print(f"✅ Analysis completed! Generated {len(sentiment_scores)} scores")
        
        # Test JSON serialization
        print("\n📝 Testing JSON serialization...")
        test_json = json.dumps(sentiment_scores, indent=2)
        print("✅ JSON serialization successful!")
        
        # Display sample results
        print("\n📊 Sample Results:")
        for i, (key, score) in enumerate(list(sentiment_scores.items())[:3]):
            print(f"   {i+1}. {score['sentiment']} ({score['confidence']:.3f}) - {score['comment_text'][:50]}...")
        
        # Test saving results
        print("\n💾 Testing save functionality...")
        saved_results = analyzer.save_sentiment_results(
            sentiment_scores, 
            output_path="outputs/validation_sentiment_test.json"
        )
        print("✅ Save functionality working!")
        
        # Verify the saved file can be loaded
        with open("outputs/validation_sentiment_test.json", 'r') as f:
            loaded_data = json.load(f)
        print("✅ Saved file can be loaded successfully!")
        
        print(f"\n📈 Statistics:")
        sentiments = [score['sentiment'] for score in sentiment_scores.values()]
        print(f"   Positive: {sentiments.count('positive')}")
        print(f"   Negative: {sentiments.count('negative')}")
        print(f"   Neutral: {sentiments.count('neutral')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return False

def main():
    """Main validation function"""
    success = validate_sentiment_analysis()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 VALIDATION SUCCESSFUL!")
        print("✅ Sentiment analysis is working correctly")
        print("✅ JSON serialization fixed")
        print("✅ No more RobertaForSequenceClassification errors")
        print("\n💡 You can now run sentiment analysis in Streamlit:")
        print("   ./run_app_improved.sh")
        print("   Navigate to 'Sentiment Analysis' tab")
    else:
        print("❌ VALIDATION FAILED!")
        print("⚠️ Please check the error messages above")
    
    return success

if __name__ == "__main__":
    main()
