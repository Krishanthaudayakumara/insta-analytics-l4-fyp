#!/usr/bin/env python3
"""
Demonstration script for enhanced sentiment analysis
Shows how to use the improved BERT sentiment analyzer
"""

import pandas as pd
import json
import os
import sys

# Add src to path
sys.path.insert(0, 'src')

def demo_basic_usage():
    """Demonstrate basic usage of the enhanced sentiment analyzer"""
    print("🧠 Basic Sentiment Analysis Demo")
    print("=" * 40)
    
    from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
    
    # Initialize analyzer
    analyzer = BERTSentimentAnalyzer(use_fine_tuning=False)
    
    # Create sample Instagram-like dataset
    sample_data = {
        'post_id': ['12345', '12346', '12347', '12348', '12349'],
        'comment_text': [
            'Love this post! Amazing content 😍',
            'Not really my style, but okay',
            'This is absolutely terrible 😞',
            'Great work! Keep it up! 🔥',
            'Could be better tbh'
        ],
        'comment_owner_username': ['alice_user', 'bob_commenter', 'charlie_critic', 'diana_fan', 'eve_honest'],
        'owner_id': ['user1', 'user1', 'user2', 'user2', 'user3']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"📊 Analyzing {len(df)} sample comments...")
    
    # Run sentiment analysis
    results = analyzer.analyze_sentiment(
        df,
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        batch_size=16,
        max_length=128,
        use_gpu=False
    )
    
    # Display results
    print("\n📊 Results:")
    for comment_key, data in results.items():
        print(f"   {data['comment_owner_username']}: {data['sentiment']} ({data['confidence']:.3f})")
        print(f"      Comment: {data['comment_text']}")
        print()
    
    return results

def demo_large_dataset():
    """Demonstrate chunked processing for large datasets"""
    print("📦 Large Dataset Processing Demo")
    print("=" * 40)
    
    from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
    
    # Initialize analyzer with chunking
    analyzer = BERTSentimentAnalyzer(use_fine_tuning=False)
    analyzer.chunk_size = 100  # Process in chunks of 100
    
    # Create larger sample dataset
    comments = [
        "This is amazing!", "Love it!", "Great post!", "Awesome content!",
        "Not bad", "It's okay", "Average post", "Could be better",
        "Terrible", "Hate this", "Worst ever", "Disappointing"
    ]
    
    large_data = {
        'post_id': [f'post_{i//4}' for i in range(1000)],
        'comment_text': [comments[i % len(comments)] for i in range(1000)],
        'comment_owner_username': [f'user_{i%50}' for i in range(1000)]
    }
    
    df = pd.DataFrame(large_data)
    print(f"📊 Processing {len(df)} comments in chunks...")
    
    # Run analysis
    results = analyzer.analyze_sentiment(
        df,
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        batch_size=32,
        max_length=64,
        use_gpu=False
    )
    
    # Show summary
    sentiments = [r['sentiment'] for r in results.values()]
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    print(f"\n📊 Summary of {len(results)} processed comments:")
    for sentiment, count in sentiment_counts.items():
        print(f"   {sentiment.title()}: {count} ({count/len(results)*100:.1f}%)")
    
    return results

def demo_error_handling():
    """Demonstrate error handling capabilities"""
    print("🛡️ Error Handling Demo")
    print("=" * 40)
    
    from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
    
    analyzer = BERTSentimentAnalyzer()
    
    # Test 1: Missing required columns
    print("1. Testing missing column handling...")
    try:
        invalid_df = pd.DataFrame({'wrong_column': ['test']})
        analyzer.analyze_sentiment(invalid_df)
        print("   ❌ Should have failed")
    except KeyError as e:
        print(f"   ✅ Correctly caught error: {e}")
    
    # Test 2: Empty dataset
    print("\n2. Testing empty dataset handling...")
    empty_df = pd.DataFrame({
        'comment_text': [None, '', '   '],
        'comment_owner_username': ['user1', 'user2', 'user3']
    })
    
    results = analyzer.analyze_sentiment(empty_df)
    print(f"   ✅ Handled empty dataset: {len(results)} results")
    
    # Test 3: Mixed valid/invalid data
    print("\n3. Testing mixed data handling...")
    mixed_df = pd.DataFrame({
        'comment_text': ['Good post!', None, '', 'Bad content', '   whitespace   '],
        'comment_owner_username': ['user1', 'user2', 'user3', 'user4', 'user5']
    })
    
    results = analyzer.analyze_sentiment(mixed_df)
    print(f"   ✅ Processed mixed data: {len(results)} valid results")

def demo_enhanced_output():
    """Demonstrate enhanced output format"""
    print("📊 Enhanced Output Format Demo")
    print("=" * 40)
    
    from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
    
    analyzer = BERTSentimentAnalyzer()
    
    # Sample data
    df = pd.DataFrame({
        'post_id': ['post123', 'post456'],
        'comment_text': ['Love this! 😍', 'Not impressed 😕'],
        'comment_owner_username': ['happy_user', 'critical_user']
    })
    
    results = analyzer.analyze_sentiment(df)
    
    # Show enhanced output format
    print("📋 Enhanced output includes:")
    sample_result = list(results.values())[0]
    for key, value in sample_result.items():
        print(f"   {key}: {value}")
    
    # Save with enhanced format
    analyzer.save_sentiment_results(results, output_path="outputs/demo_enhanced_results.json")
    
    # Load and show metadata
    with open("outputs/demo_enhanced_results.json", 'r') as f:
        saved_data = json.load(f)
    
    print(f"\n💾 Saved data includes:")
    print(f"   📊 Sentiment scores: {len(saved_data['sentiment_scores'])} comments")
    print(f"   📋 Metadata: {list(saved_data['metadata'].keys())}")
    print(f"   📈 Analysis summary: {list(saved_data['analysis_summary'].keys())}")

def main():
    """Run all demonstrations"""
    print("🚀 Enhanced Sentiment Analysis Demonstration")
    print("=" * 60)
    
    # Ensure outputs directory exists
    os.makedirs('outputs', exist_ok=True)
    
    try:
        # Run demonstrations
        print("\n1️⃣ Basic Usage:")
        demo_basic_usage()
        
        print("\n2️⃣ Large Dataset Processing:")
        demo_large_dataset()
        
        print("\n3️⃣ Error Handling:")
        demo_error_handling()
        
        print("\n4️⃣ Enhanced Output:")
        demo_enhanced_output()
        
        print("\n" + "=" * 60)
        print("🎉 All demonstrations completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("   ✅ Robust error handling and data validation")
        print("   ✅ Memory-efficient processing for large datasets")
        print("   ✅ Enhanced output format with comprehensive metadata")
        print("   ✅ Graceful handling of missing or invalid data")
        print("   ✅ Comprehensive statistics and analysis summaries")
        
        print("\n🔧 Integration with Streamlit:")
        print("   • Use the enhanced UI component in your Streamlit app")
        print("   • Configure hyperparameters through the interface")
        print("   • View comprehensive visualizations and statistics")
        print("   • Export results in multiple formats")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install transformers torch pandas numpy")
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")

if __name__ == "__main__":
    main()
