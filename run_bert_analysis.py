#!/usr/bin/env python3
"""
Run BERT sentiment analysis with exact specifications:
- Model: distilbert-base-uncased  
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 3
- Dataset: All comments without owner_id filtering
"""

import sys
import os
sys.path.append('src')

from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
import pandas as pd
import json

def main():
    print("🚀 Running BERT Sentiment Analysis with specified hyperparameters...")
    print("📋 Configuration:")
    print("   - Model: distilbert-base-uncased")
    print("   - Learning Rate: 2e-5")
    print("   - Batch Size: 16") 
    print("   - Epochs: 3")
    print("   - Dataset: All comments (no owner_id filtering)")
    
    # Load preprocessed data
    if not os.path.exists('outputs/preprocessed_data.csv'):
        print("❌ Preprocessed data not found!")
        return False
        
    df = pd.read_csv('outputs/preprocessed_data.csv')
    print(f"📊 Loaded {len(df):,} comments for analysis")
    
    # Initialize analyzer with specified hyperparameters
    try:
        analyzer = BERTSentimentAnalyzer(
            model_name='distilbert-base-uncased',
            learning_rate=2e-5,
            batch_size=16,
            epochs=3,
            use_fine_tuning=False
        )
        print("✅ BERT analyzer initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        return False
    
    # Run analysis
    try:
        print("🔄 Starting analysis...")
        results = analyzer.analyze_batch(df['comment_text'].tolist())
        print(f"✅ Analysis complete! Processed {len(results):,} comments")
        
        # Verify results
        if results:
            sample_key = list(results.keys())[0]
            sample_result = results[sample_key]
            print(f"📋 Sample result model: {sample_result.get('model_used', 'Unknown')}")
            
        print("💾 Results saved to outputs/sentiment_scores.json")
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ Analysis completed successfully!' if success else '❌ Analysis failed!'}")
