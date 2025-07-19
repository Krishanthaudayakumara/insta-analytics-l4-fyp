#!/usr/bin/env python3
import sys
import os
import json
import pandas as pd

# Add current directory to path
sys.path.append('.')

def test_json_loading():
    """Test loading the evaluation JSON files"""
    print("=== Testing JSON Loading ===")
    
    for target in ['engagement', 'likes', 'comments']:
        filename = f'outputs/model_evaluation_{target}.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"✅ {target}: {len(data)} sessions, latest has {len(data[-1]['evaluation_results'])} models")
        except Exception as e:
            print(f"❌ {target}: {e}")

def test_dashboard_functions():
    """Test the dashboard functions"""
    print("\n=== Testing Dashboard Functions ===")
    
    try:
        from model_evaluation_dashboard import load_evaluation_results, create_evaluation_comparison_df
        
        # Test loading results
        engagement_results = load_evaluation_results('engagement')
        likes_results = load_evaluation_results('likes')
        comments_results = load_evaluation_results('comments')
        
        print(f"✅ Loaded {len(engagement_results)} engagement, {len(likes_results)} likes, {len(comments_results)} comments models")
        
        # Test comparison dataframe
        comparison_df = create_evaluation_comparison_df()
        print(f"✅ Created comparison DataFrame with {len(comparison_df)} rows")
        
        if not comparison_df.empty:
            print(f"   Columns: {list(comparison_df.columns)}")
            print(f"   Targets: {comparison_df['Target'].unique().tolist()}")
            print(f"   Models: {comparison_df['Model'].unique().tolist()}")
            
        return True
        
    except Exception as e:
        print(f"❌ Dashboard functions error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_json_loading()
    success = test_dashboard_functions()
    
    if success:
        print("\n🎉 All tests passed! Dashboard is ready to use.")
    else:
        print("\n❌ Some tests failed.")
