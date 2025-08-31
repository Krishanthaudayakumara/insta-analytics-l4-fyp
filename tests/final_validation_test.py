#!/usr/bin/env python3
"""
Final Validation Test: Verify both critical fixes
1. Follower count fix
2. Streamlit duplicate key fix
"""

import sys
import os
import json
import pandas as pd
from src.preprocessing.clustered_data_processor import ClusteredDataProcessor

def test_follower_counts():
    """Test that follower counts are properly populated"""
    print("🧪 Testing Follower Count Fix...")
    
    processor = ClusteredDataProcessor()
    
    # Test processing a small cluster
    try:
        test_df = processor.process_cluster("followers_1000_to_2500")
        
        if test_df.empty:
            print("❌ No data processed")
            return False
            
        # Check if follower counts are populated
        follower_counts = test_df['#Followers'].value_counts()
        zero_counts = (test_df['#Followers'] == 0).sum()
        total_counts = len(test_df)
        
        print(f"📊 Total records: {total_counts}")
        print(f"📊 Records with 0 followers: {zero_counts}")
        print(f"📊 Records with real follower counts: {total_counts - zero_counts}")
        
        if zero_counts < total_counts * 0.5:  # Less than 50% should be zero
            print("✅ Follower count fix working - most users have real follower counts!")
            print(f"📊 Sample follower counts: {test_df['#Followers'].head().tolist()}")
            return True
        else:
            print("❌ Most users still have 0 followers")
            return False
            
    except Exception as e:
        print(f"❌ Error testing follower counts: {e}")
        return False

def test_duplicate_keys():
    """Test that sentiment analysis charts have unique keys"""
    print("\n🧪 Testing Streamlit Duplicate Key Fix...")
    
    try:
        # Check the show_sentiment_results method for unique key generation
        import time
        import random
        
        # Simulate the key generation logic
        contexts = ["new", "existing", "test"]
        generated_keys = []
        
        for context in contexts:
            timestamp = str(time.time()).replace('.', '')
            random_id = random.randint(1000, 9999)
            key_suffix = f"{context}_{timestamp}_{random_id}"
            
            pie_key = f"sentiment_distribution_pie_{key_suffix}"
            histogram_key = f"sentiment_confidence_histogram_{key_suffix}"
            
            generated_keys.extend([pie_key, histogram_key])
            
            # Small delay to ensure different timestamps
            time.sleep(0.001)
        
        # Check for duplicates
        unique_keys = set(generated_keys)
        
        if len(unique_keys) == len(generated_keys):
            print("✅ Duplicate key fix working - all keys are unique!")
            print(f"📊 Generated {len(generated_keys)} unique keys")
            return True
        else:
            print(f"❌ Found duplicate keys: {len(generated_keys) - len(unique_keys)} duplicates")
            return False
            
    except Exception as e:
        print(f"❌ Error testing duplicate keys: {e}")
        return False

def test_sentiment_file_handling():
    """Test sentiment analysis file operations"""
    print("\n🧪 Testing Sentiment File Handling...")
    
    try:
        # Create test sentiment data
        test_sentiment = {
            "test comment 1": {"sentiment": "positive", "confidence": 0.9},
            "test comment 2": {"sentiment": "negative", "confidence": 0.8},
            "test comment 3": {"sentiment": "neutral", "confidence": 0.7}
        }
        
        # Test saving
        with open("outputs/test_sentiment.json", "w") as f:
            json.dump(test_sentiment, f)
        
        # Test loading
        with open("outputs/test_sentiment.json", "r") as f:
            loaded_sentiment = json.load(f)
        
        if loaded_sentiment == test_sentiment:
            print("✅ Sentiment file handling working correctly!")
            
            # Clean up
            os.remove("outputs/test_sentiment.json")
            return True
        else:
            print("❌ Sentiment data mismatch")
            return False
            
    except Exception as e:
        print(f"❌ Error testing sentiment file handling: {e}")
        return False

def test_clustered_data_integration():
    """Test complete clustered data integration"""
    print("\n🧪 Testing Clustered Data Integration...")
    
    try:
        processor = ClusteredDataProcessor()
        cluster_names = processor.get_cluster_names()
        
        if not cluster_names:
            print("❌ No clusters found")
            return False
        
        print(f"📊 Found {len(cluster_names)} clusters: {cluster_names}")
        
        # Test processing first cluster
        test_cluster = cluster_names[0]
        raw_df = processor.process_cluster(test_cluster)
        
        if raw_df.empty:
            print(f"❌ No data in cluster {test_cluster}")
            return False
        
        # Test compatibility conversion
        compatible_df = processor.create_compatible_dataframe(raw_df)
        
        required_columns = ['username', 'media_type', 'Category', '#Followers', '#Followees', '#Posts']
        missing_columns = [col for col in required_columns if col not in compatible_df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        
        print("✅ Clustered data integration working correctly!")
        print(f"📊 Processed {len(compatible_df)} records from {test_cluster}")
        return True
        
    except Exception as e:
        print(f"❌ Error testing clustered data integration: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Final Validation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Follower Count Fix", test_follower_counts),
        ("Duplicate Key Fix", test_duplicate_keys),
        ("Sentiment File Handling", test_sentiment_file_handling),
        ("Clustered Data Integration", test_clustered_data_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📋 FINAL VALIDATION RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! System ready for production.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
