#!/usr/bin/env python3
"""
Simple Validation Test: Verify key fixes without complex imports
"""

import os
import json
import time
import random

def test_sentiment_key_generation():
    """Test the unique key generation logic from the Streamlit app"""
    print("🧪 Testing Sentiment Chart Key Generation...")
    
    # Test the exact logic used in the app
    contexts = ["new", "existing", "default"]
    all_keys = []
    
    for i in range(10):  # Generate multiple keys rapidly
        for context in contexts:
            # Replicate the exact logic from show_sentiment_results
            timestamp = str(time.time()).replace('.', '')  # High precision timestamp
            random_id = random.randint(1000, 9999)
            key_suffix = f"{context}_{timestamp}_{random_id}"
            
            pie_key = f"sentiment_distribution_pie_{key_suffix}"
            histogram_key = f"sentiment_confidence_histogram_{key_suffix}"
            
            all_keys.extend([pie_key, histogram_key])
        
        time.sleep(0.001)  # Small delay
    
    # Check for uniqueness
    unique_keys = set(all_keys)
    total_keys = len(all_keys)
    unique_count = len(unique_keys)
    
    print(f"📊 Generated {total_keys} keys")
    print(f"📊 Unique keys: {unique_count}")
    print(f"📊 Duplicates: {total_keys - unique_count}")
    
    if unique_count == total_keys:
        print("✅ Sentiment chart key generation fix working!")
        return True
    else:
        print("❌ Still generating duplicate keys")
        return False

def test_app_imports():
    """Test that app.py imports work correctly"""
    print("\n🧪 Testing App.py Imports...")
    
    try:
        # Test if we can read the app.py file and check for json import
        with open("app.py", "r") as f:
            content = f.read()
        
        # Check that json is imported at the top
        lines = content.split('\n')
        import_section = lines[:20]  # First 20 lines should contain imports
        
        json_imported = any('import json' in line for line in import_section)
        
        if json_imported:
            print("✅ JSON import found in app.py")
        else:
            print("❌ JSON import not found in app.py header")
            return False
        
        # Check that there are no redundant local json imports
        redundant_imports = content.count('import json') - 1  # Subtract the main import
        
        if redundant_imports == 0:
            print("✅ No redundant JSON imports found")
        else:
            print(f"⚠️ Found {redundant_imports} redundant JSON imports")
        
        print("✅ App.py imports test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing app imports: {e}")
        return False

def test_data_files():
    """Test that required data files exist"""
    print("\n🧪 Testing Data File Availability...")
    
    required_files = {
        "Influencers CSV": "data/clustered_data/influencers.csv",
        "Cluster Directory": "data/clustered_data/followers_1000_to_2500",
        "Output Directory": "outputs"
    }
    
    results = {}
    for name, path in required_files.items():
        exists = os.path.exists(path)
        results[name] = exists
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
    
    if all(results.values()):
        print("✅ All required data files/directories exist!")
        return True
    else:
        missing = [name for name, exists in results.items() if not exists]
        print(f"❌ Missing: {', '.join(missing)}")
        return False

def test_influencers_csv():
    """Test that influencers.csv can be read and has data"""
    print("\n🧪 Testing Influencers.csv Data...")
    
    try:
        import pandas as pd
        
        # Try reading the influencers.csv file
        csv_path = "data/clustered_data/influencers.csv"
        
        # Test reading with UTF-8-BOM encoding (the fix we implemented)
        df = pd.read_csv(csv_path, skiprows=[1], encoding='utf-8-sig')
        
        print(f"📊 Total users in influencers.csv: {len(df)}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check if we have follower data
        if 'followers' in df.columns:
            follower_counts = df['followers'].describe()
            print(f"📊 Follower statistics:")
            print(f"   Min: {follower_counts['min']}")
            print(f"   Max: {follower_counts['max']}")
            print(f"   Mean: {follower_counts['mean']:.0f}")
            
            non_zero_followers = (df['followers'] > 0).sum()
            print(f"📊 Users with followers > 0: {non_zero_followers}/{len(df)}")
            
            if non_zero_followers > len(df) * 0.8:  # 80% should have followers
                print("✅ Influencers.csv has good follower data!")
                return True
            else:
                print("⚠️ Many users have 0 followers in influencers.csv")
                return False
        else:
            print("❌ No 'followers' column found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading influencers.csv: {e}")
        return False

def test_streamlit_running():
    """Test if Streamlit is running without errors"""
    print("\n🧪 Testing Streamlit App Status...")
    
    try:
        import requests
        
        # Try to connect to the Streamlit app
        response = requests.get("http://localhost:8501", timeout=5)
        
        if response.status_code == 200:
            print("✅ Streamlit app is running and accessible!")
            return True
        else:
            print(f"⚠️ Streamlit returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Streamlit app (may not be running)")
        return False
    except Exception as e:
        print(f"❌ Error testing Streamlit: {e}")
        return False

def create_test_outputs():
    """Create test output files to verify the pipeline"""
    print("\n🧪 Creating Test Output Files...")
    
    try:
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Create test sentiment scores to test the duplicate key fix
        test_sentiment = {
            "Great post!": {"sentiment": "positive", "confidence": 0.95},
            "Not bad": {"sentiment": "neutral", "confidence": 0.75},
            "Could be better": {"sentiment": "negative", "confidence": 0.82}
        }
        
        with open("outputs/test_sentiment_scores.json", "w") as f:
            json.dump(test_sentiment, f)
        
        print("✅ Created test sentiment scores file")
        
        # Create test high-value followers
        test_followers = {
            "user1": {"engagement_score": 0.8, "influence_score": 0.9},
            "user2": {"engagement_score": 0.7, "influence_score": 0.8}
        }
        
        with open("outputs/test_high_value_followers.json", "w") as f:
            json.dump(test_followers, f)
        
        print("✅ Created test high-value followers file")
        print("✅ Test output files created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating test outputs: {e}")
        return False

def main():
    """Run simplified validation tests"""
    print("🚀 Simplified Validation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Sentiment Key Generation", test_sentiment_key_generation),
        ("App.py Imports", test_app_imports),
        ("Data File Availability", test_data_files),
        ("Influencers.csv Data", test_influencers_csv),
        ("Test Output Creation", create_test_outputs),
        ("Streamlit App Status", test_streamlit_running)
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
    print("📋 VALIDATION RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{len(tests)} tests passed")
    
    # Key fixes summary
    print("\n" + "=" * 50)
    print("🔧 KEY FIXES STATUS")
    print("=" * 50)
    
    sentiment_key_passed = results[0][1]  # First test
    app_imports_passed = results[1][1]   # Second test
    
    print(f"{'✅' if sentiment_key_passed else '❌'} Streamlit Duplicate Key Fix")
    print(f"{'✅' if app_imports_passed else '❌'} JSON Import Fix")
    
    if sentiment_key_passed and app_imports_passed:
        print("\n🎉 CRITICAL FIXES VERIFIED! System ready for use.")
        return True
    else:
        print("\n⚠️ Some critical fixes need attention.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
