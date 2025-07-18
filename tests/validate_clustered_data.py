#!/usr/bin/env python3
"""
Quick validation test for the clustered data processor
"""

import sys
import os
import json
sys.path.append('src')

# Test basic functionality
def test_basic_functionality():
    print("=== Testing Clustered Data Processor ===")
    
    # Check if clustered data directory exists
    data_path = "data/clustered_data"
    print(f"1. Checking data path: {data_path}")
    if not os.path.exists(data_path):
        print("❌ Clustered data directory not found!")
        return False
    
    # List available clusters
    clusters = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"   Found clusters: {clusters}")
    
    if not clusters:
        print("❌ No cluster directories found!")
        return False
    
    # Test with the smallest cluster
    test_cluster = "followers_1000_to_2500"
    if test_cluster not in clusters:
        test_cluster = clusters[0]
    
    print(f"\n2. Testing cluster: {test_cluster}")
    
    cluster_path = os.path.join(data_path, test_cluster)
    user_folders = [d for d in os.listdir(cluster_path) if os.path.isdir(os.path.join(cluster_path, d))]
    print(f"   Found {len(user_folders)} user folders")
    
    if not user_folders:
        print("❌ No user folders found!")
        return False
    
    # Test reading one .info file
    test_user = user_folders[0]
    user_path = os.path.join(cluster_path, test_user)
    info_files = [f for f in os.listdir(user_path) if f.endswith('.info')]
    print(f"   User '{test_user}' has {len(info_files)} .info files")
    
    if not info_files:
        print("❌ No .info files found!")
        return False
    
    # Test JSON parsing
    test_file = os.path.join(user_path, info_files[0])
    print(f"   Testing file: {os.path.basename(test_file)}")
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("   ✅ Successfully parsed JSON")
        print(f"   Sample keys: {list(data.keys())[:5]}")
        
        # Check for required fields
        owner = data.get('owner', {})
        print(f"   Username: {owner.get('username', 'N/A')}")
        print(f"   Likes: {data.get('edge_media_preview_like', {}).get('count', 0)}")
        print(f"   Comments: {data.get('edge_media_to_parent_comment', {}).get('count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error parsing JSON: {e}")
        return False

def test_processor_import():
    print("\n3. Testing processor import and basic functionality")
    try:
        from preprocessing.clustered_data_processor import ClusteredDataProcessor
        print("   ✅ Import successful")
        
        processor = ClusteredDataProcessor()
        clusters = processor.get_cluster_names()
        print(f"   ✅ Found {len(clusters)} clusters via processor")
        
        return True
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False

def check_existing_outputs():
    print("\n4. Checking existing processed outputs")
    outputs_dir = "outputs"
    if os.path.exists(outputs_dir):
        csv_files = [f for f in os.listdir(outputs_dir) if f.endswith('.csv')]
        print(f"   Found {len(csv_files)} CSV files in outputs:")
        for csv_file in csv_files:
            file_path = os.path.join(outputs_dir, csv_file)
            try:
                import pandas as pd
                df = pd.read_csv(file_path)
                print(f"   - {csv_file}: {len(df)} rows, {len(df.columns)} columns")
            except:
                print(f"   - {csv_file}: [Error reading file]")
    else:
        print("   No outputs directory found")

if __name__ == "__main__":
    success = True
    
    success &= test_basic_functionality()
    success &= test_processor_import()
    check_existing_outputs()
    
    print(f"\n{'='*50}")
    if success:
        print("🎉 Basic validation PASSED!")
        print("The clustered data processor should work correctly.")
    else:
        print("💥 Validation FAILED!")
        print("There are issues with the clustered data setup.")
    print("="*50)
