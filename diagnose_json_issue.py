#!/usr/bin/env python3
"""
Diagnostic script to analyze JSON parsing issues in sentiment scores
"""

import json
import os
import sys

def analyze_json_file():
    """Analyze the sentiment_scores.json file for parsing issues"""
    
    file_path = "outputs/sentiment_scores.json"
    
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} does not exist")
        return False
    
    # Get file size
    file_size = os.path.getsize(file_path)
    print(f"📄 File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    
    # Try to read and parse JSON
    try:
        print("🔍 Reading JSON file...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📏 Content length: {len(content):,} characters")
        
        # Check for common JSON issues
        print("\n🧐 Analyzing JSON structure...")
        
        # Check if it starts and ends properly
        print(f"Starts with: '{content[:50]}...'")
        print(f"Ends with: '...{content[-50:]}'")
        
        # Try to parse
        print("\n📋 Parsing JSON...")
        data = json.loads(content)
        
        print(f"✅ JSON parsed successfully!")
        print(f"📊 Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"🔢 Number of entries: {len(data):,}")
            
            # Sample a few entries
            sample_keys = list(data.keys())[:5]
            print(f"\n📝 Sample keys: {sample_keys}")
            
            # Check entry structure
            if sample_keys:
                sample_entry = data[sample_keys[0]]
                print(f"📋 Sample entry structure: {type(sample_entry)}")
                if isinstance(sample_entry, dict):
                    print(f"   Keys: {list(sample_entry.keys())}")
                
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"   Error at line {e.lineno}, column {e.colno}")
        print(f"   Position: {e.pos}")
        
        # Try to find the problematic area
        if hasattr(e, 'pos') and e.pos:
            start = max(0, e.pos - 100)
            end = min(len(content), e.pos + 100)
            problematic_area = content[start:end]
            print(f"\n🔍 Problematic area around position {e.pos}:")
            print(f"'{problematic_area}'")
        
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_chunked_loading():
    """Test loading the file in chunks to identify memory issues"""
    
    file_path = "outputs/sentiment_scores.json"
    
    print("\n🔄 Testing chunked loading...")
    
    try:
        # Try to load and sample the data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_entries = len(data)
        print(f"📊 Total entries: {total_entries:,}")
        
        # Create smaller samples
        sample_sizes = [100, 500, 1000, 2000, 5000]
        
        for sample_size in sample_sizes:
            if sample_size <= total_entries:
                print(f"\n🎯 Testing sample size: {sample_size:,}")
                
                # Random sample
                import random
                sample_keys = random.sample(list(data.keys()), sample_size)
                sample_data = {k: data[k] for k in sample_keys}
                
                # Convert to JSON string to test serialization
                json_str = json.dumps(sample_data)
                json_size = len(json_str)
                
                print(f"   ✅ Sample JSON size: {json_size:,} chars ({json_size / 1024:.1f} KB)")
                
                # Test if it can be parsed back
                reparsed = json.loads(json_str)
                print(f"   ✅ Sample can be parsed back: {len(reparsed):,} entries")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in chunked loading: {e}")
        return False

if __name__ == "__main__":
    print("🧪 JSON Diagnostic Tool for Sentiment Analysis\n")
    
    # Analyze the main JSON file
    json_ok = analyze_json_file()
    
    if json_ok:
        # Test chunked loading
        test_chunked_loading()
    
    print(f"\n{'✅ Analysis complete!' if json_ok else '❌ Issues found!'}")
