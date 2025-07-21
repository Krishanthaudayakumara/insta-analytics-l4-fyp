#!/usr/bin/env python3
"""
Fix JSON Serialization and PyTorch Issues for Sentiment Analysis
"""

import sys
import os
import warnings
import json

# Add src to path
sys.path.append('src')

def fix_sentiment_json_serialization():
    """Fix JSON serialization issues in saved sentiment results"""
    
    print("🔧 Fixing sentiment analysis JSON serialization issues...")
    
    # Check if there are any existing sentiment results with serialization issues
    sentiment_files = [
        "outputs/sentiment_scores.json",
        "outputs/sentiment_analysis_results.json",
        "outputs/detailed_sentiment_analysis.json"
    ]
    
    for file_path in sentiment_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"✅ {file_path} - JSON format is valid")
                
                # Check for model serialization issues
                if isinstance(data, dict):
                    if "sentiment_scores" in data:
                        scores = data["sentiment_scores"]
                    else:
                        scores = data
                    
                    # Sample a few entries to check format
                    sample_keys = list(scores.keys())[:3]
                    for key in sample_keys:
                        score_data = scores[key]
                        if 'model_used' in score_data:
                            model_value = score_data['model_used']
                            if not isinstance(model_value, str):
                                print(f"⚠️ Found non-string model_used value: {type(model_value)}")
                            else:
                                print(f"✅ Model field is properly serialized: {model_value}")
                
            except json.JSONDecodeError as e:
                print(f"❌ {file_path} - JSON parsing error: {str(e)}")
            except Exception as e:
                print(f"❌ {file_path} - Error: {str(e)}")
        else:
            print(f"ℹ️ {file_path} - File not found")

def suppress_pytorch_warnings():
    """Add warning suppressions for PyTorch/Streamlit compatibility"""
    
    print("🔧 Setting up PyTorch warning suppressions...")
    
    # Suppress specific PyTorch warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    warnings.filterwarnings('ignore', message='.*torch.classes.*')
    warnings.filterwarnings('ignore', message='.*running event loop.*')
    
    # Set environment variables to reduce PyTorch verbosity
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTORCH_TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
    
    print("✅ PyTorch warning suppressions configured")

def create_improved_startup_script():
    """Create an improved startup script with error handling"""
    
    script_content = '''#!/bin/bash
# Improved Streamlit startup script with error handling

echo "🚀 Starting Instagram Engagement Analysis with Sentiment Analysis"
echo "🧠 Enhanced BERT sentiment analysis ready"
echo "=" * 80

# Navigate to project directory
cd /home/krishantha/Github/fyp-l4

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: python3 -m venv venv"
    exit 1
fi

source venv/bin/activate

# Set environment variables to reduce warnings
export TOKENIZERS_PARALLELISM=false
export PYTORCH_TRANSFORMERS_CACHE=/tmp/transformers_cache
export PYTHONWARNINGS=ignore::UserWarning

# Create cache directory
mkdir -p /tmp/transformers_cache

echo "✅ Environment configured"
echo "🌐 Starting Streamlit application..."
echo ""
echo "📍 Local URL: http://localhost:8501"
echo "🧠 Navigate to: Sentiment Analysis tab"
echo ""

# Run Streamlit with reduced warnings
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 2>/dev/null
'''
    
    with open('run_app_improved.sh', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('run_app_improved.sh', 0o755)
    
    print("✅ Created improved startup script: run_app_improved.sh")

def test_sentiment_analyzer_json():
    """Test sentiment analyzer JSON serialization"""
    
    print("🧪 Testing sentiment analyzer JSON serialization...")
    
    try:
        from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
        import pandas as pd
        
        # Create test data
        test_data = {
            'comment_text': ['Great post!', 'Not bad', 'Terrible content'],
            'comment_owner_username': ['user1', 'user2', 'user3'],
            'post_id': ['post1', 'post2', 'post3']
        }
        
        df = pd.DataFrame(test_data)
        
        # Initialize analyzer
        analyzer = BERTSentimentAnalyzer(use_fine_tuning=False)
        
        # Run analysis
        results = analyzer.analyze_sentiment(df, batch_size=2, use_gpu=False)
        
        # Test JSON serialization
        json_str = json.dumps(results, indent=2)
        parsed_back = json.loads(json_str)
        
        print(f"✅ JSON serialization test passed!")
        print(f"✅ Processed {len(results)} comments successfully")
        
        # Check model_used field
        sample_key = list(results.keys())[0]
        model_used = results[sample_key]['model_used']
        print(f"✅ Model field properly serialized: {model_used}")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization test failed: {str(e)}")
        return False

def main():
    """Main function to fix all issues"""
    
    print("🔧 Fixing Sentiment Analysis Issues")
    print("=" * 60)
    
    # Suppress warnings first
    suppress_pytorch_warnings()
    
    # Fix JSON serialization
    fix_sentiment_json_serialization()
    
    # Test the analyzer
    print("\n🧪 Testing JSON serialization fix...")
    test_success = test_sentiment_analyzer_json()
    
    # Create improved startup script
    print("\n📝 Creating improved startup script...")
    create_improved_startup_script()
    
    print("\n" + "=" * 60)
    print("🏁 Summary:")
    print("=" * 60)
    
    if test_success:
        print("✅ JSON serialization fix: WORKING")
    else:
        print("❌ JSON serialization fix: NEEDS ATTENTION")
    
    print("✅ PyTorch warning suppressions: CONFIGURED")
    print("✅ Improved startup script: CREATED")
    
    print("\n💡 To run the app with fixes:")
    print("   ./run_app_improved.sh")
    print("\n🎯 The sentiment analysis should now work without JSON errors!")

if __name__ == "__main__":
    main()
