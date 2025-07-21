#!/bin/bash
# Test Enhanced Sentiment Analysis with Virtual Environment

echo "🧠 Testing Enhanced Sentiment Analysis Implementation"
echo "📊 Using Virtual Environment for Python Dependencies"
echo "=" * 80

# Navigate to project directory
cd /home/krishantha/Github/fyp-l4

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Verify activation
echo "✅ Virtual environment activated:"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Install/update requirements
echo "📦 Installing requirements..."
pip install --upgrade pip > /dev/null 2>&1

# Install essential packages for sentiment analysis
echo "🧠 Installing sentiment analysis dependencies..."
pip install torch transformers pandas numpy streamlit plotly scikit-learn tqdm > /dev/null 2>&1

# Verify installations
echo ""
echo "🔍 Verifying package installations:"
python -c "
import torch
print(f'✅ PyTorch: {torch.__version__}')
import transformers
print(f'✅ Transformers: {transformers.__version__}')
import pandas
print(f'✅ Pandas: {pandas.__version__}')
import streamlit
print(f'✅ Streamlit: {streamlit.__version__}')
import plotly
print(f'✅ Plotly: {plotly.__version__}')
"

echo ""
echo "🚀 Running sentiment analysis tests..."

# Run the sentiment analysis test
python test_sentiment_improvements_venv.py

echo ""
echo "🎯 Test completed!"
echo "💡 If tests passed, your sentiment analysis improvements are working correctly."
echo "🌐 You can now run the Streamlit app with: ./run_app_with_venv.sh"
