#!/bin/bash
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
