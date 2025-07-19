#!/bin/bash
# Run Instagram Engagement Prediction System with Virtual Environment

echo "🚀 Starting Instagram Engagement Prediction System"
echo "📊 Dataset-Wide Analysis Ready"
echo "=" * 60

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
echo "Streamlit: $(which streamlit)"

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Checking requirements..."
    pip install -r requirements.txt > /dev/null 2>&1
fi

# Show system status
echo ""
echo "🎯 System Status:"
python -c "
import sys
sys.path.append('src')
try:
    from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
    from follower_selection.network_visualizer import FollowerNetworkVisualizer
    print('✅ Dataset analysis modules ready')
except Exception as e:
    print(f'❌ Module error: {e}')

import os
if os.path.exists('outputs/preprocessed_data.csv'):
    import pandas as pd
    df = pd.read_csv('outputs/preprocessed_data.csv')
    print(f'✅ Data ready: {len(df):,} records')
else:
    print('⚠️ Preprocessed data not found - run preprocessing first')
"

echo ""
echo "🌐 Starting Streamlit Application..."
echo "Navigate to: http://localhost:8501"
echo "Go to: 🌐 Dataset-Wide Analysis"
echo ""
echo "🎯 Features Available:"
echo "✅ Account-specific follower selection"
echo "✅ Username enhancement (@username display)"
echo "✅ Dataset-wide comprehensive analysis"
echo "✅ Interactive network visualizations"
echo "✅ Cross-account pattern analysis" 
echo "✅ Power follower identification"
echo "✅ Account similarity analysis"
echo "✅ Comprehensive insights dashboard"
echo ""

# Run Streamlit app
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
