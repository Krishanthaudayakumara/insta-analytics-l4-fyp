#!/bin/bash
# Run Dataset Analysis Validation with Virtual Environment

echo "🚀 Starting Dataset Analysis Validation with Virtual Environment"
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
echo "Python version: $(python --version)"

# Install requirements if needed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
fi

# Run validation
echo "🧪 Running Dataset Analysis Validation..."
python final_dataset_analysis_validation.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Validation completed successfully!"
else
    echo "❌ Validation failed!"
fi

# Deactivate virtual environment
deactivate
echo "🏁 Validation complete"
