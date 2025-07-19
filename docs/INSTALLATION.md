# Installation and Setup Guide

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space for installation and data
- **CPU**: Any modern processor (multi-core recommended for ML training)

### Software Requirements
- **Operating System**: 
  - Linux (Ubuntu 18.04+, CentOS 7+)
  - macOS 10.14+
  - Windows 10+
- **Python**: Version 3.10 or higher
- **Git**: For repository cloning
- **Web Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

### Network Requirements
- Internet connection for downloading dependencies
- Port 8501 available for Streamlit (or alternative port)

---

## Installation Methods

### Method 1: Direct Installation (Recommended)

#### Step 1: Clone Repository
```bash
# Clone the repository
git clone <repository-url>
cd fyp-l4

# Verify directory structure
ls -la
```

#### Step 2: Python Environment Setup
```bash
# Check Python version
python3 --version

# Create virtual environment (recommended)
python3 -m venv instagram_analysis_env

# Activate virtual environment
# On Linux/macOS:
source instagram_analysis_env/bin/activate
# On Windows:
# instagram_analysis_env\Scripts\activate

# Verify activation
which python3
```

#### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Verify installation
pip list | grep streamlit
```

### Method 2: Docker Installation (Alternative)

#### Step 1: Install Docker
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io docker-compose

# On macOS (with Homebrew)
brew install docker docker-compose

# Start Docker service
sudo systemctl start docker
```

#### Step 2: Build Docker Image
```bash
# Create Dockerfile (if not exists)
cat > Dockerfile << EOF
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
EOF

# Build image
docker build -t instagram-analysis .
```

#### Step 3: Run Container
```bash
# Run the application
docker run -p 8501:8501 instagram-analysis

# Or with volume mounting for data persistence
docker run -p 8501:8501 -v $(pwd)/data:/app/data instagram-analysis
```

---

## Environment Setup

### Virtual Environment (Recommended)

#### Creating Virtual Environment
```bash
# Create environment
python3 -m venv instagram_analysis_env

# Activate environment
source instagram_analysis_env/bin/activate

# Verify activation
echo $VIRTUAL_ENV
```

#### Managing Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Add new package
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt

# Install specific versions
pip install streamlit==1.28.0
```

### Environment Variables

#### Create .env File
```bash
# Create environment configuration
cat > .env << EOF
# Application Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_THEME_BASE=light

# Data Configuration
DATA_PATH=./data
OUTPUT_PATH=./outputs
LOG_PATH=./logs

# Model Configuration
MODEL_RANDOM_STATE=42
CV_FOLDS=5
TEST_SIZE=0.2

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
EOF
```

#### Load Environment Variables
```python
# In Python scripts
import os
from dotenv import load_dotenv

load_dotenv()

# Access variables
data_path = os.getenv('DATA_PATH', './data')
log_level = os.getenv('LOG_LEVEL', 'INFO')
```

---

## Configuration

### Streamlit Configuration

#### Create .streamlit/config.toml
```bash
# Create Streamlit config directory
mkdir -p .streamlit

# Create configuration file
cat > .streamlit/config.toml << EOF
[global]
developmentMode = false

[server]
port = 8501
address = "localhost"
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
base = "light"
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[logger]
level = "info"
messageFormat = "%(asctime)s %(message)s"
EOF
```

### Application Configuration

#### Create config.py
```python
# config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Data file paths
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data"
CLUSTERED_DATA_PATH = DATA_DIR / "clustered_data"

# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'linear_regression': {
        'fit_intercept': True
    },
    'ridge': {
        'alpha': 1.0,
        'random_state': 42
    }
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'cv_folds': 5,
    'test_size': 0.2,
    'random_state': 42
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOG_DIR / 'application.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

### Database Configuration (If Applicable)

#### SQLite Configuration
```python
# database.py
import sqlite3
from pathlib import Path

DATABASE_PATH = Path("data/instagram_analysis.db")

def get_connection():
    """Get database connection."""
    return sqlite3.connect(DATABASE_PATH)

def init_database():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            post_id TEXT PRIMARY KEY,
            user_id TEXT,
            caption TEXT,
            likes INTEGER,
            comments_count INTEGER,
            engagement_rate REAL,
            created_at TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            target_type TEXT,
            mse REAL,
            rmse REAL,
            mae REAL,
            r2_score REAL,
            created_at TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
```

---

## Verification

### Installation Verification

#### Check Python Installation
```bash
# Verify Python version
python3 --version

# Check pip installation
pip --version

# Verify virtual environment
echo $VIRTUAL_ENV
```

#### Check Package Installation
```bash
# Verify key packages
python3 -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python3 -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python3 -c "import sklearn; print(f'Scikit-learn: {sklearn.__version__}')"
python3 -c "import plotly; print(f'Plotly: {plotly.__version__}')"
```

#### Test Application Launch
```bash
# Test application startup
streamlit run app.py --server.headless true &
STREAMLIT_PID=$!

# Wait for startup
sleep 5

# Check if running
if ps -p $STREAMLIT_PID > /dev/null; then
    echo "✅ Streamlit started successfully"
    kill $STREAMLIT_PID
else
    echo "❌ Streamlit failed to start"
fi
```

### Functionality Verification

#### Test Data Loading
```python
# test_setup.py
import pandas as pd
import os

def test_data_loading():
    """Test data loading functionality."""
    test_data = {
        'post_id': ['1', '2', '3'],
        'likes': [100, 200, 150],
        'comments_count': [10, 20, 15],
        'caption': ['Test 1', 'Test 2', 'Test 3']
    }
    
    df = pd.DataFrame(test_data)
    print("✅ Data loading test passed")
    return df

def test_model_import():
    """Test model module imports."""
    try:
        from analysis import engagement_prediction
        from analysis import sentiment_analysis
        from analysis import clustering_segmentation
        print("✅ Model imports test passed")
        return True
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_dashboard_import():
    """Test dashboard module import."""
    try:
        import model_evaluation_dashboard
        print("✅ Dashboard import test passed")
        return True
    except ImportError as e:
        print(f"❌ Dashboard import failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running installation verification tests...")
    
    # Run tests
    test_data_loading()
    test_model_import()
    test_dashboard_import()
    
    print("✅ All verification tests completed!")
```

#### Run Verification Script
```bash
python3 test_setup.py
```

---

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Issues
**Problem**: Python version incompatibility

**Solution**:
```bash
# Check available Python versions
ls /usr/bin/python*

# Use specific Python version
python3.10 -m venv venv
source venv/bin/activate

# Or install Python 3.10
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-pip

# macOS (with Homebrew):
brew install python@3.10
```

#### 2. Permission Issues
**Problem**: Permission denied during installation

**Solution**:
```bash
# Use user install
pip install --user -r requirements.txt

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/
```

#### 3. Package Dependency Conflicts
**Problem**: Conflicting package versions

**Solution**:
```bash
# Create fresh environment
rm -rf instagram_analysis_env
python3 -m venv instagram_analysis_env
source instagram_analysis_env/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install packages one by one
pip install streamlit
pip install pandas
pip install scikit-learn
pip install plotly
```

#### 4. Streamlit Port Issues
**Problem**: Port 8501 already in use

**Solution**:
```bash
# Find process using port
lsof -i :8501

# Kill process if needed
kill -9 <PID>

# Or use different port
streamlit run app.py --server.port 8502
```

#### 5. Memory Issues
**Problem**: Out of memory during model training

**Solution**:
```bash
# Monitor memory usage
htop

# Reduce model complexity in config.py
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 50,  # Reduced from 100
        'max_depth': 5       # Reduced from 10
    }
}

# Use data sampling for large datasets
```

### Environment Issues

#### Virtual Environment Activation
```bash
# If activation fails
source venv/bin/activate

# Verify activation
which python3
echo $VIRTUAL_ENV

# Deactivate if needed
deactivate
```

#### Path Issues
```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Python script
import sys
sys.path.append('.')
```

### Performance Optimization

#### Memory Optimization
```python
# Optimize pandas memory usage
def optimize_memory(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

#### CPU Optimization
```python
# Use multiple cores for model training
MODEL_CONFIG = {
    'random_forest': {
        'n_jobs': -1  # Use all available cores
    }
}

# Enable parallel processing
import os
os.environ['JOBLIB_MULTIPROCESSING'] = '1'
```

---

## Post-Installation Steps

### Data Preparation
```bash
# Create sample data structure
mkdir -p data/{raw,processed_data,clustered_data}
mkdir -p outputs
mkdir -p logs

# Download sample data (if available)
# wget <sample-data-url> -O data/raw/sample_data.csv
```

### Initial Configuration
```bash
# Run initial setup
python3 -c "
import os
from pathlib import Path

# Create necessary directories
dirs = ['data/raw', 'data/processed_data', 'data/clustered_data', 'outputs', 'logs']
for dir_path in dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f'Created directory: {dir_path}')

print('✅ Initial setup completed!')
"
```

### First Run
```bash
# Start the application
streamlit run app.py

# Open browser to http://localhost:8501
# Upload sample data or run preprocessing pipeline
```

---

*Installation Guide generated: June 1, 2025*  
*Version: 1.0.0*
