# Developer Guide - Instagram User Behavior Analysis Dashboard

## Table of Contents
- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Code Organization](#code-organization)
- [Contributing Guidelines](#contributing-guidelines)
- [Development Workflows](#development-workflows)
- [Testing Strategy](#testing-strategy)
- [Deployment Guide](#deployment-guide)
- [Performance Optimization](#performance-optimization)

---

## Development Environment Setup

### Prerequisites
- Python 3.10+
- Git
- Virtual environment manager (venv, conda, or pyenv)
- Code editor (VS Code, PyCharm, etc.)

### Setting Up Development Environment

#### 1. Clone and Setup Repository
```bash
# Clone repository
git clone <repository-url>
cd fyp-l4

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

#### 2. Configure IDE
**VS Code Extensions (Recommended):**
- Python
- Pylance
- Python Docstring Generator
- GitLens
- Streamlit

**Configuration Files:**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"]
}
```

#### 3. Environment Variables
Create `.env` file in project root:
```bash
# Development settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
LOG_LEVEL=DEBUG
DEVELOPMENT_MODE=true
```

---

## Project Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                   Presentation Layer                        │
│  ┌─────────────────┐    ┌─────────────────┐               │
│  │   Streamlit UI  │    │   Dashboard     │               │
│  │    (app.py)     │    │   Components    │               │
│  └─────────────────┘    └─────────────────┘               │
├─────────────────────────────────────────────────────────────┤
│                    Business Logic Layer                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │    Analysis     │ │ Recommendations │ │ Evaluations  │ │
│  │    Modules      │ │     System      │ │   Dashboard  │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Data Access Layer                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌──────────────┐ │
│  │  Data Loaders   │ │  Model Storage  │ │ Result Cache │ │
│  │   (scripts/)    │ │  (outputs/)     │ │   (JSON)     │ │
│  └─────────────────┘ └─────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Module Dependencies
```
app.py
├── model_evaluation_dashboard.py
├── analysis/
│   ├── sentiment_analysis.py
│   ├── engagement_prediction.py
│   └── clustering_segmentation.py
├── recommendations/
│   └── post_recommender.py
├── visualizations/
│   └── engagement_trends.py
└── scripts/
    ├── clean_data.py
    ├── process_data_posts.py
    └── merge_with_influencers.py
```

---

## Code Organization

### Directory Structure and Responsibilities

#### `/analysis/` - Core ML Modules
```python
# engagement_prediction.py
- Machine learning model training
- Comprehensive evaluation metrics
- Cross-validation implementation
- Model persistence (save/load)

# sentiment_analysis.py  
- Text preprocessing and cleaning
- Sentiment classification
- Feature extraction for text data

# clustering_segmentation.py
- User behavior clustering
- Segmentation algorithms
- Cluster analysis and interpretation
```

#### `/recommendations/` - Recommendation System
```python
# post_recommender.py
- Content-based filtering
- TF-IDF vectorization
- Recommendation ranking algorithms
- Personalization logic
```

#### `/scripts/` - Data Processing
```python
# process_data_posts.py
- Raw data extraction
- Initial data formatting

# clean_data.py
- Data cleaning and validation
- Missing value handling
- Data type conversions

# merge_with_influencers.py
- Data source integration
- Join operations
- Data enrichment
```

#### `/visualizations/` - Chart Components
```python
# engagement_trends.py
- Plotly chart generation
- Interactive visualizations
- Trend analysis plots
```

### Code Style Guidelines

#### Python Style (PEP 8 + Project Conventions)
```python
# Function naming: snake_case
def calculate_engagement_metrics(data):
    pass

# Class naming: PascalCase
class EngagementPredictor:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_MODEL_PATH = "outputs/"

# Private methods: _leading_underscore
def _validate_input_data(self, data):
    pass
```

#### Documentation Standards
```python
def train_engagement_model(data, target_column, model_type="random_forest"):
    """
    Train an engagement prediction model.
    
    Args:
        data (pd.DataFrame): Training dataset
        target_column (str): Name of target variable column
        model_type (str): Type of model to train
            Options: "linear_regression", "random_forest", "ridge"
    
    Returns:
        tuple: (trained_model, evaluation_metrics, feature_names)
        
    Raises:
        ValueError: If target_column not found in data
        ModelTrainingError: If training fails
        
    Example:
        >>> model, metrics, features = train_engagement_model(
        ...     data=df, 
        ...     target_column="engagement_score",
        ...     model_type="random_forest"
        ... )
        >>> print(f"Model R²: {metrics['R2_Score']}")
    """
    pass
```

#### Error Handling Patterns
```python
# Use specific exception types
try:
    model = load_model(model_path)
except FileNotFoundError:
    logger.error(f"Model file not found: {model_path}")
    raise ModelLoadError(f"Could not load model from {model_path}")
except Exception as e:
    logger.error(f"Unexpected error loading model: {e}")
    raise

# Provide user-friendly error messages
def validate_data_format(data):
    """Validate input data format with helpful error messages."""
    required_columns = ["user_id", "post_content", "engagement_score"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Required: {required_columns}"
        )
```

---

## Contributing Guidelines

### Git Workflow

#### Branch Naming Convention
```bash
# Feature branches
feature/add-clustering-algorithm
feature/improve-recommendation-engine

# Bug fixes  
bugfix/fix-model-training-memory-leak
bugfix/resolve-dashboard-loading-issue

# Documentation
docs/update-api-reference
docs/add-user-tutorials

# Refactoring
refactor/modularize-evaluation-metrics
refactor/optimize-data-processing
```

#### Commit Message Format
```bash
# Structure: <type>(<scope>): <description>
# 
# Types: feat, fix, docs, style, refactor, test, chore
# Scope: module or component name
# Description: imperative mood, lowercase, no period

# Examples:
feat(engagement): add cross-validation to model training
fix(dashboard): resolve memory leak in evaluation display
docs(api): update function documentation with examples
refactor(data): optimize data loading performance
test(clustering): add unit tests for segmentation module
```

#### Pull Request Process
1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Development**:
   - Write code following style guidelines
   - Add/update tests
   - Update documentation

3. **Pre-commit Checks**:
   ```bash
   # Run linting
   flake8 .
   
   # Run formatting
   black .
   
   # Run tests
   pytest tests/
   
   # Type checking
   mypy analysis/ recommendations/
   ```

4. **Submit PR**:
   - Clear description of changes
   - Link related issues
   - Include screenshots for UI changes

### Code Review Guidelines

#### What to Look For
- **Functionality**: Does the code work as intended?
- **Style**: Follows project conventions?
- **Performance**: Efficient algorithms and data structures?
- **Security**: No sensitive data exposure?
- **Testing**: Adequate test coverage?
- **Documentation**: Clear docstrings and comments?

#### Review Checklist
- [ ] Code follows style guidelines
- [ ] Functions have proper docstrings
- [ ] Error handling is comprehensive
- [ ] Tests cover new functionality
- [ ] No debugging print statements
- [ ] Sensitive data is properly handled
- [ ] Performance considerations addressed

---

## Development Workflows

### Adding New ML Models

#### 1. Model Implementation
```python
# analysis/engagement_prediction.py

def train_new_algorithm_model(X, y, **kwargs):
    """
    Train a new ML algorithm for engagement prediction.
    
    Follow existing pattern for consistency:
    1. Initialize model with parameters
    2. Fit on training data
    3. Generate predictions
    4. Calculate comprehensive metrics
    5. Return model and metrics
    """
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Initialize model
    model = GradientBoostingRegressor(**kwargs)
    
    # Train model
    model.fit(X, y)
    
    # Generate predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(y, y_pred, "Gradient Boosting")
    
    return model, metrics
```

#### 2. Integration with Training Pipeline
```python
# Add to existing training functions
def train_and_save_engagement_models(df):
    """Update to include new algorithm."""
    algorithms = {
        "linear_regression": train_linear_regression_model,
        "random_forest": train_random_forest_model,
        "ridge": train_ridge_model,
        "gradient_boosting": train_new_algorithm_model,  # Add here
    }
    # ... rest of implementation
```

#### 3. Dashboard Integration
```python
# model_evaluation_dashboard.py
# Update model loading to handle new algorithm
def load_evaluation_results(target_type="engagement"):
    """Update to include new model types."""
    model_types = [
        "linear_regression", 
        "random_forest", 
        "ridge",
        "gradient_boosting"  # Add here
    ]
    # ... rest of implementation
```

### Adding New Dashboard Features

#### 1. Create Component Function
```python
# model_evaluation_dashboard.py

def create_new_feature_tab():
    """
    Create a new dashboard tab/feature.
    
    Follow Streamlit best practices:
    1. Clear section headers
    2. Input validation
    3. Progress indicators
    4. Error handling
    5. User feedback
    """
    st.subheader("🆕 New Feature")
    
    # Input controls
    with st.expander("Configuration"):
        option = st.selectbox("Select Option", ["A", "B", "C"])
        threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    
    # Processing
    if st.button("Process"):
        with st.spinner("Processing..."):
            try:
                result = process_new_feature(option, threshold)
                st.success("Processing completed!")
                st.write(result)
            except Exception as e:
                st.error(f"Error: {e}")
```

#### 2. Integration with Main Dashboard
```python
# model_evaluation_dashboard.py

def show_model_evaluation_dashboard():
    """Update main dashboard function."""
    tabs = st.tabs([
        "Overview", 
        "Detailed Metrics", 
        "Performance Comparison", 
        "History",
        "New Feature"  # Add new tab
    ])
    
    with tabs[4]:  # New feature tab
        create_new_feature_tab()
```

### Data Processing Pipeline Extensions

#### 1. Create New Processing Script
```python
# scripts/new_data_processor.py
"""
Template for new data processing scripts.
"""
import pandas as pd
import logging
from pathlib import Path

def process_new_data_source(input_path, output_path):
    """
    Process a new type of data source.
    
    Args:
        input_path (str): Path to input data
        output_path (str): Path to save processed data
    """
    try:
        # Load data
        df = pd.read_csv(input_path)
        logging.info(f"Loaded {len(df)} rows from {input_path}")
        
        # Processing steps
        df_processed = df.copy()
        # ... processing logic ...
        
        # Save results
        df_processed.to_csv(output_path, index=False)
        logging.info(f"Saved processed data to {output_path}")
        
        return df_processed
        
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        raise

if __name__ == "__main__":
    # Command line interface
    import sys
    if len(sys.argv) != 3:
        print("Usage: python new_data_processor.py <input_path> <output_path>")
        sys.exit(1)
    
    process_new_data_source(sys.argv[1], sys.argv[2])
```

#### 2. Integration with Main Pipeline
```python
# app.py - Update preprocessing section
if preprocess_data:
    with st.spinner("Processing data..."):
        # Existing processors
        subprocess.run(["python3", "scripts/process_data_posts.py"])
        subprocess.run(["python3", "scripts/merge_with_influencers.py"])
        subprocess.run(["python3", "scripts/clean_data.py"])
        
        # Add new processor
        subprocess.run(["python3", "scripts/new_data_processor.py", 
                       "input_file.csv", "output_file.csv"])
    
    st.success("Data preprocessing complete!")
```

---

## Testing Strategy

### Test Structure
```
tests/
├── unit/
│   ├── test_engagement_prediction.py
│   ├── test_sentiment_analysis.py
│   ├── test_clustering.py
│   └── test_recommendations.py
├── integration/
│   ├── test_dashboard_integration.py
│   └── test_pipeline_integration.py
├── fixtures/
│   ├── sample_data.csv
│   └── mock_models.joblib
└── conftest.py
```

### Unit Test Examples

#### Testing ML Functions
```python
# tests/unit/test_engagement_prediction.py
import pytest
import pandas as pd
import numpy as np
from analysis.engagement_prediction import calculate_comprehensive_metrics

class TestEngagementPrediction:
    
    @pytest.fixture
    def sample_predictions(self):
        """Sample data for testing metrics calculation."""
        return {
            'y_true': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y_pred': [1.1, 1.9, 3.2, 3.8, 5.1]
        }
    
    def test_calculate_comprehensive_metrics(self, sample_predictions):
        """Test metrics calculation with known values."""
        metrics = calculate_comprehensive_metrics(
            sample_predictions['y_true'],
            sample_predictions['y_pred'],
            model_name="Test Model"
        )
        
        # Test required metrics exist
        required_metrics = ['MSE', 'RMSE', 'MAE', 'R2_Score', 'MAPE']
        for metric in required_metrics:
            assert metric in metrics
        
        # Test metric ranges
        assert 0 <= metrics['R2_Score'] <= 1
        assert metrics['MSE'] >= 0
        assert metrics['RMSE'] >= 0
        assert metrics['MAE'] >= 0
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        assert metrics['R2_Score'] == pytest.approx(1.0)
        assert metrics['MSE'] == pytest.approx(0.0)
        assert metrics['RMSE'] == pytest.approx(0.0)
        assert metrics['MAE'] == pytest.approx(0.0)
```

#### Testing Dashboard Functions
```python
# tests/unit/test_dashboard.py
import pytest
from unittest.mock import Mock, patch
import streamlit as st
from model_evaluation_dashboard import load_evaluation_results

class TestDashboard:
    
    @patch('model_evaluation_dashboard.os.path.exists')
    @patch('model_evaluation_dashboard.json.load')
    def test_load_evaluation_results_success(self, mock_json_load, mock_exists):
        """Test successful loading of evaluation results."""
        # Setup mocks
        mock_exists.return_value = True
        mock_json_load.return_value = {
            'model_linear_regression': {'R2_Score': 0.85},
            'model_random_forest': {'R2_Score': 0.92}
        }
        
        # Test function
        results = load_evaluation_results('engagement')
        
        # Assertions
        assert len(results) == 2
        assert 'model_linear_regression' in results
        assert results['model_random_forest']['R2_Score'] == 0.92
    
    @patch('model_evaluation_dashboard.os.path.exists')
    def test_load_evaluation_results_file_not_found(self, mock_exists):
        """Test handling of missing evaluation files."""
        mock_exists.return_value = False
        
        results = load_evaluation_results('nonexistent')
        
        assert results == {}
```

### Integration Tests

#### Pipeline Integration
```python
# tests/integration/test_pipeline_integration.py
import pytest
import pandas as pd
import tempfile
import os
from analysis.engagement_prediction import train_and_save_engagement_models

class TestPipelineIntegration:
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        data = {
            'user_id': range(100),
            'post_content': ['Sample post'] * 100,
            'likes': np.random.randint(0, 100, 100),
            'comments': np.random.randint(0, 20, 100),
            'engagement_score': np.random.uniform(0, 1, 100)
        }
        return pd.DataFrame(data)
    
    def test_full_training_pipeline(self, sample_dataframe):
        """Test complete model training pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set temporary output directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            os.makedirs('outputs', exist_ok=True)
            
            try:
                # Run training pipeline
                train_and_save_engagement_models(sample_dataframe)
                
                # Verify outputs exist
                assert os.path.exists('outputs/model_evaluation_engagement.json')
                assert os.path.exists('outputs/model_linear_regression.joblib')
                assert os.path.exists('outputs/model_random_forest.joblib')
                assert os.path.exists('outputs/model_ridge.joblib')
                
            finally:
                os.chdir(original_dir)
```

### Running Tests

#### Local Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=analysis --cov=recommendations --cov-report=html

# Run specific test file
pytest tests/unit/test_engagement_prediction.py

# Run with verbose output
pytest -v

# Run only failed tests
pytest --lf
```

#### Continuous Integration
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.10
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest --cov=analysis --cov=recommendations
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

---

## Deployment Guide

### Production Deployment

#### Environment Preparation
```bash
# Create production environment
python -m venv venv_prod
source venv_prod/bin/activate
pip install -r requirements.txt

# Set production environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_LOGGER_LEVEL=warning
```

#### Deployment Options

##### Option 1: Local Production Server
```bash
# Start production server
streamlit run app.py --server.port 8501 --server.headless true

# With custom configuration
streamlit run app.py --server.runOnSave false --server.address 0.0.0.0
```

##### Option 2: Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run Docker container
docker build -t instagram-dashboard .
docker run -p 8501:8501 instagram-dashboard
```

##### Option 3: Cloud Deployment (Streamlit Cloud)
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Performance Optimization

#### Memory Management
```python
# Use caching for expensive operations
@st.cache_data
def load_large_dataset(file_path):
    """Cache large dataset loading."""
    return pd.read_csv(file_path)

@st.cache_resource
def load_trained_model(model_path):
    """Cache model loading."""
    return joblib.load(model_path)

# Clear cache when needed
if st.button("Clear Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
```

#### Database Integration
```python
# For larger deployments, consider database integration
import sqlite3

def save_evaluation_to_db(results, target_type):
    """Save evaluation results to database instead of JSON."""
    conn = sqlite3.connect('models.db')
    
    # Create table if not exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_type TEXT,
            model_name TEXT,
            metrics TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert results
    for model_name, metrics in results.items():
        conn.execute(
            'INSERT INTO evaluations (target_type, model_name, metrics) VALUES (?, ?, ?)',
            (target_type, model_name, json.dumps(metrics))
        )
    
    conn.commit()
    conn.close()
```

#### Load Balancing
```python
# For high-traffic deployments
import multiprocessing as mp

def parallel_model_training(data_chunks):
    """Train models in parallel for better performance."""
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(train_single_model, data_chunks)
    return results
```

---

## Performance Optimization

### Code Optimization

#### Efficient Data Processing
```python
# Use vectorized operations instead of loops
# Slow
def calculate_engagement_slow(df):
    engagement = []
    for _, row in df.iterrows():
        eng = (row['likes'] + row['comments']) / row['followers']
        engagement.append(eng)
    return engagement

# Fast
def calculate_engagement_fast(df):
    return (df['likes'] + df['comments']) / df['followers']
```

#### Memory-Efficient Loading
```python
# Load data in chunks for large files
def load_large_csv_efficiently(file_path, chunk_size=10000):
    """Load large CSV files in chunks to manage memory."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = process_data_chunk(chunk)
        chunks.append(processed_chunk)
    
    return pd.concat(chunks, ignore_index=True)
```

#### Model Training Optimization
```python
# Use appropriate algorithms for data size
def select_optimal_algorithm(data_size):
    """Select algorithm based on data characteristics."""
    if data_size < 1000:
        return "linear_regression"  # Fast for small data
    elif data_size < 10000:
        return "random_forest"      # Good balance
    else:
        return "ridge"              # Scalable for large data
```

### Dashboard Optimization

#### Streamlit Performance Tips
```python
# Use session state for expensive computations
if 'expensive_result' not in st.session_state:
    st.session_state.expensive_result = expensive_computation()

# Limit automatic reruns
if st.button("Update Results"):
    # Only update when user requests
    st.session_state.results = generate_new_results()

# Use containers for better layout performance
placeholder = st.empty()
with placeholder.container():
    display_dynamic_content()
```

### Monitoring and Logging

#### Application Monitoring
```python
# Add performance monitoring
import time
import logging

def monitor_function_performance(func):
    """Decorator to monitor function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logging.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    
    return wrapper

@monitor_function_performance
def train_model(data):
    # Model training code
    pass
```

#### Error Tracking
```python
# Implement comprehensive error tracking
import traceback

def handle_errors_gracefully(func):
    """Decorator for graceful error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_details = {
                'function': func.__name__,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log error
            logging.error(f"Error in {func.__name__}: {error_details}")
            
            # Display user-friendly message
            st.error(f"An error occurred: {str(e)}")
            
            # Return None or appropriate default
            return None
    
    return wrapper
```

---

This developer guide provides comprehensive information for contributing to the Instagram User Behavior Analysis Dashboard project. For additional technical details, refer to the [API_REFERENCE.md](API_REFERENCE.md) documentation.
