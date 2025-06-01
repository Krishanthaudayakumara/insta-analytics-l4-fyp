# Instagram User Behavior Analysis Dashboard - Documentation

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [API Reference](#api-reference)
- [User Guide](#user-guide)
- [Developer Guide](#developer-guide)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Project Overview

The Instagram User Behavior Analysis Dashboard is a comprehensive machine learning application designed to analyze Instagram user behavior patterns, predict engagement metrics, and provide actionable insights through an interactive web interface.

### Key Features
- **Data Processing Pipeline**: Automated data extraction, cleaning, and preprocessing
- **Machine Learning Models**: Multiple algorithms for engagement prediction
- **Interactive Dashboard**: Streamlit-based web interface with real-time analytics
- **Comprehensive Evaluation**: 15+ evaluation metrics for model performance
- **Content Recommendations**: AI-powered post recommendation system
- **Clustering Analysis**: User segmentation and behavior clustering

### Technology Stack
- **Backend**: Python 3.10+
- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: Plotly, matplotlib
- **Data Storage**: CSV files, JSON configurations, Joblib models
- **Web Framework**: Streamlit with custom components

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Web Interface (Streamlit)               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Dashboard  │  │  Evaluation  │  │ Recommender  │      │
│  │   (app.py)   │  │  Dashboard   │  │   System     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    Core Analysis Modules                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Engagement  │  │  Sentiment   │  │  Clustering  │      │
│  │  Prediction  │  │   Analysis   │  │ Segmentation │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                   Data Processing Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Data Extract │  │  Data Clean  │  │ Data Merge   │      │
│  │ (process_*)  │  │ (clean_data) │  │(merge_influ.)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                      Data Storage                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Raw Data    │  │   Models     │  │ Evaluations  │      │
│  │    (CSV)     │  │  (.joblib)   │  │   (JSON)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### File Structure
```
fyp-l4/
├── app.py                              # Main Streamlit application
├── model_evaluation_dashboard.py       # Model evaluation interface
├── requirements.txt                    # Python dependencies
├── README.md                          # Project documentation
├── analysis/                          # Core ML modules
│   ├── engagement_prediction.py       # ML models and evaluation
│   ├── sentiment_analysis.py          # Text sentiment analysis
│   └── clustering_segmentation.py     # User clustering
├── recommendations/                   # Recommendation system
│   └── post_recommender.py           # Content recommendation
├── scripts/                          # Data processing scripts
│   ├── process_data_posts.py         # Data extraction
│   ├── clean_data.py                 # Data cleaning
│   └── merge_with_influencers.py     # Data merging
├── visualizations/                   # Chart and plot modules
│   └── engagement_trends.py          # Trend visualization
├── data/                             # Data storage
│   ├── processed_data/               # Cleaned datasets
│   └── clustered_data/               # Clustered results
├── outputs/                          # Model outputs
│   ├── model_*.joblib                # Trained models
│   ├── model_evaluation_*.json       # Evaluation results
│   └── *.png                         # Generated visualizations
└── docs/                             # Documentation
    └── *.md                          # Documentation files
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git (for cloning repository)

### Installation Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd fyp-l4
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python3 -c "import streamlit; print('Streamlit installed successfully')"
   ```

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

### Required Dependencies
- streamlit>=1.28.0
- pandas>=1.5.0
- numpy>=1.21.0
- scikit-learn>=1.1.0
- plotly>=5.0.0
- joblib>=1.2.0
- matplotlib>=3.5.0
- seaborn>=0.11.0

---

## API Reference

### Core Modules

#### `analysis/engagement_prediction.py`
Main module for machine learning model training and evaluation.

**Key Functions:**
- `calculate_comprehensive_metrics(y_true, y_pred, model_name)`: Calculate 15+ evaluation metrics
- `evaluate_model_with_cross_validation(model, X, y, cv_folds=5)`: Perform cross-validation
- `save_evaluation_results(results, target_type)`: Save evaluation to JSON
- `run()`: Main training pipeline execution

**Classes:**
- No specific classes, functional programming approach

**Usage Example:**
```python
from analysis.engagement_prediction import calculate_comprehensive_metrics

# Calculate metrics for model predictions
metrics = calculate_comprehensive_metrics(
    y_true=actual_values,
    y_pred=predicted_values,
    model_name="Random Forest"
)
print(f"R² Score: {metrics['R2_Score']}")
print(f"RMSE: {metrics['RMSE']}")
```

#### `model_evaluation_dashboard.py`
Interactive dashboard for model evaluation and comparison.

**Key Functions:**
- `load_evaluation_results(target_type)`: Load evaluation data from JSON
- `create_evaluation_comparison_df(target_types)`: Create comparison DataFrame
- `create_metrics_comparison_chart(df)`: Generate comparison visualizations
- `show_model_evaluation_dashboard()`: Main dashboard function

**Usage Example:**
```python
import model_evaluation_dashboard as med

# Display the evaluation dashboard
med.show_model_evaluation_dashboard()
```

#### `recommendations/post_recommender.py`
Content-based recommendation system for Instagram posts.

**Key Functions:**
- `run(df)`: Main recommendation pipeline
- Content similarity calculation using TF-IDF vectorization
- Engagement-based ranking algorithm

**Usage Example:**
```python
from recommendations.post_recommender import run

# Generate recommendations
recommendations = run(dataframe)
```

### Data Processing Modules

#### `scripts/clean_data.py`
Data cleaning and preprocessing utilities.

**Key Functions:**
- `clean_and_deduplicate(input_csv, output_csv)`: Clean and deduplicate data
- Text normalization and standardization
- Missing value handling

#### `scripts/merge_with_influencers.py`
Data merging and integration utilities.

**Key Functions:**
- `merge_datasets()`: Merge multiple data sources
- Influencer data integration
- Data consistency validation

---

## User Guide

### Getting Started

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Access Dashboard**
   Open browser and navigate to: `http://localhost:8501`

3. **Upload Data**
   - Use sidebar file uploader
   - Or click "Preprocess Raw Data" for full pipeline

### Main Features

#### 📊 Data Analysis Dashboard
- **Data Overview**: Statistics and data quality metrics
- **Engagement Analysis**: User engagement patterns and trends
- **Sentiment Analysis**: Post sentiment distribution and insights
- **Clustering Results**: User segmentation and behavior groups

#### 🎯 Model Performance Evaluation
- **Overview Tab**: Key metrics summary and model status
- **Detailed Metrics Tab**: Complete evaluation breakdown (15+ metrics)
- **Performance Comparison**: Model-to-model analytics
- **History Tab**: Training timeline and progress tracking

#### 🎮 Interactive Training
- **🎯 Train Engagement Models**: Full engagement prediction workflow
- **👍💬 Train Likes/Comments Models**: Social metrics training
- **🔄 Retrain All Models**: Complete system refresh
- **🗑️ Clear All Evaluations**: Reset functionality

#### 📈 Visualizations
- Interactive charts and plots
- Real-time data updates
- Customizable view options
- Export capabilities

### Workflow Examples

#### Training New Models
1. Navigate to "🎯 Model Performance Evaluation"
2. Click "🎯 Train Engagement Models"
3. Monitor progress in real-time
4. View results in dashboard tabs

#### Comparing Model Performance
1. Access "Performance Comparison" tab
2. Select models to compare
3. Analyze metrics side-by-side
4. Export comparison results

#### Generating Recommendations
1. Load data in main dashboard
2. Navigate to recommendations section
3. View top recommended posts
4. Analyze recommendation rationale

---

## Developer Guide

### Adding New Features

#### Creating a New Analysis Module

1. **Create Module File**
   ```python
   # analysis/new_analysis.py
   import pandas as pd
   import numpy as np
   
   def run(df):
       """Main analysis function"""
       # Your analysis logic here
       return results
   ```

2. **Import in Main App**
   ```python
   # app.py
   from analysis import new_analysis
   
   # Add to analysis section
   if st.button("Run New Analysis"):
       results = new_analysis.run(df)
       st.write(results)
   ```

#### Adding New Evaluation Metrics

1. **Extend Metrics Function**
   ```python
   # analysis/engagement_prediction.py
   def calculate_comprehensive_metrics(y_true, y_pred, model_name="Model"):
       # ...existing metrics...
       
       # Add new metric
       new_metric = calculate_new_metric(y_true, y_pred)
       
       metrics.update({
           'New_Metric': new_metric
       })
       
       return metrics
   ```

2. **Update Dashboard Display**
   ```python
   # model_evaluation_dashboard.py
   def display_detailed_metrics(results):
       # ...existing displays...
       
       # Add new metric display
       col_new = st.columns(1)[0]
       with col_new:
           st.metric("New Metric", f"{result['New_Metric']:.4f}")
   ```

### Code Standards

#### Python Style Guide
- Follow PEP 8 conventions
- Use descriptive variable names
- Add comprehensive docstrings
- Include type hints where appropriate

#### Documentation Requirements
- All public functions must have docstrings
- Include parameter descriptions and return types
- Provide usage examples for complex functions
- Update documentation when adding features

#### Testing Guidelines
- Write unit tests for new functions
- Test edge cases and error conditions
- Validate data integrity
- Performance testing for large datasets

### Configuration Files

#### `requirements.txt`
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
plotly>=5.0.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

#### Environment Variables
- `STREAMLIT_THEME`: UI theme configuration
- `DATA_PATH`: Default data directory path
- `MODEL_PATH`: Model storage directory
- `LOG_LEVEL`: Logging verbosity level

---

## Configuration

### Application Settings

#### Streamlit Configuration
```python
# app.py
st.set_page_config(
    page_title="Instagram User Behavior Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

#### Model Parameters
```python
# analysis/engagement_prediction.py
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10
    },
    'linear_regression': {
        'fit_intercept': True
    },
    'ridge': {
        'alpha': 1.0,
        'random_state': 42
    }
}
```

### Data Configuration

#### File Paths
```python
DATA_PATHS = {
    'raw_data': 'data/raw/',
    'processed_data': 'data/processed_data/',
    'clustered_data': 'data/clustered_data/',
    'models': 'outputs/',
    'evaluations': 'outputs/'
}
```

#### Data Schema
```python
REQUIRED_COLUMNS = [
    'post_id', 'user_id', 'caption', 'hashtags',
    'likes', 'comments_count', 'engagement_rate',
    'post_type', 'timestamp'
]
```

---

## Troubleshooting

### Common Issues

#### 1. Module Import Errors
**Problem**: `ModuleNotFoundError` when importing custom modules

**Solution**:
```bash
# Ensure correct Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
cd /path/to/fyp-l4
python3 -m streamlit run app.py
```

#### 2. Data Loading Issues
**Problem**: CSV file not found or corrupt

**Solution**:
```python
# Check file existence
import os
if not os.path.exists('data/processed_data/cleaned_merged_user_post_data.csv'):
    print("Data file not found. Run preprocessing pipeline.")
    
# Validate data format
df = pd.read_csv('data.csv')
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
```

#### 3. Model Training Failures
**Problem**: sklearn model training errors

**Solution**:
```python
# Check data types and missing values
print(df.dtypes)
print(df.isnull().sum())

# Handle missing values
df = df.dropna()

# Ensure numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns
```

#### 4. Dashboard Display Issues
**Problem**: Streamlit components not rendering

**Solution**:
```bash
# Clear Streamlit cache
streamlit cache clear

# Update Streamlit
pip install --upgrade streamlit

# Check browser compatibility
# Use Chrome/Firefox for best experience
```

### Performance Optimization

#### Memory Management
```python
# Use chunking for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)

# Clear unused variables
del large_dataframe
import gc
gc.collect()
```

#### Caching Strategies
```python
# Use Streamlit caching
@st.cache_data
def load_large_dataset():
    return pd.read_csv('large_dataset.csv')

# Cache model predictions
@st.cache_data
def predict_engagement(_model, features):
    return model.predict(features)
```

### Logging and Debugging

#### Enable Debug Logging
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Streamlit Debug Mode
```bash
streamlit run app.py --logger.level=debug
```

---

## Support and Contributing

### Getting Help
- Check this documentation first
- Review error logs in `logs/` directory
- Search existing issues in project repository
- Create new issue with detailed error information

### Contributing Guidelines
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow code standards and add tests
4. Update documentation as needed
5. Submit pull request with detailed description

### Development Setup
```bash
# Clone repository
git clone <repo-url>
cd fyp-l4

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
python3 -m pytest tests/

# Start development server
streamlit run app.py --server.runOnSave=true
```

---

*Documentation generated: June 1, 2025*  
*Version: 1.0.0*
