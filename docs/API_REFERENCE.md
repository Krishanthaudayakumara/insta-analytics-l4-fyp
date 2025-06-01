# API Reference Documentation

## Table of Contents
- [Core Analysis Modules](#core-analysis-modules)
- [Dashboard Modules](#dashboard-modules)
- [Data Processing Modules](#data-processing-modules)
- [Utility Modules](#utility-modules)
- [Model Configuration](#model-configuration)

---

## Core Analysis Modules

### `analysis/engagement_prediction.py`

#### Overview
The main machine learning module responsible for training, evaluating, and managing engagement prediction models.

#### Functions

##### `calculate_comprehensive_metrics(y_true, y_pred, model_name="Model")`
Calculate 15+ comprehensive evaluation metrics for regression models.

**Parameters:**
- `y_true` (array-like): True target values
- `y_pred` (array-like): Predicted values  
- `model_name` (str): Name of the model for reporting

**Returns:**
- `dict`: Dictionary containing evaluation metrics:
  - `MSE`: Mean Squared Error
  - `RMSE`: Root Mean Squared Error
  - `MAE`: Mean Absolute Error
  - `R2_Score`: Coefficient of determination
  - `MAPE`: Mean Absolute Percentage Error
  - `Explained_Variance`: Explained variance score
  - `Max_Error`: Maximum residual error
  - `Residuals_Std`: Standard deviation of residuals
  - `Residuals_Mean`: Mean of residuals
  - `Accuracy_10_Percent`: Predictions within 10% accuracy
  - `Accuracy_20_Percent`: Predictions within 20% accuracy

**Example:**
```python
from analysis.engagement_prediction import calculate_comprehensive_metrics

metrics = calculate_comprehensive_metrics(
    y_true=[1.0, 2.0, 3.0],
    y_pred=[1.1, 1.9, 3.2],
    model_name="Random Forest"
)
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"R²: {metrics['R2_Score']:.4f}")
```

##### `evaluate_model_with_cross_validation(model, X, y, cv_folds=5)`
Perform k-fold cross-validation and return detailed evaluation results.

**Parameters:**
- `model`: Scikit-learn model instance
- `X` (DataFrame): Feature matrix
- `y` (Series): Target vector
- `cv_folds` (int): Number of cross-validation folds (default: 5)

**Returns:**
- `dict`: Cross-validation results with mean and std of scores

**Example:**
```python
from sklearn.ensemble import RandomForestRegressor
from analysis.engagement_prediction import evaluate_model_with_cross_validation

model = RandomForestRegressor(n_estimators=100, random_state=42)
cv_results = evaluate_model_with_cross_validation(model, X_train, y_train)
print(f"CV Mean Score: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
```

##### `save_evaluation_results(results, target_type="engagement")`
Save evaluation results to JSON file with timestamp.

**Parameters:**
- `results` (list): List of evaluation result dictionaries
- `target_type` (str): Type of target variable ("engagement", "likes", "comments")

**Returns:**
- `None`

**Side Effects:**
- Creates/updates JSON file in `outputs/model_evaluation_{target_type}.json`

##### `run()`
Execute the complete model training and evaluation pipeline.

**Parameters:**
- None (reads data from default path)

**Returns:**
- `None`

**Side Effects:**
- Trains and saves models to `outputs/` directory
- Generates evaluation results and saves to JSON
- Prints training progress and results

**Example:**
```python
from analysis import engagement_prediction

# Run complete training pipeline
engagement_prediction.run()
```

---

### `analysis/sentiment_analysis.py`

#### Overview
Text analysis module for extracting sentiment from Instagram post captions and comments.

#### Functions

##### `run(df)`
Perform sentiment analysis on post captions.

**Parameters:**
- `df` (DataFrame): Input dataframe with caption column

**Returns:**
- `DataFrame`: Enhanced dataframe with sentiment scores

**Example:**
```python
from analysis.sentiment_analysis import run

# Analyze sentiment of captions
df_with_sentiment = run(original_dataframe)
print(df_with_sentiment['sentiment_score'].describe())
```

---

### `analysis/clustering_segmentation.py`

#### Overview
User segmentation and behavior clustering module using K-means and other clustering algorithms.

#### Functions

##### `run(df)`
Perform user clustering based on engagement patterns and behavior metrics.

**Parameters:**
- `df` (DataFrame): Input dataframe with user behavior data

**Returns:**
- `DataFrame`: Dataframe with cluster assignments and cluster analysis

**Example:**
```python
from analysis.clustering_segmentation import run

# Perform user clustering
clustered_df = run(user_behavior_data)
print(f"Number of clusters: {clustered_df['cluster'].nunique()}")
```

---

## Dashboard Modules

### `model_evaluation_dashboard.py`

#### Overview
Interactive dashboard module for model evaluation, comparison, and management.

#### Functions

##### `load_evaluation_results(target_type="engagement")`
Load evaluation results from JSON storage.

**Parameters:**
- `target_type` (str): Target type ("engagement", "likes", "comments")

**Returns:**
- `dict` or `list`: Latest evaluation results or empty if not found

**Example:**
```python
import model_evaluation_dashboard as med

results = med.load_evaluation_results("engagement")
if results:
    print(f"Found {len(results)} model evaluations")
```

##### `create_evaluation_comparison_df(target_types=["engagement", "likes", "comments"])`
Create comprehensive comparison DataFrame of all model evaluations.

**Parameters:**
- `target_types` (list): List of target types to include in comparison

**Returns:**
- `DataFrame`: Comparison table with all models and metrics

##### `create_metrics_comparison_chart(df)`
Generate interactive Plotly charts for metrics comparison.

**Parameters:**
- `df` (DataFrame): Evaluation results dataframe

**Returns:**
- `plotly.graph_objects.Figure`: Interactive comparison chart

##### `show_model_evaluation_dashboard()`
Display the complete model evaluation dashboard interface.

**Parameters:**
- None

**Returns:**
- None (renders Streamlit interface)

**Example:**
```python
import model_evaluation_dashboard as med
import streamlit as st

# Display in Streamlit app
if st.button("Show Model Evaluation"):
    med.show_model_evaluation_dashboard()
```

---

### `app.py`

#### Overview
Main Streamlit application module providing the primary user interface.

#### Key Sections

##### Data Upload and Processing
- File upload interface
- Automated data preprocessing pipeline
- Data validation and quality checks

##### Analysis Dashboard
- Data overview and statistics
- Engagement analysis
- Sentiment analysis results
- Clustering visualization

##### Model Training Interface
- Interactive model training buttons
- Real-time progress tracking
- Training status and recommendations

##### Visualization Section
- Interactive charts and plots
- Trend analysis
- Performance metrics display

**Configuration:**
```python
st.set_page_config(
    page_title="Instagram User Behavior Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

---

## Data Processing Modules

### `scripts/process_data_posts.py`

#### Overview
Data extraction and initial processing module for Instagram post data.

#### Functions

##### Main Processing Pipeline
- Raw data extraction from source files
- Initial data validation and formatting
- Feature engineering and derived metrics calculation

**Usage:**
```bash
python3 scripts/process_data_posts.py
```

---

### `scripts/clean_data.py`

#### Overview
Data cleaning and preprocessing utilities for ensuring data quality.

#### Functions

##### `clean_and_deduplicate(input_csv, output_csv)`
Clean and deduplicate the merged Instagram data.

**Parameters:**
- `input_csv` (str): Path to input CSV file
- `output_csv` (str): Path to output cleaned CSV file

**Returns:**
- `None`

**Side Effects:**
- Creates cleaned CSV file
- Logs cleaning operations to `clean_data.log`

**Cleaning Operations:**
- Remove rows with zero comments
- Filter empty captions and hashtags
- Standardize text encoding
- Handle missing values
- Remove duplicate posts
- Normalize data formats

**Example:**
```python
from scripts.clean_data import clean_and_deduplicate

clean_and_deduplicate(
    'data/raw/merged_data.csv',
    'data/processed_data/cleaned_data.csv'
)
```

---

### `scripts/merge_with_influencers.py`

#### Overview
Data merging module for combining post data with influencer information.

#### Functions

##### Main Merging Pipeline
- Merge post data with influencer profiles
- Resolve data conflicts and inconsistencies
- Validate merged data integrity

**Usage:**
```bash
python3 scripts/merge_with_influencers.py
```

---

## Utility Modules

### `recommendations/post_recommender.py`

#### Overview
Content-based recommendation system for Instagram posts.

#### Functions

##### `run(df)`
Generate post recommendations based on engagement and content similarity.

**Parameters:**
- `df` (DataFrame): Input dataframe with post data

**Returns:**
- `dict`: Recommendation results including:
  - Top posts by engagement
  - Content-based similar posts
  - Recommendation scores

**Algorithm:**
1. **Engagement-based Ranking**: Sort posts by likes and comments
2. **Content Similarity**: Use TF-IDF vectorization on captions
3. **Cosine Similarity**: Calculate content similarity scores
4. **Hybrid Recommendations**: Combine engagement and content signals

**Example:**
```python
from recommendations.post_recommender import run

recommendations = run(post_dataframe)
print("Top Recommended Posts:")
for post in recommendations['top_posts']:
    print(f"- {post['caption'][:50]}... (Likes: {post['likes']})")
```

---

### `visualizations/engagement_trends.py`

#### Overview
Visualization utilities for creating charts and plots of engagement trends.

#### Functions

##### `run(df)`
Generate engagement trend visualizations.

**Parameters:**
- `df` (DataFrame): Input dataframe with engagement data

**Returns:**
- Various matplotlib/plotly figures

**Chart Types:**
- Time-series engagement trends
- Engagement by post type
- User engagement distribution
- Correlation heatmaps

---

## Model Configuration

### Model Parameters

#### Random Forest Configuration
```python
RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}
```

#### Ridge Regression Configuration
```python
RIDGE_CONFIG = {
    'alpha': 1.0,
    'fit_intercept': True,
    'normalize': False,
    'random_state': 42,
    'solver': 'auto'
}
```

#### Linear Regression Configuration
```python
LINEAR_REGRESSION_CONFIG = {
    'fit_intercept': True,
    'normalize': False,
    'copy_X': True,
    'n_jobs': None
}
```

### Feature Engineering

#### Standard Features
```python
STANDARD_FEATURES = [
    'likes',
    'comments_count', 
    'hashtags_count',
    'caption_length',
    'post_hour',
    'post_day_of_week',
    'is_weekend'
]
```

#### Derived Features
```python
DERIVED_FEATURES = [
    'engagement_rate',
    'likes_per_follower',
    'comments_per_like',
    'hashtag_density',
    'caption_sentiment_score'
]
```

### Evaluation Metrics Configuration

#### Primary Metrics
```python
PRIMARY_METRICS = [
    'MSE',
    'RMSE', 
    'MAE',
    'R2_Score',
    'MAPE'
]
```

#### Secondary Metrics
```python
SECONDARY_METRICS = [
    'Explained_Variance',
    'Max_Error',
    'Residuals_Std',
    'Residuals_Mean',
    'Accuracy_10_Percent',
    'Accuracy_20_Percent'
]
```

### Cross-Validation Configuration
```python
CV_CONFIG = {
    'cv_folds': 5,
    'shuffle': True,
    'random_state': 42,
    'scoring': ['neg_mean_squared_error', 'r2']
}
```

---

## Error Handling and Validation

### Data Validation
```python
def validate_dataframe(df):
    """Validate input dataframe structure and content."""
    required_columns = ['post_id', 'likes', 'comments_count', 'caption']
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['likes']):
        raise TypeError("'likes' column must be numeric")
    
    # Check for empty dataframe
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    return True
```

### Model Validation
```python
def validate_model_results(results):
    """Validate model evaluation results."""
    required_metrics = ['MSE', 'RMSE', 'MAE', 'R2_Score']
    
    for metric in required_metrics:
        if metric not in results:
            raise KeyError(f"Missing required metric: {metric}")
        
        if not isinstance(results[metric], (int, float)):
            raise TypeError(f"Metric {metric} must be numeric")
    
    return True
```

---

*API Documentation generated: June 1, 2025*  
*Version: 1.0.0*
