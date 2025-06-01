# User Guide - Instagram User Behavior Analysis Dashboard

## Table of Contents
- [Getting Started](#getting-started)
- [Dashboard Overview](#dashboard-overview)
- [Step-by-Step Workflows](#step-by-step-workflows)
- [Feature Tutorials](#feature-tutorials)
- [Data Management](#data-management)
- [Interpreting Results](#interpreting-results)
- [Best Practices](#best-practices)
- [Common Use Cases](#common-use-cases)

---

## Getting Started

### First Time Setup
1. **Complete Installation**: Follow the [INSTALLATION.md](INSTALLATION.md) guide
2. **Prepare Your Data**: Ensure your Instagram data is in CSV format
3. **Launch Application**: Run `streamlit run app.py`
4. **Access Dashboard**: Open `http://localhost:8501` in your browser

### Quick Start Checklist
- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data file ready (CSV format)
- [ ] Application launched successfully

---

## Dashboard Overview

### Main Interface Components

#### 1. **Data Selection & Actions Sidebar**
- File uploader for CSV data
- Data preprocessing pipeline trigger
- Analysis pipeline execution
- Model training controls

#### 2. **Core Analysis Sections**
- **Sentiment Analysis**: Text emotion and sentiment extraction
- **User Clustering**: Behavioral segmentation
- **Engagement Prediction**: ML-powered engagement forecasting

#### 3. **Model Performance Evaluation**
- Interactive model training buttons
- Comprehensive metrics dashboard
- Performance comparison tools
- Training history tracking

#### 4. **Personalized Recommendations**
- AI-powered content suggestions
- User-specific recommendations
- Performance-based ranking

---

## Step-by-Step Workflows

### Workflow 1: Complete Data Analysis (New Users)

#### Step 1: Data Preparation
1. **Upload Data**:
   ```
   Sidebar → "Upload cleaned_merged_user_post_data.csv"
   → Select your CSV file
   ```

2. **Verify Data Loading**:
   - Look for "Data loaded successfully!" message
   - Check data preview in main area

#### Step 2: Run Initial Analysis
1. **Execute Analysis Pipeline**:
   ```
   Sidebar → "Run Analysis Pipeline" button
   ```

2. **Monitor Progress**:
   - Sentiment Analysis progress bar
   - User Clustering completion
   - Engagement Prediction results

#### Step 3: Train ML Models
1. **Navigate to Model Evaluation**:
   ```
   Main Area → "🎯 Model Performance Evaluation" section
   ```

2. **Train Models Sequentially**:
   ```
   Click: "🎯 Train Engagement Models"
   Wait for completion → Success message
   
   Click: "👍💬 Train Likes/Comments Models"  
   Wait for completion → Success message
   ```

#### Step 4: Analyze Results
1. **Explore Dashboard Tabs**:
   - **Overview**: Model status and recent activity
   - **Detailed Metrics**: Comprehensive performance metrics
   - **Performance Comparison**: Cross-model comparisons
   - **History**: Training timeline and trends

2. **Generate Recommendations**:
   ```
   Scroll to → "Advanced Personalized Post Recommendations"
   Review AI-generated suggestions
   ```

### Workflow 2: Model Retraining (Existing Users)

#### Step 1: Check Current Status
1. **Review Model Performance**:
   ```
   Model Evaluation → "Overview" tab
   Check training status and last updated times
   ```

#### Step 2: Retrain Models
1. **Full Retraining**:
   ```
   Click: "🔄 Retrain All Models"
   Monitor progress indicators
   ```

2. **Selective Retraining**:
   ```
   Choose specific model type:
   - Engagement models only
   - Likes/Comments models only
   ```

#### Step 3: Compare Performance
1. **Before/After Analysis**:
   ```
   History tab → Compare training sessions
   Performance Comparison → Evaluate improvements
   ```

### Workflow 3: Data Processing Pipeline

#### Step 1: Raw Data Processing
1. **Trigger Full Pipeline**:
   ```
   Sidebar → "Preprocess Raw Data (Full Pipeline)"
   ```

2. **Pipeline Stages**:
   - Data extraction from raw files
   - Data merging with influencer data
   - Data cleaning and validation
   - Final dataset generation

#### Step 2: Verify Processing
1. **Check Output**:
   ```
   Look for: "Data preprocessing complete!"
   Verify: data/processed_data/cleaned_merged_user_post_data.csv
   ```

---

## Feature Tutorials

### Using the Model Evaluation Dashboard

#### Training Models Interactively
1. **Access Training Controls**:
   ```
   Navigate to "🎯 Model Performance Evaluation"
   Locate action buttons section
   ```

2. **Train Engagement Models**:
   ```
   Click: "🎯 Train Engagement Models"
   
   What happens:
   - Trains 3 algorithms (Linear Regression, Random Forest, Ridge)
   - Calculates 15+ evaluation metrics
   - Performs 5-fold cross-validation
   - Saves results to JSON files
   ```

3. **Monitor Training Progress**:
   ```
   Watch for:
   - Progress spinner
   - Success/error messages
   - Status card updates
   ```

#### Interpreting Evaluation Metrics

##### **Overview Tab**
- **Training Status**: Shows which models are trained
- **Best Models**: Highlights top performers by R² score
- **Recent Activity**: Timeline of training sessions
- **Recommendations**: Suggested next actions

##### **Detailed Metrics Tab**
Key metrics to focus on:

1. **R² Score**: Model accuracy (higher = better, max = 1.0)
   - > 0.8: Excellent
   - 0.6-0.8: Good
   - 0.4-0.6: Fair
   - < 0.4: Poor

2. **RMSE (Root Mean Square Error)**: Prediction error (lower = better)
   - Compare across models for same target

3. **MAE (Mean Absolute Error)**: Average prediction error (lower = better)
   - More interpretable than RMSE

4. **MAPE (Mean Absolute Percentage Error)**: Percentage-based error
   - < 10%: Excellent
   - 10-20%: Good
   - 20-50%: Fair
   - > 50%: Poor

##### **Performance Comparison Tab**
- **Side-by-side metrics**: Compare all models at once
- **Best model highlighting**: Automatically identifies top performers
- **Visual comparisons**: Bar charts for easy comparison

##### **History Tab**
- **Training timeline**: Track improvement over time
- **Historical comparison**: Compare current vs. previous training sessions
- **Performance trends**: Identify patterns and improvements

### Content Recommendation System

#### Generating Recommendations
1. **Access Recommendations**:
   ```
   Scroll to: "Advanced Personalized Post Recommendations"
   ```

2. **Recommendation Process**:
   - Content similarity analysis using TF-IDF
   - Engagement prediction integration
   - Personalized ranking algorithms

3. **Understanding Results**:
   - **Recommended Posts**: Top-ranked content suggestions
   - **Similarity Scores**: Content relevance metrics
   - **Predicted Engagement**: Expected performance metrics

---

## Data Management

### Supported Data Formats

#### Input Data Requirements
- **Format**: CSV files
- **Required Columns**: 
  - User identification fields
  - Post content/description
  - Engagement metrics (likes, comments)
  - Temporal data (post time/date)

#### Example Data Structure
```csv
user_id,post_id,content,likes,comments,timestamp,category
user_001,post_123,"Great sunset photo!",45,3,2024-01-15 18:30:00,lifestyle
user_002,post_124,"Coffee morning vibes",23,1,2024-01-15 09:15:00,food
```

### Data Storage Locations

#### Input Data
- `data/`: Raw data files
- `data/processed_data/`: Cleaned datasets
- `data/clustered_data/`: Clustering results

#### Output Data
- `outputs/`: Model files and evaluation results
- `outputs/model_*.joblib`: Trained machine learning models
- `outputs/model_evaluation_*.json`: Evaluation metrics
- `outputs/*.png`: Generated visualizations

### Data Quality Best Practices

1. **Data Cleaning**:
   - Remove duplicates before upload
   - Ensure consistent date formats
   - Handle missing values appropriately

2. **Data Volume**:
   - Minimum 1,000 rows for reliable training
   - Recommended 10,000+ rows for best results

3. **Data Completeness**:
   - Ensure all required columns are present
   - Minimize missing values in key fields

---

## Interpreting Results

### Model Performance Metrics

#### Engagement Prediction Results
- **High R² (>0.8)**: Model accurately predicts engagement
- **Low RMSE**: Predictions close to actual values
- **Balanced Metrics**: No single metric should be extremely poor

#### Clustering Analysis
- **Cluster Count**: Optimal number of user segments
- **Cluster Characteristics**: Behavioral patterns per cluster
- **Cluster Stability**: Consistent groupings across runs

#### Sentiment Analysis
- **Sentiment Distribution**: Positive/Negative/Neutral ratios
- **Sentiment Trends**: Changes over time
- **Correlation with Engagement**: Sentiment impact on performance

### Recommendation Quality
- **Relevance**: How well recommendations match user preferences
- **Diversity**: Variety in recommended content
- **Performance Prediction**: Expected engagement for recommendations

---

## Best Practices

### Data Preparation
1. **Clean Data First**: Always preprocess raw data before analysis
2. **Consistent Formats**: Ensure uniform data formatting
3. **Adequate Volume**: Use sufficient data for reliable models

### Model Training
1. **Regular Retraining**: Update models with new data
2. **Performance Monitoring**: Track metrics over time
3. **Cross-Validation**: Always use validation for robust results

### Dashboard Usage
1. **Sequential Workflow**: Follow recommended step-by-step process
2. **Monitor Progress**: Watch for completion messages
3. **Save Results**: Download important outputs before clearing

### Performance Optimization
1. **Memory Management**: Close unused tabs/applications
2. **Data Size**: Use appropriate data sizes for your system
3. **Network Stability**: Ensure stable internet for downloads

---

## Common Use Cases

### Use Case 1: Content Strategy Optimization
**Goal**: Improve post engagement rates

**Workflow**:
1. Upload historical post data
2. Train engagement prediction models
3. Analyze top-performing content characteristics
4. Generate recommendations for future posts
5. Monitor performance improvements

### Use Case 2: Audience Segmentation
**Goal**: Understand user behavior patterns

**Workflow**:
1. Run clustering analysis on user data
2. Examine cluster characteristics
3. Develop targeted content strategies
4. Track cluster-specific engagement trends

### Use Case 3: Performance Benchmarking
**Goal**: Establish baseline metrics and track improvements

**Workflow**:
1. Train initial models on historical data
2. Document baseline performance metrics
3. Regularly retrain models with new data
4. Track performance trends over time
5. Identify areas for improvement

### Use Case 4: A/B Testing Analysis
**Goal**: Compare different content strategies

**Workflow**:
1. Segment data by test groups
2. Train separate models for each group
3. Compare performance metrics
4. Identify winning strategies
5. Implement successful approaches

---

## Troubleshooting Common Issues

### Model Training Issues
- **Low Performance**: Increase data volume or improve data quality
- **Training Failures**: Check data format and completeness
- **Memory Errors**: Reduce data size or increase system memory

### Dashboard Issues
- **Slow Loading**: Check system resources and data size
- **UI Errors**: Refresh browser and check console for errors
- **Missing Results**: Verify model training completion

### Data Issues
- **Upload Failures**: Check file format and size
- **Processing Errors**: Verify data structure and column names
- **Inconsistent Results**: Check for data quality issues

For additional support, refer to the [INSTALLATION.md](INSTALLATION.md) troubleshooting section or consult the [API_REFERENCE.md](API_REFERENCE.md) for technical details.
