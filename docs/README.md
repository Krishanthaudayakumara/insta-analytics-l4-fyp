# Instagram User Behavior Analysis: ML-Driven Personalized Engagement Modeling

## 🎯 Project Overview
**Title:** Instagram User Behavior Analysis: Machine Learning-Driven Personalized Engagement Modeling for High-Value Followers

**Objective:** Implement a comprehensive system to create individual engagement profiles for high-value Instagram followers, predict engagement likelihood, and provide content strategy guidelines using advanced ML models.

## ✨ Key Features
- **🎯 Individual Engagement Profiles**: Granular profiles for high-value followers (top 10% by engagement/influence)
- **🤖 Advanced ML Models**: Random Forest, XGBoost, LightGBM, TabNet, GNN, BERT
- **💭 Sentiment-Engagement Integration**: BERT-based sentiment analysis combined with ML models
- **📊 Interactive Streamlit Dashboard**: Complete web interface for all pipelines
- **⚡ Real-time Analysis**: Live engagement prediction and profile generation
- **📈 Comprehensive Visualization**: Advanced plots and performance metricsUser Behavior Analysis: ML-Driven Personalized Engagement Modeling

## Project Overview
**Title:** Instagram User Behavior Analysis: Machine Learning-Driven Personalized Engagement Modeling for High-Value Followers

**Objective:** Implement a system to create individual engagement profiles for high-value Instagram followers, predict engagement likelihood, and provide content strategy guidelines using advanced ML models.

## Key Features
- **Individual Engagement Profiles**: Granular profiles for high-value followers (top 10% by engagement/influence)
- **Advanced ML Models**: Random Forest, XGBoost, LightGBM, TabNet, GNN, BERT
- **Sentiment-Engagement Integration**: BERT-based sentiment analysis combined with ML models
- **Interactive Streamlit Dashboard**: Web interface for all pipelines

## Project Structure
```
├── data/                          # Dataset files
├── src/                          # Source code modules
│   ├── preprocessing/            # Data preprocessing
│   ├── follower_selection/       # High-value follower identification
│   ├── sentiment_analysis/       # BERT sentiment analysis
│   ├── models/                   # ML model implementations
│   ├── evaluation/               # Model evaluation
│   └── profiling/                # Profile generation
├── outputs/                      # Generated outputs
├── app.py                        # Streamlit main application
└── requirements.txt              # Dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- At least 4GB RAM for ML models
- Internet connection for BERT model downloads

### Installation & Launch
1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run the Streamlit application:**
```bash
streamlit run app.py
```

3. **Access the application:**
   - Open your browser to `http://localhost:8501`
   - The app will automatically launch

## 🎮 Application Features

### 📊 Dashboard Sections
1. **Overview**: Project introduction and dataset summary
2. **Data Preprocessing**: Load and clean Instagram data
3. **High-Value Follower Selection**: Identify top 10% followers using K-Means clustering
4. **Sentiment Analysis**: BERT-based comment sentiment analysis
5. **Model Training**: Train 6 different ML models with hyperparameter optimization
6. **Model Evaluation**: Compare models with comprehensive metrics
7. **Profile Generation**: Create individual engagement profiles
8. **Visualization**: Interactive charts and insights

### 🔧 Advanced Features
- **Multi-Model Support**: Random Forest, XGBoost, LightGBM, TabNet, GNN, BERT
- **Feature Engineering**: Automatic creation of engagement and influence scores
- **Hyperparameter Optimization**: Grid search and cross-validation
- **Real-time Predictions**: Live engagement probability calculation
- **Export Capabilities**: Download profiles and visualizations

## 📁 Dataset Requirements

### Required Columns
- `media_type`: Content type (photo/video/album)
- `Category`: Content category (fashion/travel/food/lifestyle/tech)
- `likes`: Post likes count
- `comments_count`: Number of comments
- `comment_text`: Comment content for sentiment analysis
- `comment_owner_username`: Commenter username
- `comment_likes`: Comment likes count
- `#Followers`: Follower count
- `comment_owner_username`: Follower identification
- `comment_likes`: Engagement frequency
- `#Followers`: Influence scoring

### Sample Data Format
```csv
post_id,owner_id,likes,comments_count,media_type,Category,comment_text,comment_owner_username,comment_likes,#Followers
1997412906295247760,2713844557,233,6,photo,travel,Great post!,user123,1,20448
```

## 🔄 Current System Status

### ✅ Completed Components
- **🏗️ Clean Project Structure**: Modular architecture with src/ directory
- **📊 Streamlit Application**: Full web interface running on localhost:8501
- **🔧 Data Processing**: Advanced preprocessing with feature engineering
- **🎯 Follower Selection**: K-Means clustering for top 10% identification
- **💭 Sentiment Analysis**: BERT/RoBERTa integration for comment analysis
- **🤖 ML Models**: 6 advanced models with hyperparameter optimization
- **📈 Evaluation Framework**: Comprehensive metrics and visualizations
- **👤 Profile Generation**: Individual user engagement profiles
- **📦 Dependencies**: All required packages installed and verified

### 🚀 Ready for Use
The system is **fully operational** and ready for:
- Real Instagram dataset analysis
- High-value follower identification
- Engagement prediction modeling
- Personalized content strategy generation

## 🛠️ Technical Architecture

### Core Modules
1. **DataProcessor** (`src/preprocessing/`): Data cleaning and feature engineering
2. **HighValueFollowerSelector** (`src/follower_selection/`): K-Means clustering
3. **BERTSentimentAnalyzer** (`src/sentiment_analysis/`): Transformer-based sentiment
4. **ModelTrainer** (`src/models/`): Multi-model training pipeline
5. **ModelEvaluator** (`src/evaluation/`): Performance metrics and comparison
6. **ProfileGenerator** (`src/profiling/`): Individual profile creation

### Machine Learning Pipeline
```
Raw Data → Preprocessing → Feature Engineering → High-Value Selection
    ↓
Sentiment Analysis → Model Training → Evaluation → Profile Generation
```

## 📊 Model Performance
The system supports comprehensive model comparison with:
- **Accuracy, Precision, Recall, F1-Score**
- **ROC-AUC curves and confusion matrices**
- **Feature importance analysis**
- **Cross-validation results**

## 🎯 Use Cases

### For Marketers
- Identify high-value followers for targeted campaigns
- Predict engagement likelihood for content optimization
- Generate personalized content strategies

### For Researchers
- Analyze Instagram user behavior patterns
- Study engagement prediction models
- Evaluate sentiment-engagement correlations

### For Businesses
- Optimize influencer partnerships
- Improve content strategy ROI
- Enhance audience targeting

## 🔧 Configuration

### Model Parameters
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **XGBoost**: learning_rate, max_depth, n_estimators
- **LightGBM**: num_leaves, learning_rate, feature_fraction
- **TabNet**: n_d, n_a, n_steps, gamma
- **BERT**: Pre-trained transformers (bert-base-uncased, roberta-base)

### System Requirements
- **Memory**: 4GB+ RAM for ML models
- **Storage**: 2GB+ for model weights and data
- **GPU**: Optional for faster BERT inference

## 📚 Documentation

### API Reference
Each module includes comprehensive docstrings and type hints for easy integration.

### Example Usage
```python
from src.preprocessing.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer

# Initialize components
processor = DataProcessor()
trainer = ModelTrainer()

# Process data and train models
processed_data = processor.process_data(raw_data)
model = trainer.train_random_forest(X, y)
```

## 🎉 Getting Started Guide

1. **Launch the App**: `streamlit run app.py`
2. **Load Your Data**: Upload Instagram CSV file
3. **Preprocess**: Clean and engineer features
4. **Select Followers**: Identify high-value users
5. **Analyze Sentiment**: Process comment sentiment
6. **Train Models**: Compare ML algorithms
7. **Generate Profiles**: Create individual insights
8. **Export Results**: Download profiles and visualizations

---

**Status**: ✅ **FULLY OPERATIONAL** - Ready for production use!  
**Last Updated**: July 17, 2025  
**Version**: 2.0 - Advanced ML Pipeline
