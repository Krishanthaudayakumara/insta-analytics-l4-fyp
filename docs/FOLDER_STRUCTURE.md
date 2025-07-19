# Instagram Engagement Prediction System - Folder Structure

## 📁 Project Organization

This document outlines the clean, organized folder structure of the Instagram Engagement Prediction System.

### 🏠 Root Directory
```
fyp-l4/
├── app.py                    # Main clean modular application (RECOMMENDED)
├── app_legacy.py            # Legacy enhanced application with live predictions  
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
└── venv/                    # Virtual environment (local development)
```

### 📊 Data Directory (`data/`)
```
data/
├── clustered_data/          # Clustered datasets by follower count
│   ├── influencers.csv      # Main influencer dataset
│   ├── followers_1000_to_2500/
│   ├── followers_2500_to_5000/
│   ├── followers_5000_to_7500/
│   ├── followers_7500_to_10000/
│   ├── followers_10000_to_50000/
│   └── followers_greater_than_50000/
├── processed_data/          # Processed and cleaned datasets
│   ├── cleaned_merged_user_post_data.csv
│   ├── combined_clusters_*.csv
│   ├── conversion_stats.json
│   └── individual_clusters/
└── visualizations/          # Generated visualization images
    ├── correlation_heatmap.png
    ├── likes_by_category.png
    └── ...
```

### 🔧 Source Code (`src/`)
```
src/
├── preprocessing/           # Data preprocessing modules
│   ├── data_processor.py
│   └── clustered_data_processor.py
├── follower_selection/      # High-value follower selection
│   └── high_value_selector.py
├── sentiment_analysis/      # Sentiment analysis components
│   └── bert_analyzer.py
├── models/                  # ML model training
│   └── model_trainer.py
├── evaluation/              # Model evaluation
│   └── model_evaluator.py
├── profiling/               # User profile generation
│   └── profile_generator.py
└── ui/                      # Modular UI components
    ├── base.py              # Base UI component
    ├── overview.py          # System overview
    ├── preprocessing.py     # Data preprocessing UI
    ├── follower_selection.py # Follower selection UI
    ├── sentiment_analysis.py # Sentiment analysis UI
    ├── model_training.py    # Model training UI
    ├── model_evaluation.py  # Model evaluation UI
    ├── profile_generation.py # Profile generation UI
    ├── visualization.py     # Results visualization UI
    └── live_prediction.py   # Live predictions UI
```

### 📤 Outputs Directory (`outputs/`)
```
outputs/
├── preprocessed_data.csv    # Preprocessed dataset
├── high_value_followers.json # Selected high-value followers
├── sentiment_scores.json    # Sentiment analysis results
├── rf_model.pkl            # Random Forest model
├── xgb_model.pkl           # XGBoost model
├── lgb_model.pkl           # LightGBM model
├── metrics.json            # Model performance metrics
├── profiles.json           # Generated user profiles
├── guidelines.json         # Content guidelines
├── training_results.json   # Training session results
├── evaluation_results.json # Model evaluation results
└── multi_model_results.json # Multi-model comparison results
```

### 🧪 Tests Directory (`tests/`)
```
tests/
├── run_tests.py            # Main test runner
├── test_*.py               # Individual test files
├── simple_test*.py         # Simple validation tests
├── validate_*.py           # System validation scripts
├── verify_*.py             # Verification scripts
├── debug_test.py           # Debug utilities
└── final_*_test.py         # Final integration tests
```

### 📚 Documentation (`docs/`)
```
docs/
├── README.md               # Main project documentation
├── FOLDER_STRUCTURE.md     # This file - project organization
├── COMPLETION_REPORT.md    # Project completion reports
├── FINAL_COMPLETION_REPORT.md
├── SYSTEM_FIX_REPORT.md    # System fixes documentation
├── VISUALIZATION_FIXES_REPORT.md
└── LIVE_PREDICTIONS_*.md   # Live predictions documentation
```

### 📊 Reports Directory (`reports/`)
```
reports/
├── test_results.txt        # Test execution results
├── validation_results.txt  # System validation results
└── test_output.txt         # Test output logs
```

### 🔧 Scripts Directory (`scripts/`)
```
scripts/
├── launch_app.py           # Easy app launcher for both versions
├── system_status.py        # System status checker
├── manual_test.py          # Manual testing utilities
└── run_app.py             # Alternative app runner
```

## 🚀 Benefits of This Organization

### ✅ **Clear Separation of Concerns**
- **Applications**: Main files in root for easy access
- **Source Code**: Modular components in `src/`
- **Data**: Raw and processed data in `data/`
- **Outputs**: Generated results in `outputs/`
- **Tests**: All testing files in `tests/`
- **Documentation**: All docs in `docs/`

### ✅ **Easy Navigation**
- Developers can quickly find what they need
- Clear distinction between different file types
- Logical grouping of related functionality

### ✅ **Maintainability**
- Easy to add new components
- Clear structure for team collaboration
- Scalable architecture

### ✅ **Professional Structure**
- Follows industry best practices
- Production-ready organization
- Easy deployment and maintenance

## 🎯 Usage Guidelines

### **Development**
- Main development in `src/` directory
- Add new UI components to `src/ui/`
- Add tests to `tests/` directory

### **Running Applications**
```bash
# Main modular application (RECOMMENDED)
streamlit run app.py

# Legacy enhanced application  
streamlit run app_legacy.py

# Using launcher script
python scripts/launch_app.py --app main    # Launches app.py (modular)
python scripts/launch_app.py --app legacy  # Launches app_legacy.py
```

### **Testing**
```bash
# Run all tests
python tests/run_tests.py

# Run specific tests
python tests/test_specific_component.py
```

### **Documentation**
- Update relevant `.md` files in `docs/`
- Keep `README.md` as the main entry point
- Document new features and changes

---

**🌟 This organized structure ensures the Instagram Engagement Prediction System remains clean, maintainable, and professional! 🌟**
