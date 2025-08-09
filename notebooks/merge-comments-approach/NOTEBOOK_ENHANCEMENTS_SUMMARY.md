# Instagram Sentiment & Engagement Prediction - Enhanced ML Notebook

## 🎯 **COMPLETED ENHANCEMENTS**

### **Target Variables Added:**
1. **Comment Sentiment Prediction**: `comments_effective_sentiment` (-1.0 to +1.0)
2. **Sentiment-Weighted Engagement**: `likes + (comment_sentiment * weight)`  
3. **Comment Emotion Prediction**: `comments_emotion` (joy/sadness/anger/fear/surprise/disgust)
4. **Engagement Quality Score**: Follower-adjusted engagement with sentiment boost

### **User Profile Features Integrated:**
- `#Followers`: User's follower count (log-transformed for better scaling)
- `#Followees`: User's following count (log-transformed)  
- `#Posts`: User's total post count (log-transformed)

### **Data Leakage Prevention:**
✅ **Safe Features** (available before posting):
- Caption sentiment, hashtags, media type
- User profile metrics (#Followers, #Followees, #Posts)
- Content features (length, mentions, URLs)
- Temporal features (posting time)

❌ **Excluded Features** (available only after posting):
- Actual comment text content
- Post engagement metrics (likes, comments as features)
- Comment-derived metrics

### **Enhanced ML Pipeline:**
1. **Target Variable Calculation**: Automatically creates sentiment-weighted engagement
2. **Comprehensive Feature Preprocessing**: Handles text, numerical, and categorical features
3. **Multi-Target Prediction**: Trains separate models for each target variable
4. **Advanced Text Processing**: TF-IDF + BERT embeddings for caption/hashtag analysis
5. **Profile-Aware Modeling**: Incorporates user follower/following dynamics

### **Model Architecture:**
- **XGBoost Regressors**: For numerical targets (sentiment scores, engagement)
- **XGBoost Classifiers**: For categorical targets (emotion categories)
- **Feature Engineering**: Log-transformed profile features, TF-IDF, BERT embeddings
- **Validation**: Comprehensive metrics (R², MAE, RMSE, Accuracy)

## 🚀 **HOW TO USE:**

### **1. Upload Dataset**
The notebook expects your CSV file with columns:
```
caption, hashtags, media_type, #Followers, #Followees, #Posts,
comments_effective_sentiment, comments_emotion, likes, etc.
```

### **2. Run Training**
The notebook will:
- Calculate sentiment-weighted engagement targets
- Extract leak-free features  
- Train multi-target models
- Provide comprehensive validation

### **3. Make Predictions**
Use the prediction function to forecast:
- Comment sentiment quality
- Sentiment-weighted engagement potential
- Expected comment emotions

## 📊 **SAMPLE PREDICTION:**
```python
predict_instagram_engagement(
    caption="Enjoying a beautiful sunset at the beach! 🌅",
    hashtags="#sunset #beach #photography #nature",
    media_type="photo",
    followers=5954,
    followees=1876, 
    posts=317
)

# Output:
# 💬 Comment Sentiment: 0.750 (Positive)
# 🔥 Sentiment-Weighted Engagement: 307
# 😊 Comment Emotion: joy
```

## 🔍 **KEY IMPROVEMENTS:**

1. **Multi-Target Learning**: Simultaneously predicts sentiment, emotion, and engagement
2. **Profile-Aware**: Uses follower dynamics to adjust predictions
3. **Leak-Free**: Only uses pre-posting features for real-world applicability
4. **Production-Ready**: Comprehensive validation and error handling
5. **Scalable**: Efficient feature processing for large datasets

## 📈 **BUSINESS VALUE:**

- **Content Planning**: Predict engagement before posting
- **Sentiment Optimization**: Forecast comment sentiment quality
- **Audience Insights**: Understand emotion responses
- **ROI Estimation**: Calculate sentiment-weighted engagement potential
- **Profile Strategy**: Leverage follower/following dynamics for better predictions

## 🛠 **TECHNICAL FEATURES:**

- **Advanced NLP**: BERT embeddings + TF-IDF for deep text understanding
- **Robust Preprocessing**: Handles missing data, outliers, and edge cases
- **Feature Engineering**: Log-transformations, sentiment weighting, profile metrics
- **Model Validation**: Multiple metrics, confusion matrices, feature importance
- **Memory Efficient**: Optimized data types and processing pipeline

The notebook is now **production-ready** for predicting Instagram engagement sentiment, sentiment-weighted engagement metrics, and comment emotions using only pre-posting features including user profile data.
