import pandas as pd
import logging
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def classify_sentiment(text):
    if not isinstance(text, str): return "neutral"
    polarity = TextBlob(text).sentiment.polarity
    return "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"

def vader_sentiment(text):
    if not isinstance(text, str): return 0.0
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']

def run(df):
    logging.info("Running sentiment analysis on captions (TextBlob and VADER)...")
    caption_col = 'Caption' if 'Caption' in df.columns else 'caption'
    # TextBlob
    df['caption_sentiment'] = df[caption_col].apply(classify_sentiment)
    # VADER
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
    df['caption_sentiment_vader'] = df[caption_col].apply(vader_sentiment)
    logging.info("Sentiment analysis complete. Example sentiment counts: %s", df['caption_sentiment'].value_counts().to_dict())
    return df