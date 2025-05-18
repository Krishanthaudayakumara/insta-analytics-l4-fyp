from textblob import TextBlob

def classify_sentiment(text):
    if not isinstance(text, str): return "neutral"
    polarity = TextBlob(text).sentiment.polarity
    return "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"

def run(df):
    # Use 'caption' if 'Caption' is not present
    caption_col = 'Caption' if 'Caption' in df.columns else 'caption'
    df['caption_sentiment'] = df[caption_col].apply(classify_sentiment)
    return df