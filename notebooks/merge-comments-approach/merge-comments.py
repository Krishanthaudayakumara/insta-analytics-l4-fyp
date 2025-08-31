import pandas as pd
from transformers import pipeline
import torch

# Check device availability
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'cuda' if device == 0 else 'cpu'}")

# Load the sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

# Load your dataset
csv_path = "preprocessed_data.csv"
df = pd.read_csv(csv_path)

# Group all comment texts by post_id
grouped = df.groupby('post_id')['comment_text'].apply(lambda comments: ' '.join(str(c) for c in comments if pd.notna(c)))

# Optionally preview
print("Merged comment text for sentiment analysis (first few rows):")
print(grouped.head())

# Apply sentiment analysis with explicit max_length
results = sentiment_pipeline(grouped.tolist(), truncation=True, max_length=512, batch_size=8)

# Convert to DataFrame
sentiment_df = pd.DataFrame(results, index=grouped.index)
sentiment_df.columns = ['merged_comment_sentiment', 'sentiment_score']

# Join with original posts (drop duplicates for posts)
post_df = df.drop_duplicates(subset='post_id').set_index('post_id')
final_df = post_df.join(sentiment_df)

# Save the result (optional)
final_df.to_csv("posts_with_sentiment.csv", index=False)

print("✅ Sentiment analysis complete. Output saved to posts_with_sentiment.csv")
