import pandas as pd
import json

# === Step 1: Load CSVs ===
df_main = pd.read_csv("preprocessed_data.csv")
df_post_level = pd.read_csv("engineered_data_filtered.csv")

# === Step 2: Load JSON files ===
with open("sentiment_scores.json", "r", encoding="utf-8") as f:
    sentiment_data = json.load(f)["sentiment_scores"]

with open("high_value_followers_215706556.json", "r", encoding="utf-8") as f:
    high_value_data = json.load(f)
    high_value_usernames = set(high_value_data["high_value_followers"].keys())

# === Step 3: Convert Sentiment Scores JSON to DataFrame ===
sentiment_records = []
for entry in sentiment_data.values():
    sentiment_records.append({
        "post_id": int(entry["post_id"]),
        "comment_owner_username": entry["comment_owner_username"],
        "comment_text": entry["comment_text"],
        "sentiment": entry["sentiment"],
        "sentiment_confidence": entry["confidence"],
        "sentiment_positive": entry["positive"],
        "sentiment_negative": entry["negative"],
        "sentiment_neutral": entry["neutral"]
    })

df_sentiment = pd.DataFrame(sentiment_records)

# === Step 4: Merge Sentiment into Main Dataset ===
df_merged = pd.merge(df_main, df_sentiment,
                     on=["post_id", "comment_owner_username", "comment_text"],
                     how="left")

# === Step 5: Add High-Value Follower Indicator ===
df_merged["is_high_value_follower"] = df_merged["comment_owner_username"].isin(high_value_usernames)

# === Step 6: Merge Post-Level Metrics (e.g. sentiment_weighted_engagement) ===
df_final = pd.merge(df_merged, df_post_level,
                    on=["post_id", "likes"],
                    how="left",
                    suffixes=('', '_post'))

# === Step 7: Save Final Combined Dataset ===
df_final.to_csv("final_combined_dataset.csv", index=False)
print("✅ Merged dataset saved as final_combined_dataset.csv")
