import pandas as pd
from analysis import sentiment_analysis, clustering_segmentation, engagement_prediction
from recommendations import post_recommender
from visualizations import engagement_trends

def main():
    # Load dataset once
    df = pd.read_csv("data/processed_data/cleaned_merged_user_post_data.csv")

    print("1. Running Sentiment Analysis...")
    df = sentiment_analysis.run(df)

    print("2. Running Clustering/User Segmentation...")
    df = clustering_segmentation.run(df)

    print("3. Predicting Engagement...")
    engagement_prediction.run(df)

    print("4. Generating Engagement Visualizations...")
    engagement_trends.run(df)

    print("5. Generating Recommendations...")
    post_recommender.run(df)

    # Optionally save the final result
    df.to_csv("data/final_with_all_outputs.csv", index=False)

if __name__ == "__main__":
    main()
