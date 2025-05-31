import os
import logging
import pandas as pd
from analysis import sentiment_analysis, clustering_segmentation, engagement_prediction
from recommendations import post_recommender
from visualizations import engagement_trends

def setup_logging():
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename='logs/project.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

def main():
    setup_logging()
    logging.info("Starting Instagram User Behavior Analysis Project")
    # Load dataset once
    try:
        df = pd.read_csv("data/processed_data/cleaned_merged_user_post_data.csv")
        logging.info("Loaded cleaned_merged_user_post_data.csv successfully.")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    try:
        logging.info("1. Running Sentiment Analysis...")
        df = sentiment_analysis.run(df)
        logging.info("Sentiment analysis complete.")
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {e}")

    try:
        logging.info("2. Running Clustering/User Segmentation...")
        df = clustering_segmentation.run(df)
        logging.info("Clustering/User segmentation complete.")
    except Exception as e:
        logging.error(f"Clustering/User segmentation failed: {e}")

    try:
        logging.info("3. Predicting Engagement...")
        engagement_prediction.run(df)
        logging.info("Engagement prediction complete.")
    except Exception as e:
        logging.error(f"Engagement prediction failed: {e}")

    try:
        logging.info("4. Generating Engagement Visualizations...")
        engagement_trends.run(df)
        logging.info("Engagement visualizations complete.")
    except Exception as e:
        logging.error(f"Engagement visualizations failed: {e}")

    try:
        logging.info("5. Generating Recommendations...")
        post_recommender.run(df)
        logging.info("Recommendations generated.")
    except Exception as e:
        logging.error(f"Recommendation generation failed: {e}")

    # Optionally save the final result
    try:
        os.makedirs('data', exist_ok=True)
        df.to_csv("data/final_with_all_outputs.csv", index=False)
        logging.info("Final output saved to data/final_with_all_outputs.csv")
    except Exception as e:
        logging.error(f"Failed to save final output: {e}")

if __name__ == "__main__":
    main()
