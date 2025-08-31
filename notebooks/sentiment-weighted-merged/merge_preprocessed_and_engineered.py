import pandas as pd

# Load the CSV files
preprocessed_df = pd.read_csv("/home/krishantha/Github/fyp-l4/outputs/preprocessed_data.csv")
engineered_df = pd.read_csv("/home/krishantha/Github/fyp-l4/outputs/engineered_data_filtered.csv")

# Merge the two DataFrames on 'post_id' using left join to preserve all rows from preprocessed_data
merged_df = pd.merge(preprocessed_df, engineered_df, on='post_id', how='left')

# Optional: Save the merged DataFrame to a new CSV file
merged_df.to_csv("merged_data.csv", index=False)

print("Merge completed. Shape of merged data:", merged_df.shape)
