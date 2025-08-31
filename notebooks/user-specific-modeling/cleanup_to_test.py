import pandas as pd

# Load the dataset (update the file path if necessary)
input_file = 'final_combined_dataset.csv'
output_file = 'test-cleaned_dataset.csv'

# Features you want to keep
features_to_keep = [
    'caption',
    'hashtags',
    'media_type',
    'caption_length',
    'num_hashtags',
    'has_mention',
    'has_url',
    'timestamp',
    'sentiment_weighted_engagement'
]

# Read CSV
df = pd.read_csv(input_file)

# Keep only the specified columns
cleaned_df = df[features_to_keep]

# Remove duplicate records (improved method)
initial_count = len(cleaned_df)

# Step 1: Basic duplicate removal
cleaned_df = cleaned_df.drop_duplicates()
after_basic = len(cleaned_df)

# Step 2: Handle floating point precision issues if duplicates remain
if cleaned_df.duplicated().sum() > 0:
    print(f"Warning: {cleaned_df.duplicated().sum()} duplicates remain after basic removal")
    # Round numeric columns to handle floating point precision
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        cleaned_df[col] = cleaned_df[col].round(10)
    
    # Try duplicate removal again
    cleaned_df = cleaned_df.drop_duplicates()

# Step 3: Force removal if still duplicates exist
if cleaned_df.duplicated().sum() > 0:
    print(f"Warning: {cleaned_df.duplicated().sum()} duplicates still remain, using aggressive removal")
    # Group by all columns and take first occurrence
    cleaned_df = cleaned_df.reset_index(drop=True)
    group_cols = list(cleaned_df.columns)
    cleaned_df = cleaned_df.groupby(group_cols, dropna=False).first().reset_index()

final_count = len(cleaned_df)
duplicates_removed = initial_count - final_count

# Save to new CSV
cleaned_df.to_csv(output_file, index=False)

print(f"Cleaned dataset saved to '{output_file}' with {final_count} records and {len(features_to_keep)} columns.")
print(f"Removed {duplicates_removed} duplicate records (original: {initial_count}, final: {final_count})")

# Final verification
final_duplicates = pd.read_csv(output_file).duplicated().sum()
if final_duplicates == 0:
    print("✅ SUCCESS: No duplicates remain in the saved file!")
else:
    print(f"⚠️  WARNING: {final_duplicates} duplicates still exist in the saved file!")
