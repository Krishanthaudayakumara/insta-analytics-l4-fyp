import os
import sys
sys.path.append('src')

from preprocessing.clustered_data_processor import ClusteredDataProcessor

# Initialize processor
processor = ClusteredDataProcessor()

# Get cluster names
clusters = processor.get_cluster_names()
print(f"Found {len(clusters)} clusters: {clusters}")

# Test with the smallest cluster first
if clusters:
    test_cluster = "followers_1000_to_2500"  # Start with a smaller cluster
    print(f"\nProcessing cluster: {test_cluster}")
    
    try:
        df = processor.process_cluster(test_cluster)
        print(f"Success! DataFrame shape: {df.shape}")
        
        if not df.empty:
            print(f"Columns: {list(df.columns)}")
            print(f"First few rows:")
            print(df.head(2))
            
            # Test the compatibility function
            compatible_df = processor.create_compatible_dataframe(df)
            print(f"\nCompatible DataFrame shape: {compatible_df.shape}")
            
            # Save a sample
            sample_output = "outputs/test_sample.csv"
            compatible_df.head(100).to_csv(sample_output, index=False)
            print(f"Sample saved to: {sample_output}")
        else:
            print("DataFrame is empty!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
