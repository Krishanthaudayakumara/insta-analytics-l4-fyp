#!/usr/bin/env python3

print("=== Simple Test Script ===")
print("Starting imports...")

try:
    import pandas as pd
    import numpy as np
    print("✓ Basic imports successful")
    print(f"Pandas version: {pd.__version__}")
    print(f"Numpy version: {np.__version__}")
except Exception as e:
    print(f"✗ Basic imports failed: {e}")
    exit(1)

# Test data generation
print("\nTesting sample data generation...")
try:
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'post_id': [f"post_{i}" for i in range(n_samples)],
        'owner_id': np.random.randint(1000, 9999, n_samples),
        'likes': np.random.exponential(100, n_samples).astype(int),
        'comments_count': np.random.exponential(20, n_samples).astype(int),
        'media_type': np.random.choice(['photo', 'video', 'album'], n_samples),
        'Category': np.random.choice(['fashion', 'travel', 'food', 'lifestyle', 'tech'], n_samples),
        'comment_text': [f"Great post! Comment {i}" for i in range(n_samples)],
        'comment_owner_username': [f"commenter_{i%50}" for i in range(n_samples)],
        'comment_likes': np.random.exponential(5, n_samples).astype(int),
        'Followers': np.random.exponential(1000, n_samples).astype(int)
    }
    
    df = pd.DataFrame(data)
    print(f"✓ Sample data generated: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Save sample data
    df.to_csv('sample_data.csv', index=False)
    print("✓ Sample data saved to sample_data.csv")
    
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    exit(1)

print("\n=== Simple Test Completed Successfully ===")
