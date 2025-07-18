import sys
import os
import json
sys.path.append('src')

# Direct test of the updated processor
print("Testing updated ClusteredDataProcessor...")

from preprocessing.clustered_data_processor import ClusteredDataProcessor

# Initialize
processor = ClusteredDataProcessor()
print(f"User stats loaded: {len(processor.user_stats)}")

# Get a known user from the data
known_users = list(processor.user_stats.keys())[:5]
print(f"Sample users: {known_users}")

# Test with specific cluster
cluster_path = os.path.join(processor.base_path, "followers_1000_to_2500")
user_folders = [d for d in os.listdir(cluster_path) if os.path.isdir(os.path.join(cluster_path, d))]
print(f"Users in cluster: {user_folders[:5]}")

# Find a user that's in both the cluster and the stats
test_user = None
for user in user_folders:
    if user in processor.user_stats:
        test_user = user
        break

if test_user:
    print(f"Testing with user: {test_user}")
    user_stats = processor.user_stats[test_user]
    print(f"  Stats: {user_stats}")
    
    # Process just this one user manually
    user_path = os.path.join(cluster_path, test_user)
    info_files = [f for f in os.listdir(user_path) if f.endswith('.info')]
    
    if info_files:
        info_file = info_files[0]
        print(f"  Testing file: {info_file}")
        
        with open(os.path.join(user_path, info_file), 'r', encoding='utf-8') as f:
            post_data = json.load(f)
            
        owner = post_data.get('owner', {})
        username = owner.get('username', test_user)
        
        # Get stats for this user
        user_stats = processor.user_stats.get(username, {})
        followers_count = user_stats.get('followers', 0)
        
        print(f"  Username: {username}")
        print(f"  Followers from stats: {followers_count}")
        
        # Create test row
        test_row = {
            'username': username,
            '#Followers': followers_count,
            '#Followees': user_stats.get('followees', 0),
            '#Posts': user_stats.get('posts', 0),
            'Category': user_stats.get('category', ''),
            'likes': post_data.get('edge_media_preview_like', {}).get('count', 0)
        }
        
        print(f"Test result: {test_row}")
        
        # Save test result
        import pandas as pd
        test_df = pd.DataFrame([test_row])
        test_df.to_csv('outputs/single_user_test.csv', index=False)
        print("Single user test saved to outputs/single_user_test.csv")
        
else:
    print("No matching user found in both cluster and stats")

print("Manual test completed.")
