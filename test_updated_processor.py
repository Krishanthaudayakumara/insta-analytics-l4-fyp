#!/usr/bin/env python3
"""
Test the updated clustered data processor with follower information
"""

import sys
import os
sys.path.append('src')

from preprocessing.clustered_data_processor import ClusteredDataProcessor

def test_user_stats_loading():
    print("=== Testing Updated ClusteredDataProcessor ===")
    
    processor = ClusteredDataProcessor()
    
    print(f"1. User stats loaded: {len(processor.user_stats)} users")
    
    # Show sample user stats
    sample_users = list(processor.user_stats.keys())[:5]
    for user in sample_users:
        stats = processor.user_stats[user]
        print(f"   {user}: {stats['followers']} followers, {stats['followees']} following, {stats['posts']} posts, category: {stats['category']}")
    
    # Test processing a small cluster
    clusters = processor.get_cluster_names()
    if clusters:
        test_cluster = "followers_1000_to_2500"  # Small cluster for testing
        print(f"\n2. Testing cluster: {test_cluster}")
        
        try:
            # Process just a few users to test
            df = processor.process_cluster(test_cluster)
            
            if not df.empty:
                print(f"   ✅ Successfully processed {len(df)} rows")
                
                # Check the follower columns
                unique_followers = df['#Followers'].unique()
                unique_followees = df['#Followees'].unique() 
                unique_posts = df['#Posts'].unique()
                
                print(f"   Follower counts found: {sorted(unique_followers)}")
                print(f"   Followee counts found: {sorted(unique_followees)}")
                print(f"   Post counts found: {sorted(unique_posts)}")
                
                # Show sample row
                sample_row = df.iloc[0]
                print(f"\n   Sample data:")
                print(f"   Username: {sample_row['username']}")
                print(f"   Followers: {sample_row['#Followers']}")
                print(f"   Following: {sample_row['#Followees']}")
                print(f"   Posts: {sample_row['#Posts']}")
                print(f"   Category: {sample_row['Category']}")
                print(f"   Likes: {sample_row['likes']}")
                
                # Test compatibility transformation
                compatible_df = processor.create_compatible_dataframe(df)
                print(f"\n   ✅ Compatible dataframe created with {len(compatible_df)} rows")
                
                # Save test output
                output_file = "outputs/test_updated_processor.csv"
                compatible_df.head(50).to_csv(output_file, index=False)
                print(f"   ✅ Sample saved to: {output_file}")
                
                return True
            else:
                print("   ❌ Empty dataframe returned")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("No clusters found!")
        return False

if __name__ == "__main__":
    success = test_user_stats_loading()
    print(f"\n{'='*50}")
    if success:
        print("🎉 Updated processor test PASSED!")
        print("Follower counts should now be populated correctly.")
    else:
        print("💥 Test FAILED!")
    print("="*50)
