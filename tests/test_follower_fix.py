#!/usr/bin/env python3
"""
Quick test to verify follower counts are now working
"""

import sys
import os
sys.path.append('src')

def test_follower_counts():
    print("Testing Follower Count Fix...")
    
    try:
        from preprocessing.clustered_data_processor import ClusteredDataProcessor
        
        # Initialize processor
        processor = ClusteredDataProcessor()
        
        # Check if user stats loaded
        print(f"✅ User stats loaded: {len(processor.user_stats)} users")
        
        if len(processor.user_stats) > 0:
            # Show a few sample users
            sample_users = list(processor.user_stats.items())[:3]
            for username, stats in sample_users:
                print(f"   {username}: {stats['followers']} followers, {stats['category']} category")
        
        # Test with a small sample
        clusters = processor.get_cluster_names()
        if clusters:
            test_cluster = "followers_1000_to_2500"
            print(f"\n🔍 Testing cluster: {test_cluster}")
            
            # Process just a small amount to test
            df = processor.process_cluster(test_cluster)
            
            if not df.empty:
                # Check the follower columns specifically
                non_zero_followers = df[df['#Followers'] > 0]
                print(f"✅ Processed {len(df)} rows")
                print(f"✅ Rows with follower data: {len(non_zero_followers)}")
                
                if len(non_zero_followers) > 0:
                    sample = non_zero_followers.iloc[0]
                    print(f"✅ Sample user: {sample['username']}")
                    print(f"   - Followers: {sample['#Followers']}")
                    print(f"   - Following: {sample['#Followees']}")
                    print(f"   - Posts: {sample['#Posts']}")
                    print(f"   - Category: {sample['Category']}")
                    
                    # Quick save to test output
                    test_output = "outputs/follower_test.csv"
                    non_zero_followers.head(10).to_csv(test_output, index=False)
                    print(f"✅ Test sample saved to: {test_output}")
                    
                    return True
                else:
                    print("❌ No rows with follower data found")
                    return False
            else:
                print("❌ No data processed")
                return False
        else:
            print("❌ No clusters found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_follower_counts()
    print(f"\n{'='*50}")
    if success:
        print("🎉 FOLLOWER COUNT FIX SUCCESSFUL!")
        print("The #Followers, #Followees, and #Posts columns should now be populated.")
    else:
        print("💥 Issue still exists - needs further investigation")
    print("="*50)
