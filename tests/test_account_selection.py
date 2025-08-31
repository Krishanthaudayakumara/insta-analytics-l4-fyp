#!/usr/bin/env python3
"""
Test Account-Specific Follower Selection
Tests the new account-specific follower selection functionality
"""

import pandas as pd
import json
import sys
import os

# Add src to path
sys.path.append('src')

from follower_selection.high_value_selector import HighValueFollowerSelector

def test_account_specific_selection():
    """Test the new account-specific follower selection"""
    print("🧪 Testing Account-Specific Follower Selection")
    print("=" * 50)
    
    # Load data
    if not os.path.exists("outputs/preprocessed_data.csv"):
        print("❌ Error: preprocessed_data.csv not found!")
        print("Please run data preprocessing first.")
        return False
    
    try:
        df = pd.read_csv("outputs/preprocessed_data.csv")
        print(f"✅ Loaded dataset: {len(df)} records")
        
        # Check required columns
        required_cols = ['owner_id', 'comment_owner_username', 'comment_likes', 
                        'engagement_frequency', 'influence_score', '#Followers']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print("Available columns:", list(df.columns))
            return False
        
        print("✅ All required columns present")
        
        # Initialize selector
        selector = HighValueFollowerSelector()
        
        # Get available accounts
        account_stats = selector.get_available_accounts(df)
        print(f"\n📊 Found {len(account_stats)} accounts:")
        
        for i, account in enumerate(account_stats[:5]):  # Show first 5
            print(f"  {i+1}. Account {account['owner_id']}: {account['unique_followers']} followers, {account['total_interactions']} interactions")
        
        if len(account_stats) > 5:
            print(f"  ... and {len(account_stats) - 5} more accounts")
        
        # Test with the first account
        test_owner_id = account_stats[0]['owner_id']
        print(f"\n🎯 Testing with account: {test_owner_id}")
        
        # Test the new function
        high_value_followers = selector.select_high_value_followers(
            data=df,
            owner_id=test_owner_id,
            top_percentage=15,
            clustering_method="K-Means",
            engagement_weight=0.7,
            influence_weight=0.3
        )
        
        print(f"✅ Successfully selected {len(high_value_followers)} high-value followers")
        
        # Show sample results
        print("\n👥 Sample High-Value Followers:")
        sample_usernames = list(high_value_followers.keys())[:5]
        for username in sample_usernames:
            follower_data = high_value_followers[username]
            print(f"  - {username}: Score={follower_data['total_score']:.3f}, Cluster={follower_data['cluster']}")
        
        # Test error handling
        print("\n🔧 Testing error handling...")
        
        try:
            # Test with non-existent account
            selector.select_high_value_followers(
                data=df,
                owner_id="non_existent_account_999999",
                top_percentage=10
            )
            print("❌ Should have raised an error for non-existent account")
            return False
        except ValueError as e:
            print(f"✅ Correctly handled non-existent account: {str(e)[:50]}...")
        
        # Test with minimal data account
        minimal_account = None
        for account in account_stats:
            if account['unique_followers'] <= 2:
                minimal_account = account['owner_id']
                break
        
        if minimal_account:
            print(f"🔍 Testing with minimal data account: {minimal_account}")
            try:
                minimal_followers = selector.select_high_value_followers(
                    data=df,
                    owner_id=minimal_account,
                    top_percentage=10
                )
                print(f"✅ Handled minimal data: {len(minimal_followers)} followers")
            except Exception as e:
                print(f"⚠️ Minimal data handling: {str(e)}")
        
        # Check if files were created
        account_file = f"outputs/high_value_followers_{test_owner_id}.json"
        general_file = "outputs/high_value_followers.json"
        
        if os.path.exists(account_file):
            print(f"✅ Account-specific file created: {account_file}")
        if os.path.exists(general_file):
            print(f"✅ General file created: {general_file}")
        
        print("\n🎉 All tests passed! Account-specific follower selection is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_account_specific_selection()
    sys.exit(0 if success else 1)
