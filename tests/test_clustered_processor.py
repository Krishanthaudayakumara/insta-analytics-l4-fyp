#!/usr/bin/env python3
"""
Test script for ClusteredDataProcessor
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.clustered_data_processor import ClusteredDataProcessor

def test_clustered_processor():
    print("Testing ClusteredDataProcessor...")
    
    # Initialize processor
    processor = ClusteredDataProcessor()
    
    # Test getting cluster names
    print("\n1. Testing get_cluster_names():")
    cluster_names = processor.get_cluster_names()
    print(f"Found clusters: {cluster_names}")
    
    if not cluster_names:
        print("❌ No clusters found!")
        return False
    
    # Test processing a small cluster
    print(f"\n2. Testing process_cluster() with '{cluster_names[0]}':")
    try:
        df = processor.process_cluster(cluster_names[0])
        print(f"✅ Successfully processed cluster!")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns)}")
        
        if not df.empty:
            print(f"   - Sample data:")
            print(df.head(2))
            
            # Test compatibility transformation
            print("\n3. Testing create_compatible_dataframe():")
            compatible_df = processor.create_compatible_dataframe(df)
            print(f"✅ Successfully created compatible dataframe!")
            print(f"   - Shape: {compatible_df.shape}")
            print(f"   - New columns added: {len(compatible_df.columns) - len(df.columns)}")
            
            # Save test output
            output_path = f"outputs/test_{cluster_names[0]}_processed.csv"
            compatible_df.to_csv(output_path, index=False)
            print(f"✅ Test output saved to: {output_path}")
            
            return True
        else:
            print("❌ Processed dataframe is empty!")
            return False
            
    except Exception as e:
        print(f"❌ Error processing cluster: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clustered_processor()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
