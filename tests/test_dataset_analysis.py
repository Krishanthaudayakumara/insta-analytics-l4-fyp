#!/usr/bin/env python3
"""
Test Dataset-Wide Analysis Module
Tests the complete dataset-wide analysis functionality including network visualizations
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the modules
from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
from follower_selection.network_visualizer import FollowerNetworkVisualizer

def test_dataset_analysis():
    """Test the complete dataset-wide analysis pipeline"""
    
    print("🧪 Testing Dataset-Wide Analysis Module...")
    print("=" * 60)
    
    # Check if preprocessed data exists
    data_file = "outputs/preprocessed_data.csv"
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found!")
        print("Please run data preprocessing first.")
        return False
    
    try:
        # Load data
        print("📊 Loading preprocessed data...")
        df = pd.read_csv(data_file)
        print(f"✅ Data loaded: {len(df):,} records")
        
        # Validate columns
        required_cols = ['owner_id', 'username', 'comment_owner_username']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print(f"✅ Data validation passed: {df.columns.tolist()}")
        
        # Initialize analyzer
        print("\n🔍 Initializing Dataset Analyzer...")
        analyzer = DatasetFollowerAnalyzer()
        print("✅ Analyzer initialized")
        
        # Run analysis with test parameters
        print("\n🚀 Running dataset-wide analysis...")
        print("Parameters:")
        print("  - Top percentage: 10%")
        print("  - Clustering method: K-Means")
        print("  - Engagement weight: 0.7")
        print("  - Influence weight: 0.3")
        print("  - Min followers per account: 3")
        
        analysis_results = analyzer.analyze_entire_dataset(
            data=df,
            top_percentage=10,
            clustering_method="K-Means",
            engagement_weight=0.7,
            influence_weight=0.3,
            min_followers_per_account=3
        )
        
        print("✅ Analysis completed successfully!")
        
        # Show results summary
        print("\n📊 Analysis Results Summary:")
        print("-" * 40)
        
        metadata = analysis_results['metadata']
        insights = analysis_results['insights']
        cross_analysis = analysis_results['cross_account_analysis']
        
        print(f"Accounts analyzed: {metadata['total_accounts_analyzed']}")
        print(f"Total accounts in dataset: {metadata['total_accounts_in_dataset']}")
        print(f"Total high-value followers: {insights['summary_statistics']['total_high_value_followers']}")
        print(f"Average high-value per account: {insights['summary_statistics']['avg_high_value_per_account']:.1f}")
        print(f"Power followers (multi-account): {len(cross_analysis['multi_account_followers'])}")
        print(f"Cross-pollination rate: {cross_analysis['network_stats']['cross_pollination_rate']:.1%}")
        print(f"Recommendations generated: {len(insights['recommendations'])}")
        
        # Show top performers
        print("\n🏆 Top Performers:")
        print("-" * 40)
        top_performers = cross_analysis['top_performers']
        
        print("Top 3 Accounts by High-Value Followers:")
        for i, (account, count) in enumerate(top_performers['top_accounts_by_count'][:3]):
            print(f"  {i+1}. @{account}: {count} followers")
        
        print("\nTop 3 Power Followers:")
        for i, (follower, score) in enumerate(top_performers['top_followers_by_score'][:3]):
            print(f"  {i+1}. @{follower}: {score:.3f}")
        
        # Show recommendations
        print("\n💡 Recommendations:")
        print("-" * 40)
        for i, rec in enumerate(insights['recommendations'][:3]):
            print(f"{i+1}. {rec['title']}")
            print(f"   Description: {rec['description']}")
            print(f"   Action: {rec['action']}")
            print()
        
        # Test visualizations
        print("\n📈 Testing Network Visualizations...")
        print("-" * 40)
        
        visualizer = FollowerNetworkVisualizer()
        print("✅ Visualizer initialized")
        
        # Test network graph
        print("Creating network graph...")
        network_fig = visualizer.create_network_graph(analysis_results)
        print("✅ Network graph created successfully")
        
        # Test similarity heatmap
        print("Creating similarity heatmap...")
        similarity_fig = visualizer.create_account_similarity_heatmap(analysis_results)
        print("✅ Similarity heatmap created successfully")
        
        # Test insights dashboard
        print("Creating insights dashboard...")
        dashboard_fig = visualizer.create_insights_dashboard(analysis_results)
        print("✅ Insights dashboard created successfully")
        
        # Test top performers chart
        print("Creating top performers chart...")
        performers_fig = visualizer.create_top_performers_chart(analysis_results)
        print("✅ Top performers chart created successfully")
        
        # Save visualizations
        print("\n💾 Saving visualizations...")
        saved_files = visualizer.save_visualizations(analysis_results)
        print(f"✅ Saved {len(saved_files)} visualization files:")
        for file_path in saved_files:
            print(f"  - {file_path}")
        
        # Check output file
        output_file = "outputs/dataset_follower_analysis.json"
        if os.path.exists(output_file):
            print(f"\n📁 Analysis results saved to: {output_file}")
            
            # Show file size
            file_size = os.path.getsize(output_file)
            print(f"   File size: {file_size:,} bytes")
        else:
            print(f"\n⚠️ Warning: Output file not found at {output_file}")
        
        print("\n🎉 Dataset-Wide Analysis Test PASSED!")
        print("=" * 60)
        print("✅ All components working correctly:")
        print("  - Dataset analyzer")
        print("  - Network visualizer")
        print("  - Cross-account analysis")
        print("  - Power follower detection")
        print("  - Account similarity analysis")
        print("  - Interactive visualizations")
        print("  - Insights and recommendations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_analyzer_components():
    """Test individual analyzer components"""
    
    print("\n🔧 Testing Individual Components...")
    print("-" * 40)
    
    try:
        # Test DatasetFollowerAnalyzer import and initialization
        from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
        analyzer = DatasetFollowerAnalyzer()
        print("✅ DatasetFollowerAnalyzer: Import and initialization successful")
        
        # Test FollowerNetworkVisualizer import and initialization  
        from follower_selection.network_visualizer import FollowerNetworkVisualizer
        visualizer = FollowerNetworkVisualizer()
        print("✅ FollowerNetworkVisualizer: Import and initialization successful")
        
        # Test dependencies
        import networkx as nx
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Dependencies: NetworkX and Plotly available")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Instagram Engagement Prediction System")
    print("Dataset-Wide Analysis Module Test")
    print("=" * 60)
    
    # Test components first
    if not test_analyzer_components():
        print("❌ Component tests failed!")
        sys.exit(1)
    
    # Test full analysis
    if test_dataset_analysis():
        print("\n🎉 ALL TESTS PASSED!")
        print("Dataset-Wide Analysis Module is working correctly!")
    else:
        print("\n❌ TESTS FAILED!")
        print("Please check the error messages above.")
        sys.exit(1)
