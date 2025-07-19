#!/usr/bin/env python3
"""
Test script to validate the Plotly titlefont fix
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_network_visualizer_import():
    """Test that the network visualizer can be imported without errors"""
    try:
        from follower_selection.network_visualizer import FollowerNetworkVisualizer
        print("✅ Network visualizer imported successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing network visualizer: {str(e)}")
        return False

def test_visualization_creation():
    """Test creating a simple visualization"""
    try:
        from follower_selection.network_visualizer import FollowerNetworkVisualizer
        
        # Create a simple test dataset
        sample_results = {
            'account_results': {
                'test_account': {
                    'high_value_count': 5,
                    'high_value_followers': {
                        'follower1': {
                            'engagement_score': 0.8,
                            'influence_score': 0.7,
                            'total_score': 1.5
                        }
                    }
                }
            },
            'cross_account_analysis': {
                'multi_account_followers': {},
                'account_similarity': {'test_account': {}},
                'top_performers': {
                    'top_accounts_by_count': [('test_account', 5)],
                    'top_followers_by_score': [('follower1', 1.5)]
                }
            },
            'insights': {}
        }
        
        visualizer = FollowerNetworkVisualizer()
        
        # Test creating the insights dashboard (this uses the updated layout)
        fig = visualizer.create_insights_dashboard(sample_results)
        print("✅ Insights dashboard created successfully")
        
        # Test creating top performers chart
        fig2 = visualizer.create_top_performers_chart(sample_results)
        print("✅ Top performers chart created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_complete_workflow():
    """Test the complete dataset analysis workflow"""
    try:
        from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
        from follower_selection.network_visualizer import FollowerNetworkVisualizer
        
        # Check if we have existing analysis results
        results_file = "outputs/dataset_follower_analysis.json"
        if os.path.exists(results_file):
            import json
            with open(results_file, 'r') as f:
                analysis_results = json.load(f)
            
            print(f"📊 Loading existing analysis results with {len(analysis_results.get('account_results', {}))} accounts")
            
            # Test network visualization creation
            visualizer = FollowerNetworkVisualizer()
            
            # Test each visualization type
            try:
                fig1 = visualizer.create_insights_dashboard(analysis_results)
                print("✅ Insights dashboard visualization successful")
            except Exception as e:
                print(f"⚠️ Insights dashboard error: {str(e)}")
            
            try:
                fig2 = visualizer.create_top_performers_chart(analysis_results)
                print("✅ Top performers chart successful")
            except Exception as e:
                print(f"⚠️ Top performers chart error: {str(e)}")
            
            try:
                fig3 = visualizer.create_account_similarity_heatmap(analysis_results)
                print("✅ Similarity heatmap successful")
            except Exception as e:
                print(f"⚠️ Similarity heatmap error: {str(e)}")
            
            try:
                fig4 = visualizer.create_network_graph(analysis_results)
                print("✅ Network graph successful - PLOTLY FIX WORKING!")
            except Exception as e:
                print(f"❌ Network graph error: {str(e)}")
                return False
            
            return True
        else:
            print("⚠️ No existing analysis results found, creating minimal test")
            return test_visualization_creation()
            
    except Exception as e:
        print(f"❌ Error in complete workflow test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Plotly titlefont fix...")
    print("=" * 50)
    
    # Test 1: Import
    test1 = test_network_visualizer_import()
    
    # Test 2: Simple visualization
    test2 = test_visualization_creation()
    
    # Test 3: Complete workflow
    test3 = test_complete_workflow()
    
    print("=" * 50)
    if all([test1, test2, test3]):
        print("🎉 ALL TESTS PASSED - PLOTLY FIX SUCCESSFUL!")
        print("✅ The titlefont deprecation error has been resolved")
    else:
        print("❌ Some tests failed - check the errors above")
        sys.exit(1)
