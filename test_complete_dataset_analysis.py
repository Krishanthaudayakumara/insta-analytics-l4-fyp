#!/usr/bin/env python3
"""
Complete Dataset Analysis Test
Tests both full and simple modes of the dataset analysis system
"""

import sys
import os
sys.path.append('src')

def test_dataset_analysis_imports():
    """Test all required imports"""
    print("🧪 Testing Dataset Analysis Imports...")
    print("-" * 50)
    
    try:
        # Test main modules
        from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
        print("✅ DatasetFollowerAnalyzer import successful")
        
        from follower_selection.network_visualizer import FollowerNetworkVisualizer
        print("✅ FollowerNetworkVisualizer import successful")
        
        # Test UI components
        from ui.dataset_analysis import DatasetAnalysisComponent
        print("✅ DatasetAnalysisComponent import successful")
        
        from ui.simple_dataset_analyzer import SimpleDatasetAnalyzer
        print("✅ SimpleDatasetAnalyzer import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_simple_analyzer():
    """Test the simple analyzer functionality"""
    print("\n🧪 Testing Simple Analyzer...")
    print("-" * 50)
    
    try:
        import pandas as pd
        from ui.simple_dataset_analyzer import SimpleDatasetAnalyzer
        
        # Check if preprocessed data exists
        if not os.path.exists("outputs/preprocessed_data.csv"):
            print("⚠️ No preprocessed data found, skipping simple analyzer test")
            return True
        
        # Load data
        df = pd.read_csv("outputs/preprocessed_data.csv")
        print(f"📊 Loaded data: {len(df)} records")
        
        # Initialize simple analyzer
        analyzer = SimpleDatasetAnalyzer()
        print("✅ Simple analyzer initialized")
        
        # Run analysis (with small parameters for testing)
        results = analyzer.analyze_entire_dataset(
            data=df,
            top_percentage=5,  # Small percentage for testing
            min_followers_per_account=2  # Lower threshold for testing
        )
        
        print("✅ Simple analysis completed successfully!")
        
        # Check results structure
        required_keys = ['metadata', 'account_results', 'cross_account_analysis', 'insights', 'high_value_followers_global']
        for key in required_keys:
            if key in results:
                print(f"✅ {key}: Present")
            else:
                print(f"❌ {key}: Missing")
                return False
        
        # Show summary
        metadata = results['metadata']
        insights = results['insights']
        
        print(f"\n📊 Simple Analysis Results:")
        print(f"   Accounts analyzed: {metadata['total_accounts_analyzed']}")
        print(f"   Total high-value followers: {insights['summary_statistics']['total_high_value_followers']}")
        print(f"   Recommendations: {len(insights['recommendations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_analyzer():
    """Test the full analyzer functionality"""
    print("\n🧪 Testing Full Analyzer...")
    print("-" * 50)
    
    try:
        import pandas as pd
        from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
        from follower_selection.network_visualizer import FollowerNetworkVisualizer
        
        # Check if preprocessed data exists
        if not os.path.exists("outputs/preprocessed_data.csv"):
            print("⚠️ No preprocessed data found, skipping full analyzer test")
            return True
        
        # Load data
        df = pd.read_csv("outputs/preprocessed_data.csv")
        print(f"📊 Loaded data: {len(df)} records")
        
        # Initialize full analyzer
        analyzer = DatasetFollowerAnalyzer()
        visualizer = FollowerNetworkVisualizer()
        print("✅ Full analyzer and visualizer initialized")
        
        # Run analysis (with small parameters for testing)
        results = analyzer.analyze_entire_dataset(
            data=df,
            top_percentage=5,  # Small percentage for testing
            min_followers_per_account=2  # Lower threshold for testing
        )
        
        print("✅ Full analysis completed successfully!")
        
        # Test visualization creation
        print("🎨 Testing visualizations...")
        
        # Test network graph
        network_fig = visualizer.create_network_graph(results)
        print("✅ Network graph created")
        
        # Test similarity heatmap
        similarity_fig = visualizer.create_account_similarity_heatmap(results)
        print("✅ Similarity heatmap created")
        
        # Test insights dashboard
        dashboard_fig = visualizer.create_insights_dashboard(results)
        print("✅ Insights dashboard created")
        
        # Test top performers chart
        performers_fig = visualizer.create_top_performers_chart(results)
        print("✅ Top performers chart created")
        
        # Show summary
        metadata = results['metadata']
        insights = results['insights']
        cross_analysis = results['cross_account_analysis']
        
        print(f"\n📊 Full Analysis Results:")
        print(f"   Accounts analyzed: {metadata['total_accounts_analyzed']}")
        print(f"   Total high-value followers: {insights['summary_statistics']['total_high_value_followers']}")
        print(f"   Power followers: {len(cross_analysis['multi_account_followers'])}")
        print(f"   Cross-pollination rate: {cross_analysis['network_stats']['cross_pollination_rate']:.1%}")
        print(f"   Recommendations: {len(insights['recommendations'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_component():
    """Test the UI component initialization"""
    print("\n🧪 Testing UI Component...")
    print("-" * 50)
    
    try:
        from ui.dataset_analysis import DatasetAnalysisComponent
        from ui.base import BaseUIComponent
        
        # Create a mock app for testing
        class MockApp:
            pass
        
        mock_app = MockApp()
        
        # Initialize component
        component = DatasetAnalysisComponent(mock_app)
        print("✅ DatasetAnalysisComponent initialized successfully")
        
        # Check if component has required methods
        required_methods = ['show', '_run_dataset_analysis', '_display_analysis_results', '_display_simple_analysis_results']
        for method in required_methods:
            if hasattr(component, method):
                print(f"✅ {method}: Present")
            else:
                print(f"❌ {method}: Missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ UI component test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Dataset Analysis System - Complete Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_dataset_analysis_imports),
        ("Simple Analyzer Test", test_simple_analyzer),
        ("Full Analyzer Test", test_full_analyzer),
        ("UI Component Test", test_ui_component)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🚀 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Dataset Analysis System is ready for use!")
        print("\n📋 Usage Instructions:")
        print("1. Run: streamlit run app.py")
        print("2. Navigate to '🌐 Dataset-Wide Analysis'")
        print("3. Configure parameters and run analysis")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
        print("Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
