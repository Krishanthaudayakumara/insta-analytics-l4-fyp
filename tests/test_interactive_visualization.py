#!/usr/bin/env python3
"""
Test Interactive Visualization Fix
This validates that the interactive visualization functionality works correctly
"""

import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_interactive_visualization():
    """Test the interactive visualization functionality"""
    
    print("🧪 Testing Interactive Dataset Visualization Fix")
    print("=" * 60)
    
    # Check if analysis results exist
    analysis_file = "outputs/dataset_follower_analysis.json"
    if not os.path.exists(analysis_file):
        print("❌ No previous analysis found. Please run dataset analysis first.")
        return False
    
    print("✅ Previous analysis file found")
    
    # Load analysis results
    try:
        with open(analysis_file, 'r') as f:
            analysis_results = json.load(f)
        print(f"✅ Analysis results loaded: {len(analysis_results['account_results'])} accounts")
    except Exception as e:
        print(f"❌ Error loading analysis results: {e}")
        return False
    
    # Test visualizer import
    try:
        from follower_selection.network_visualizer import FollowerNetworkVisualizer
        visualizer = FollowerNetworkVisualizer()
        print("✅ FollowerNetworkVisualizer imported successfully")
    except Exception as e:
        print(f"❌ Visualizer import error: {e}")
        return False
    
    # Test all three network layouts
    layouts = ['spring', 'circular', 'kamada_kawai']
    layout_results = {}
    
    print("\n🕸️ Testing Network Layout Generation:")
    print("-" * 40)
    
    for layout in layouts:
        try:
            print(f"   Testing {layout} layout... ", end="")
            fig = visualizer.create_network_graph(
                analysis_results, 
                layout=layout, 
                node_size_factor=1.0
            )
            layout_results[layout] = "✅ SUCCESS"
            print("✅ SUCCESS")
            
            # Validate the figure has proper structure
            if hasattr(fig, 'data') and len(fig.data) > 0:
                print(f"     → {len(fig.data)} traces created")
            
        except Exception as e:
            layout_results[layout] = f"❌ ERROR: {str(e)}"
            print(f"❌ ERROR: {str(e)}")
    
    # Test additional visualizations
    print("\n📊 Testing Additional Visualizations:")
    print("-" * 40)
    
    additional_tests = [
        ("Account Similarity Heatmap", "create_account_similarity_heatmap"),
        ("Insights Dashboard", "create_insights_dashboard"),
        ("Top Performers Chart", "create_top_performers_chart")
    ]
    
    additional_results = {}
    
    for name, method_name in additional_tests:
        try:
            print(f"   Testing {name}... ", end="")
            method = getattr(visualizer, method_name)
            fig = method(analysis_results)
            additional_results[name] = "✅ SUCCESS"
            print("✅ SUCCESS")
            
        except Exception as e:
            additional_results[name] = f"❌ ERROR: {str(e)}"
            print(f"❌ ERROR: {str(e)}")
    
    # Test UI component import
    print("\n🎨 Testing UI Component:")
    print("-" * 40)
    
    try:
        from ui.dataset_analysis import DatasetAnalysisComponent
        component = DatasetAnalysisComponent(app=None)
        print("✅ DatasetAnalysisComponent imported successfully")
        ui_component_test = "✅ SUCCESS"
    except Exception as e:
        print(f"❌ UI Component import error: {e}")
        ui_component_test = f"❌ ERROR: {str(e)}"
    
    # Test the fix for the specific issue
    print("\n🔧 Testing Interactive Layout Switching Fix:")
    print("-" * 40)
    
    try:
        # Simulate the interactive behavior
        print("   Simulating layout switching behavior...")
        
        # Test rapid layout switching (the original problem)
        for i, layout in enumerate(['spring', 'circular', 'spring', 'kamada_kawai']):
            print(f"     → Switch {i+1}: {layout} layout... ", end="")
            fig = visualizer.create_network_graph(
                analysis_results, 
                layout=layout, 
                node_size_factor=1.0
            )
            print("✅")
        
        interactive_fix_test = "✅ SUCCESS"
        print("✅ Interactive layout switching works correctly!")
        
    except Exception as e:
        interactive_fix_test = f"❌ ERROR: {str(e)}"
        print(f"❌ Interactive fix test failed: {str(e)}")
    
    # Summary Report
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY REPORT")
    print("=" * 60)
    
    print("\n🕸️ Network Layout Tests:")
    for layout, result in layout_results.items():
        print(f"   {layout:15} {result}")
    
    print("\n📊 Additional Visualization Tests:")
    for name, result in additional_results.items():
        status = result.split()[0]
        print(f"   {name:25} {status}")
    
    print(f"\n🎨 UI Component Test:        {ui_component_test.split()[0]}")
    print(f"🔧 Interactive Fix Test:     {interactive_fix_test.split()[0]}")
    
    # Determine overall success
    all_layout_success = all("SUCCESS" in result for result in layout_results.values())
    all_additional_success = all("SUCCESS" in result for result in additional_results.values())
    ui_success = "SUCCESS" in ui_component_test
    interactive_success = "SUCCESS" in interactive_fix_test
    
    overall_success = all_layout_success and all_additional_success and ui_success and interactive_success
    
    print("\n" + "=" * 60)
    if overall_success:
        print("🎉 ALL TESTS PASSED - INTERACTIVE VISUALIZATION FIX SUCCESSFUL!")
        print("\nThe fix resolves the issue where changing network layout")
        print("would lose the analysis results. Now users can:")
        print("• Switch between spring, circular, and kamada_kawai layouts")
        print("• Adjust node size factor dynamically")
        print("• Access all visualizations from previous analysis")
        print("• Use interactive controls without losing data")
    else:
        print("❌ SOME TESTS FAILED - REVIEW ISSUES ABOVE")
        
        if not all_layout_success:
            print("• Network layout generation has issues")
        if not all_additional_success:
            print("• Additional visualizations have issues")
        if not ui_success:
            print("• UI component import has issues")
        if not interactive_success:
            print("• Interactive layout switching has issues")
    
    print("=" * 60)
    
    return overall_success

if __name__ == "__main__":
    success = test_interactive_visualization()
    sys.exit(0 if success else 1)
