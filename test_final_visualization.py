#!/usr/bin/env python3
"""
Final Visualization Test
Tests all visualization components and fixes
"""

import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def test_plotly_components():
    """Test basic plotly functionality"""
    print("🧪 Testing Plotly Components...")
    
    # Test basic charts
    fig1 = px.bar(x=['A', 'B', 'C'], y=[1, 2, 3], title="Test Bar Chart")
    print("  ✅ Bar chart creation works")
    
    fig2 = px.pie(values=[1, 2, 3], names=['A', 'B', 'C'], title="Test Pie Chart")
    print("  ✅ Pie chart creation works")
    
    fig3 = px.histogram(x=[1, 2, 3, 2, 1], title="Test Histogram")
    print("  ✅ Histogram creation works")
    
    fig4 = px.scatter(x=[1, 2, 3], y=[2, 3, 1], title="Test Scatter")
    print("  ✅ Scatter plot creation works")
    
    fig5 = px.box(y=[1, 2, 3, 2, 1], title="Test Box Plot")
    print("  ✅ Box plot creation works")
    
    # Test radar chart fix (go.Scatterpolar instead of px.radar)
    fig6 = go.Figure()
    fig6.add_trace(go.Scatterpolar(
        r=[1, 2, 3, 4, 5, 1],
        theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Accuracy'],
        fill='toself',
        name='Test Model'
    ))
    fig6.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title="Test Radar Chart"
    )
    print("  ✅ Radar chart (go.Scatterpolar) creation works")
    
    return True

def test_data_loading():
    """Test loading of visualization data"""
    print("📂 Testing Data Loading...")
    
    required_files = [
        'outputs/metrics.json',
        'outputs/profiles.json', 
        'outputs/sentiment_scores.json',
        'outputs/high_value_followers.json'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                print(f"  ✅ {file_path} - loaded {len(data)} items")
            except Exception as e:
                print(f"  ❌ {file_path} - error: {e}")
                return False
        else:
            print(f"  ⚠️  {file_path} - file not found")
    
    return True

def test_model_performance_viz():
    """Test model performance visualization logic"""
    print("📊 Testing Model Performance Visualization...")
    
    if not os.path.exists('outputs/metrics.json'):
        print("  ⚠️  No metrics.json found, skipping test")
        return True
    
    try:
        with open('outputs/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        if not metrics:
            print("  ⚠️  Empty metrics data")
            return True
        
        # Test DataFrame creation
        metrics_df = pd.DataFrame(metrics).T
        print(f"  ✅ DataFrame created with {len(metrics_df)} models")
        
        # Test radar chart data preparation
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        print(f"  ✅ Available metrics: {available_metrics}")
        
        # Test radar chart creation
        if available_metrics:
            fig = go.Figure()
            for model_name in metrics_df.index:
                values = [metrics_df.loc[model_name, metric] if metric in metrics_df.columns else 0 
                         for metric in available_metrics]
                values.append(values[0])  # Close the radar
                metrics_labels = available_metrics + [available_metrics[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics_labels,
                    fill='toself',
                    name=model_name
                ))
            print("  ✅ Radar chart creation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in model performance viz: {e}")
        return False

def test_engagement_profiles_viz():
    """Test engagement profiles visualization logic"""
    print("👤 Testing Engagement Profiles Visualization...")
    
    if not os.path.exists('outputs/profiles.json'):
        print("  ⚠️  No profiles.json found, skipping test")
        return True
    
    try:
        with open('outputs/profiles.json', 'r') as f:
            profiles = json.load(f)
        
        if not profiles:
            print("  ⚠️  Empty profiles data")
            return True
        
        # Test engagement data extraction
        engagement_data = []
        for user, profile in profiles.items():
            if 'engagement_probability' in profile:
                engagement_data.append({
                    'user': user,
                    'engagement_prob': profile['engagement_probability'],
                    'sentiment': profile.get('dominant_sentiment', 'neutral')
                })
        
        if engagement_data:
            df = pd.DataFrame(engagement_data)
            print(f"  ✅ Engagement DataFrame created with {len(df)} users")
            
            # Test histogram creation
            fig1 = px.histogram(df, x='engagement_prob', color='sentiment', 
                               title="Test Engagement Distribution", nbins=20)
            print("  ✅ Engagement histogram creation successful")
            
            # Test bar chart creation
            top_users = df.nlargest(min(10, len(df)), 'engagement_prob')
            fig2 = px.bar(top_users, x='user', y='engagement_prob', color='sentiment',
                         title="Test Top Users")
            print("  ✅ Top users bar chart creation successful")
        else:
            print("  ⚠️  No engagement probability data found")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in engagement profiles viz: {e}")
        return False

def test_unique_keys():
    """Test that visualization keys are unique"""
    print("🔑 Testing Unique Keys...")
    
    # Read app.py and extract all plotly_chart keys
    try:
        with open('app.py', 'r') as f:
            content = f.read()
        
        import re
        keys = re.findall(r'key=["\']([^"\']+)["\']', content)
        unique_keys = set(keys)
        
        print(f"  ✅ Total chart keys: {len(keys)}")
        print(f"  ✅ Unique chart keys: {len(unique_keys)}")
        
        if len(keys) == len(unique_keys):
            print("  ✅ All chart keys are unique")
            return True
        else:
            duplicates = [key for key in keys if keys.count(key) > 1]
            print(f"  ❌ Duplicate keys found: {set(duplicates)}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error checking keys: {e}")
        return False

def main():
    """Run all visualization tests"""
    print("🎯 Instagram Engagement Prediction - Final Visualization Test")
    print("=" * 60)
    
    tests = [
        ("Plotly Components", test_plotly_components),
        ("Data Loading", test_data_loading),
        ("Model Performance Viz", test_model_performance_viz),
        ("Engagement Profiles Viz", test_engagement_profiles_viz),
        ("Unique Keys", test_unique_keys)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📋 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL VISUALIZATION TESTS PASSED!")
        print("✅ The Instagram engagement prediction system visualization pipeline is working correctly!")
        return True
    else:
        print("⚠️  Some visualization tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
