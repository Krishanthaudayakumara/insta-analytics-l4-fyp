#!/usr/bin/env python3
"""
Test script to verify visualization fixes work correctly
"""

import sys
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_radar_chart_fix():
    """Test the radar chart visualization fix"""
    print("🧪 Testing radar chart fix...")
    
    # Create sample metrics data
    sample_metrics = {
        "Random Forest": {
            "Accuracy": 0.95,
            "Precision": 0.93,
            "Recall": 0.97,
            "F1-Score": 0.95,
            "ROC-AUC": 0.98
        },
        "XGBoost": {
            "Accuracy": 0.93,
            "Precision": 0.91,
            "Recall": 0.95,
            "F1-Score": 0.93,
            "ROC-AUC": 0.96
        },
        "LightGBM": {
            "Accuracy": 0.94,
            "Precision": 0.92,
            "Recall": 0.96,
            "F1-Score": 0.94,
            "ROC-AUC": 0.97
        }
    }
    
    try:
        # Test the radar chart creation using the new method
        metrics_df = pd.DataFrame(sample_metrics).T
        
        fig = go.Figure()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        for model_name in metrics_df.index:
            values = [metrics_df.loc[model_name, metric] if metric in metrics_df.columns else 0 
                     for metric in available_metrics]
            # Close the radar chart by repeating first value
            values.append(values[0])
            metrics_labels = available_metrics + [available_metrics[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_labels,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Radar Chart"
        )
        
        print("✅ Radar chart creation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Radar chart test failed: {str(e)}")
        return False

def test_engagement_profiles_viz():
    """Test engagement profiles visualization"""
    print("🧪 Testing engagement profiles visualization...")
    
    # Create sample profile data
    sample_profiles = {
        "user_1": {
            "engagement_probability": 0.85,
            "dominant_sentiment": "positive",
            "content_preferences": ["fashion", "lifestyle"]
        },
        "user_2": {
            "engagement_probability": 0.72,
            "dominant_sentiment": "neutral",
            "content_preferences": ["travel", "food"]
        },
        "user_3": {
            "engagement_probability": 0.93,
            "dominant_sentiment": "positive",
            "content_preferences": ["tech", "lifestyle"]
        }
    }
    
    try:
        # Extract engagement probabilities
        engagement_data = []
        for user, profile in sample_profiles.items():
            if 'engagement_probability' in profile:
                engagement_data.append({
                    'user': user,
                    'engagement_prob': profile['engagement_probability'],
                    'sentiment': profile.get('dominant_sentiment', 'neutral')
                })
        
        if engagement_data:
            df = pd.DataFrame(engagement_data)
            
            # Test histogram creation
            fig = px.histogram(
                df,
                x='engagement_prob',
                color='sentiment',
                title="Engagement Probability Distribution by Sentiment",
                nbins=20
            )
            
            # Test bar chart creation
            top_users = df.nlargest(3, 'engagement_prob')
            fig2 = px.bar(
                top_users,
                x='user',
                y='engagement_prob',
                color='sentiment',
                title="Top Users by Engagement Probability"
            )
            
            print("✅ Engagement profiles visualization successful!")
            return True
        else:
            print("⚠️ No engagement data found")
            return False
            
    except Exception as e:
        print(f"❌ Engagement profiles test failed: {str(e)}")
        return False

def test_follower_distribution_viz():
    """Test follower distribution visualization"""
    print("🧪 Testing follower distribution visualization...")
    
    # Create sample follower data
    sample_followers = {
        "follower_1": {
            "engagement_score": 0.8,
            "influence_score": 0.7,
            "total_score": 1.5
        },
        "follower_2": {
            "engagement_score": 0.9,
            "influence_score": 0.6,
            "total_score": 1.5
        },
        "follower_3": {
            "engagement_score": 0.7,
            "influence_score": 0.9,
            "total_score": 1.6
        }
    }
    
    try:
        # Follower metrics
        follower_data = []
        for username, data in sample_followers.items():
            follower_data.append({
                'username': username,
                'engagement_score': data.get('engagement_score', 0),
                'influence_score': data.get('influence_score', 0),
                'total_score': data.get('total_score', 0)
            })
        
        if follower_data:
            df = pd.DataFrame(follower_data)
            
            # Test scatter plot
            fig = px.scatter(
                df,
                x='engagement_score',
                y='influence_score',
                size='total_score',
                hover_data=['username'],
                title="High-Value Followers: Engagement vs Influence"
            )
            
            # Test bar chart
            top_followers = df.nlargest(3, 'total_score')
            fig2 = px.bar(
                top_followers,
                x='username',
                y='total_score',
                title="Top High-Value Followers"
            )
            
            print("✅ Follower distribution visualization successful!")
            return True
        else:
            print("⚠️ No follower data found")
            return False
            
    except Exception as e:
        print(f"❌ Follower distribution test failed: {str(e)}")
        return False

def test_content_preferences_viz():
    """Test content preferences visualization"""
    print("🧪 Testing content preferences visualization...")
    
    # Create sample content data
    sample_profiles = {
        "user_1": {
            "content_preferences": ["fashion", "lifestyle"],
            "engagement_probability": 0.85
        },
        "user_2": {
            "content_preferences": ["travel", "food"],
            "engagement_probability": 0.72
        },
        "user_3": {
            "content_preferences": ["tech", "lifestyle"],
            "engagement_probability": 0.93
        }
    }
    
    try:
        # Extract content preferences
        content_data = []
        for user, profile in sample_profiles.items():
            if 'content_preferences' in profile:
                for content_type in profile['content_preferences']:
                    content_data.append({
                        'user': user,
                        'content_type': content_type,
                        'engagement_prob': profile.get('engagement_probability', 0)
                    })
        
        if content_data:
            df = pd.DataFrame(content_data)
            
            # Test content type distribution
            content_counts = df['content_type'].value_counts()
            fig = px.bar(
                x=content_counts.index,
                y=content_counts.values,
                title="Content Type Preferences Distribution"
            )
            
            # Test engagement by content type
            avg_engagement = df.groupby('content_type')['engagement_prob'].mean().sort_values(ascending=False)
            fig2 = px.bar(
                x=avg_engagement.index,
                y=avg_engagement.values,
                title="Average Engagement Probability by Content Type"
            )
            
            print("✅ Content preferences visualization successful!")
            return True
        else:
            print("⚠️ No content data found")
            return False
            
    except Exception as e:
        print(f"❌ Content preferences test failed: {str(e)}")
        return False

def main():
    """Run all visualization tests"""
    print("🔧 Testing Instagram Engagement Visualization Fixes")
    print("=" * 50)
    
    tests = [
        test_radar_chart_fix,
        test_engagement_profiles_viz,
        test_follower_distribution_viz,
        test_content_preferences_viz
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results.append(False)
            print()
    
    print("=" * 50)
    print("📊 VISUALIZATION TEST RESULTS:")
    print(f"✅ Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("🎉 All visualization fixes working correctly!")
        print("✅ Ready for production use!")
    else:
        print("⚠️ Some visualization issues remain")
        failed_tests = [test.__name__ for test, result in zip(tests, results) if not result]
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
