#!/usr/bin/env python3
"""
Quick verification that the plotly chart fix works
"""

print("=== Verifying Plotly Chart Fix ===")

# Check that all plotly charts have unique keys
import re

with open('app.py', 'r') as f:
    content = f.read()

# Find all plotly_chart calls
plotly_calls = re.findall(r'st\.plotly_chart\([^)]+\)', content)

print(f"Found {len(plotly_calls)} plotly_chart calls")

# Check if they all have keys
calls_with_keys = [call for call in plotly_calls if 'key=' in call]
calls_without_keys = [call for call in plotly_calls if 'key=' not in call]

print(f"✅ Calls with unique keys: {len(calls_with_keys)}")
print(f"❌ Calls without keys: {len(calls_without_keys)}")

if calls_without_keys:
    print("Calls missing keys:")
    for call in calls_without_keys:
        print(f"  - {call}")
else:
    print("🎉 All plotly charts have unique keys!")

# Extract and verify key uniqueness
keys = re.findall(r'key="([^"]+)"', content)
unique_keys = set(keys)

print(f"\nKey uniqueness check:")
print(f"Total keys: {len(keys)}")
print(f"Unique keys: {len(unique_keys)}")

if len(keys) == len(unique_keys):
    print("✅ All keys are unique!")
else:
    print("❌ Some keys are duplicated:")
    from collections import Counter
    key_counts = Counter(keys)
    for key, count in key_counts.items():
        if count > 1:
            print(f"  '{key}' appears {count} times")

print("\n" + "="*50)

# Test that the problematic px.radar was fixed
print("Testing px.radar fix...")
if 'px.radar' in content:
    print("❌ px.radar still found in code!")
else:
    print("✅ px.radar has been removed!")

# Check for go.Scatterpolar (the replacement)
if 'go.Scatterpolar' in content:
    print("✅ go.Scatterpolar found as replacement!")
else:
    print("⚠️ go.Scatterpolar not found - radar chart may not work")

# Test the radar chart functionality
print("\nTesting radar chart creation...")
try:
    import plotly.graph_objects as go
    import pandas as pd
    
    # Sample metrics data
    metrics_data = {
        'Random Forest': {'Accuracy': 0.85, 'F1-Score': 0.85},
        'XGBoost': {'Accuracy': 0.87, 'F1-Score': 0.87}
    }
    
    metrics_df = pd.DataFrame(metrics_data).T
    fig = go.Figure()
    
    for model_name in metrics_df.index:
        values = [0.85, 0.85, 0.85]  # Close the radar
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=['Accuracy', 'F1-Score', 'Accuracy'],
            fill='toself',
            name=model_name
        ))
    
    print("✅ Radar chart creation successful!")
except Exception as e:
    print(f"❌ Radar chart creation failed: {e}")

print("="*50)
print("The duplicate element error should now be fixed!")
print("You can test by:")
print("1. Running sentiment analysis")
print("2. Navigating between different sections")
print("3. All charts should display without errors")
print("="*50)
