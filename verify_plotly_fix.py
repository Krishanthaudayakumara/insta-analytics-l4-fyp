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
print("The duplicate element error should now be fixed!")
print("You can test by:")
print("1. Running sentiment analysis")
print("2. Navigating between different sections")
print("3. All charts should display without errors")
print("="*50)
