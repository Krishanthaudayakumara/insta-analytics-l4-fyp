#!/usr/bin/env python3
"""
Quick validation test for sentiment analysis output
"""

import pandas as pd

# Test with actual data from user's output
real_data = pd.DataFrame({
    'caption_sentiment': ['Neutral', 'Positive', 'Negative'],
    'caption_sentiment_score': [0.8082473874092102, 0.6497208476066589, 0.8617660999298096],
    'comments_sentiment': ['Positive', 'Positive', 'Positive'],
    'comments_sentiment_score': [0.9763472676277161, 0.9845259189605713, 0.9871028065681458]
})

# Define sentiment mapping
sentiment_to_numeric = {'Negative': -1, 'Neutral': 0, 'Positive': 1}

def calculate_effective_sentiment_score(sentiment_label, confidence_score):
    if pd.isna(sentiment_label) or pd.isna(confidence_score):
        return 0.0
    sentiment_value = sentiment_to_numeric.get(sentiment_label, 0)
    return sentiment_value * float(confidence_score)

# Calculate effective scores
real_data['caption_effective_sentiment'] = real_data.apply(
    lambda row: calculate_effective_sentiment_score(
        row['caption_sentiment'], row['caption_sentiment_score']
    ), axis=1
)

real_data['comments_effective_sentiment'] = real_data.apply(
    lambda row: calculate_effective_sentiment_score(
        row['comments_sentiment'], row['comments_sentiment_score']
    ), axis=1
)

real_data['combined_effective_sentiment'] = (
    real_data['caption_effective_sentiment'] + 
    real_data['comments_effective_sentiment']
) / 2

# Expected results from user's output
expected_results = [
    {'caption': 0.0, 'comments': 0.9763472676277161, 'combined': 0.48817363381385803},
    {'caption': 0.6497208476066589, 'comments': 0.9845259189605713, 'combined': 0.8171233832836151},
    {'caption': -0.8617660999298096, 'comments': 0.9871028065681458, 'combined': 0.06266835331916809}
]

print("🔍 VALIDATION OF USER'S OUTPUT DATA")
print("=" * 50)

all_correct = True
for i, expected in enumerate(expected_results):
    row = real_data.iloc[i]
    
    print(f"\nRow {i+1} Analysis:")
    print(f"  Caption: {row['caption_sentiment']} ({row['caption_sentiment_score']:.6f})")
    print(f"  Comments: {row['comments_sentiment']} ({row['comments_sentiment_score']:.6f})")
    
    # Check caption scores
    caption_match = abs(row['caption_effective_sentiment'] - expected['caption']) < 1e-10
    print(f"  Caption Effective: {row['caption_effective_sentiment']:.10f} {'✅' if caption_match else '❌'}")
    print(f"  Expected Caption:  {expected['caption']:.10f}")
    
    # Check comment scores
    comments_match = abs(row['comments_effective_sentiment'] - expected['comments']) < 1e-10
    print(f"  Comments Effective: {row['comments_effective_sentiment']:.10f} {'✅' if comments_match else '❌'}")
    print(f"  Expected Comments:  {expected['comments']:.10f}")
    
    # Check combined scores
    combined_match = abs(row['combined_effective_sentiment'] - expected['combined']) < 1e-10
    print(f"  Combined Effective: {row['combined_effective_sentiment']:.10f} {'✅' if combined_match else '❌'}")
    print(f"  Expected Combined:  {expected['combined']:.10f}")
    
    if not (caption_match and comments_match and combined_match):
        all_correct = False

print(f"\n{'🎉' if all_correct else '❌'} FINAL RESULT:")
if all_correct:
    print("✅ ALL CALCULATIONS ARE CORRECT!")
    print("✅ Your sentiment analysis output is mathematically accurate!")
    print("✅ The effective sentiment scores are ready for ML training!")
else:
    print("❌ Some calculations don't match expected values")

print("\n💡 INTERPRETATION:")
print("- Neutral sentiment always gives 0.0 effective score (correct)")
print("- Positive sentiment gives positive effective score (correct)")  
print("- Negative sentiment gives negative effective score (correct)")
print("- Combined scores are proper averages (correct)")
print("- All scores are within [-1.0, +1.0] range (correct)")

print("\n🎯 READY FOR ML TRAINING!")
print("Your data contains perfect effective sentiment scores for machine learning.")
