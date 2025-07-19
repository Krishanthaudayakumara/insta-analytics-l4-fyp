#!/usr/bin/env python3
"""
Quick test to verify the model evaluation dashboard functionality
"""
import json
import pandas as pd

def test_evaluation_data():
    """Test that evaluation data is accessible and complete"""
    
    targets = ['engagement', 'likes', 'comments']
    total_models = 0
    
    print("🎯 Model Evaluation Dashboard Test")
    print("=" * 50)
    
    for target in targets:
        filename = f'outputs/model_evaluation_{target}.json'
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if data:
                latest_eval = data[-1]['evaluation_results']
                models_count = len(latest_eval)
                total_models += models_count
                
                print(f"\n📊 {target.upper()} MODELS ({models_count} models)")
                print("-" * 40)
                
                for model in latest_eval:
                    model_name = model['Model']
                    mse = model['MSE']
                    r2 = model['R²_Score']
                    mape = model['MAPE_%']
                    within_20 = model['Within_20%_Error']
                    
                    print(f"  {model_name}:")
                    print(f"    MSE: {mse:.4f}")
                    print(f"    R²: {r2:.4f}")
                    print(f"    MAPE: {mape:.1f}%")
                    print(f"    Within 20% Error: {within_20:.1f}%")
                    
                    # Determine performance level
                    if r2 > 0.8:
                        performance = "🟢 Excellent"
                    elif r2 > 0.6:
                        performance = "🟡 Good"
                    elif r2 > 0.4:
                        performance = "🟠 Fair"
                    else:
                        performance = "🔴 Poor"
                    
                    print(f"    Performance: {performance}")
                    print()
                    
        except FileNotFoundError:
            print(f"❌ {target}: No evaluation data found")
        except Exception as e:
            print(f"❌ {target}: Error loading data - {e}")
    
    print("=" * 50)
    print(f"📈 SUMMARY: {total_models} total models evaluated across {len(targets)} targets")
    
    if total_models > 0:
        print("✅ Model evaluation dashboard is ready!")
        print("🚀 You can now view comprehensive metrics in the Streamlit app")
    else:
        print("⚠️  No model evaluations found. Run model training first.")

if __name__ == "__main__":
    test_evaluation_data()
