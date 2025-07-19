#!/usr/bin/env python3
"""
Simple validation check for core functionality
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime

def check_evaluation_files():
    """Check if evaluation files exist and are valid"""
    print("🧪 VALIDATION CHECK - MODEL EVALUATION DASHBOARD")
    print("=" * 55)
    print("\n1. 📊 Evaluation Files Check:")
    
    eval_files = [
        'model_evaluation_engagement.json',
        'model_evaluation_likes.json', 
        'model_evaluation_comments.json'
    ]
    
    all_valid = True
    for file in eval_files:
        path = f'outputs/{file}'
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    print(f"   ✅ {file} - Valid JSON with {len(data)} entries")
            except json.JSONDecodeError:
                print(f"   ❌ {file} - Invalid JSON format")
                all_valid = False
        else:
            print(f"   ❌ {file} - Missing")
            all_valid = False
    
    return all_valid

def check_model_files():
    """Check if model files exist"""
    print("\n2. 🤖 Model Files Check:")
    
    expected_models = [
        'model_linear_regression.joblib',
        'model_random_forest.joblib', 
        'model_ridge.joblib',
        'model_linear_regression_likes.joblib',
        'model_random_forest_likes.joblib',
        'model_ridge_likes.joblib',
        'model_linear_regression_comments.joblib',
        'model_random_forest_comments.joblib',
        'model_ridge_comments.joblib'
    ]
    
    all_exist = True
    for model in expected_models:
        path = f'outputs/{model}'
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            print(f"   ✅ {model} - {file_size} bytes")
        else:
            print(f"   ❌ {model} - Missing")
            all_exist = False
    
    return all_exist

def check_data_files():
    """Check if data files exist"""
    print("\n3. 📁 Data Files Check:")
    
    data_file = 'data/final_with_all_outputs.csv'
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file)
            print(f"   ✅ {data_file} - {len(df)} rows, {len(df.columns)} columns")
            return True
        except Exception as e:
            print(f"   ❌ {data_file} - Error reading: {e}")
            return False
    else:
        print(f"   ❌ {data_file} - Missing")
        return False

def check_core_functionality():
    """Test core functionality"""
    print("\n4. 🔧 Core Functionality Check:")
    
    try:
        # Test importing core modules
        sys.path.append('analysis')
        from engagement_prediction import calculate_comprehensive_metrics
        print("   ✅ engagement_prediction module imports successfully")
        
        import model_evaluation_dashboard
        print("   ✅ model_evaluation_dashboard module imports successfully")
        
        return True
    except Exception as e:
        print(f"   ❌ Core module import failed: {e}")
        return False

def main():
    """Run all validation checks"""
    
    # Change to project directory
    os.chdir('/home/krishantha/Github/fyp-l4')
    
    checks = [
        check_evaluation_files(),
        check_model_files(), 
        check_data_files(),
        check_core_functionality()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"\n📋 VALIDATION SUMMARY:")
    print("=" * 30)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("\n🎉 DASHBOARD IS READY!")
        print("\n🚀 To run the dashboard:")
        print("   streamlit run app.py")
        print("\n📍 Navigate to: Model Performance Evaluation section")
        print("   Use the action buttons to train and evaluate models")
        return True
    else:
        print(f"⚠️  {total - passed} CHECKS FAILED ({passed}/{total})")
        print("\n🔧 Please review and fix the issues above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
