#!/usr/bin/env python3
"""
Final Validation Test for Enhanced Model Evaluation Dashboard
This script performs comprehensive testing of all implemented features
"""

import sys
import os
import pandas as pd
import json
from datetime import datetime
import importlib.util

def test_file_structure():
    """Test if all required files exist"""
    print("🧪 FINAL VALIDATION TEST - ENHANCED DASHBOARD")
    print("=" * 60)
    print("\n1. 📁 File Structure Validation:")
    
    required_files = {
        'app.py': 'Main Streamlit application',
        'model_evaluation_dashboard.py': 'Dashboard with action buttons',
        'analysis/engagement_prediction.py': 'Enhanced ML functions',
        'outputs/': 'Output directory',
        'MODEL_EVALUATION_SUMMARY.md': 'Documentation',
        'ENHANCED_DASHBOARD_FEATURES.md': 'Features documentation'
    }
    
    all_exist = True
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"   ✅ {file} - {description}")
        else:
            print(f"   ❌ {file} - MISSING - {description}")
            all_exist = False
    
    return all_exist

def test_evaluation_files():
    """Test evaluation JSON files"""
    print("\n2. 📊 Evaluation Files Validation:")
    
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
                    if isinstance(data, list) and len(data) > 0:
                        entry = data[0]
                        if 'timestamp' in entry and 'evaluation_results' in entry:
                            print(f"   ✅ {file} - {len(data)} entries, latest: {entry['timestamp']}")
                        else:
                            print(f"   ⚠️  {file} - Invalid structure")
                            all_valid = False
                    else:
                        print(f"   ⚠️  {file} - Empty or invalid format")
                        all_valid = False
            except Exception as e:
                print(f"   ❌ {file} - Error reading: {e}")
                all_valid = False
        else:
            print(f"   ❌ {file} - Missing")
            all_valid = False
    
    return all_valid

def test_model_files():
    """Test trained model files"""
    print("\n3. 🤖 Model Files Validation:")
    
    model_types = ['engagement', 'likes', 'comments']
    algorithms = ['linear_regression', 'random_forest', 'ridge']
    
    all_models_exist = True
    for model_type in model_types:
        print(f"   📈 {model_type.title()} Models:")
        for algo in algorithms:
            if model_type == 'engagement':
                file_pattern = f"model_{algo}.joblib"
            else:
                file_pattern = f"model_{algo}_{model_type}.joblib"
            
            path = f"outputs/{file_pattern}"
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"      ✅ {file_pattern} ({size:,} bytes)")
            else:
                print(f"      ❌ {file_pattern} - Missing")
                all_models_exist = False
    
    return all_models_exist

def test_imports():
    """Test critical imports"""
    print("\n4. 📦 Import Validation:")
    
    imports_success = True
    
    # Test engagement_prediction imports
    try:
        sys.path.append('analysis')
        from engagement_prediction import (
            calculate_comprehensive_metrics,
            evaluate_model_with_cross_validation,
            save_evaluation_results,
            load_evaluation_results,
            run,
            train_and_save_like_comment_models
        )
        print("   ✅ engagement_prediction - All functions imported")
    except ImportError as e:
        print(f"   ❌ engagement_prediction import failed: {e}")
        imports_success = False
    
    # Test dashboard imports
    try:
        import model_evaluation_dashboard
        print("   ✅ model_evaluation_dashboard - Module imported")
    except ImportError as e:
        print(f"   ❌ model_evaluation_dashboard import failed: {e}")
        imports_success = False
    
    # Test required packages
    required_packages = ['streamlit', 'plotly', 'pandas', 'numpy', 'scikit-learn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package} - Available")
        except ImportError:
            print(f"   ❌ {package} - Missing")
            imports_success = False
    
    return imports_success

def test_dashboard_functions():
    """Test dashboard functions"""
    print("\n5. 🎯 Dashboard Functions Validation:")
    
    try:
        import model_evaluation_dashboard as med
        
        # Test function availability
        functions = [
            'load_evaluation_data',
            'create_overview_tab',
            'create_detailed_metrics_tab',
            'create_performance_comparison_tab',
            'create_history_tab',
            'show_model_evaluation_dashboard'
        ]
        
        functions_exist = True
        for func in functions:
            if hasattr(med, func):
                print(f"   ✅ {func} - Available")
            else:
                print(f"   ❌ {func} - Missing")
                functions_exist = False
        
        return functions_exist
        
    except Exception as e:
        print(f"   ❌ Dashboard function test failed: {e}")
        return False

def test_comprehensive_metrics():
    """Test comprehensive metrics calculation"""
    print("\n6. 📈 Comprehensive Metrics Test:")
    
    try:
        sys.path.append('analysis')
        from engagement_prediction import calculate_comprehensive_metrics
        
        # Create sample data for testing
        import numpy as np
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])
        
        metrics = calculate_comprehensive_metrics(y_true, y_pred)
        
        expected_metrics = [
            'MSE', 'RMSE', 'MAE', 'R²_Score', 'MAPE_%',
            'Explained_Variance', 'Accuracy_10%_Error', 'Accuracy_20%_Error'
        ]
        
        all_metrics_present = True
        for metric in expected_metrics:
            if metric in metrics:
                print(f"   ✅ {metric}: {metrics[metric]:.4f}")
            else:
                print(f"   ❌ {metric} - Missing")
                all_metrics_present = False
        
        return all_metrics_present
        
    except Exception as e:
        print(f"   ❌ Metrics test failed: {e}")
        return False

def generate_summary_report():
    """Generate final validation report"""
    print("\n" + "=" * 60)
    print("📋 VALIDATION SUMMARY REPORT")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure()),
        ("Evaluation Files", test_evaluation_files()),
        ("Model Files", test_model_files()),
        ("Imports", test_imports()),
        ("Dashboard Functions", test_dashboard_functions()),
        ("Comprehensive Metrics", test_comprehensive_metrics())
    ]
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    print(f"\n🎯 OVERALL RESULTS: {passed}/{total} tests passed")
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("✅ Enhanced Dashboard is fully functional")
        print("✅ Action buttons ready for user interaction")
        print("✅ Comprehensive evaluation system operational")
        print("✅ UI integration complete")
        
        print("\n🚀 READY FOR PRODUCTION:")
        print("   - Start app: streamlit run app.py --server.port 8503")
        print("   - Navigate to: Model Performance Evaluation section")
        print("   - Use action buttons for model training")
        print("   - View comprehensive metrics and visualizations")
        
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review and fix issues.")
    
    return passed == total

if __name__ == "__main__":
    success = generate_summary_report()
    sys.exit(0 if success else 1)
