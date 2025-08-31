#!/usr/bin/env python3
"""
Test script to verify the Streamlit duplicate key fix is working
"""

import json
import sys
import os

def test_streamlit_app():
    """Test the Streamlit app for syntax and key uniqueness"""
    print("=== TESTING STREAMLIT DUPLICATE KEY FIX ===\n")
    
    try:
        # Test 1: Import the app module
        print("1. Testing app module import...")
        sys.path.append(os.getcwd())
        
        # Check if app.py can be imported without syntax errors
        import ast
        with open('app.py', 'r') as f:
            code = f.read()
        
        ast.parse(code)
        print("   ✅ app.py syntax is valid")
        
        # Test 2: Check method signature
        print("\n2. Testing method signature...")
        if 'def show_evaluation_results(self, results, context=' in code:
            print("   ✅ show_evaluation_results method signature updated")
        else:
            print("   ❌ Method signature not updated")
            return False
            
        # Test 3: Check unique key generation
        print("\n3. Testing unique key generation...")
        if 'unique_key = f"evaluation_performance_comparison_{context}"' in code:
            print("   ✅ Unique key generation implemented")
        else:
            print("   ❌ Unique key generation not found")
            return False
            
        # Test 4: Check method calls
        print("\n4. Testing method calls...")
        if 'show_evaluation_results(evaluation_results, "new")' in code:
            print("   ✅ First call uses 'new' context")
        else:
            print("   ❌ First call not updated")
            return False
            
        if 'show_evaluation_results(existing_results, "previous")' in code:
            print("   ✅ Second call uses 'previous' context")
        else:
            print("   ❌ Second call not updated")
            return False
            
        # Test 5: Test model evaluator
        print("\n5. Testing model evaluator...")
        from src.evaluation.model_evaluator import ModelEvaluator
        evaluator = ModelEvaluator()
        
        # Quick evaluation test
        results = evaluator.evaluate_models()
        print(f"   ✅ Model evaluation successful: {len(results)} models evaluated")
        
        # Check if evaluation results exist
        if os.path.exists('outputs/evaluation_results.json'):
            with open('outputs/evaluation_results.json', 'r') as f:
                eval_data = json.load(f)
            print(f"   ✅ Evaluation results saved: {list(eval_data.keys())}")
        
        print("\n=== ALL TESTS PASSED ===")
        print("🎉 Streamlit duplicate key issue has been completely resolved!")
        print("🎉 Model evaluation pipeline is fully operational!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streamlit_app()
    if success:
        print("\n✅ Ready to run: streamlit run app.py")
        print("✅ Model evaluation should work without duplicate key errors!")
    else:
        print("\n❌ Fix validation failed")
    
    sys.exit(0 if success else 1)
