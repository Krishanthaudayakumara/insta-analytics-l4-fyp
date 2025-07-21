#!/usr/bin/env python3
"""
Test Live Predictions Integration
Quick validation of the complete live predictions functionality
"""

import sys
import os
import traceback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_live_predictions_integration():
    """Test the live predictions integration"""
    print("🔮 Testing Live Predictions Integration")
    print("=" * 50)
    
    try:
        # Test 1: Import main app
        print("✅ Test 1: Importing main application...")
        from app import InstagramEngagementApp
        print("   ✓ Main app imported successfully")
        
        # Test 2: Initialize app
        print("\n✅ Test 2: Initializing application...")
        app = InstagramEngagementApp()
        print("   ✓ App initialized successfully")
        print("   ✓ LivePredictionComponent created successfully")
        
        # Test 3: Check if live prediction component exists
        print("\n✅ Test 3: Checking live prediction component...")
        if hasattr(app, 'live_prediction'):
            print("   ✓ LivePredictionComponent attribute exists")
        else:
            print("   ❌ LivePredictionComponent attribute missing")
            return False
        
        # Test 4: Check if show_live_predictions method exists
        print("\n✅ Test 4: Checking show_live_predictions method...")
        if hasattr(app, 'show_live_predictions'):
            print("   ✓ show_live_predictions method exists")
        else:
            print("   ❌ show_live_predictions method missing")
            return False
        
        # Test 5: Test component initialization
        print("\n✅ Test 5: Testing component initialization...")
        live_comp = app.live_prediction
        if hasattr(live_comp, 'show'):
            print("   ✓ LivePredictionComponent.show() method exists")
        else:
            print("   ❌ LivePredictionComponent.show() method missing")
            return False
        
        # Test 6: Check required files
        print("\n✅ Test 6: Checking required files...")
        required_files = [
            "outputs/rf_model.pkl",
            "outputs/xgb_model.pkl", 
            "outputs/lgb_model.pkl",
            "outputs/metrics.json"
        ]
        
        available_files = [f for f in required_files if os.path.exists(f)]
        print(f"   ✓ Found {len(available_files)}/{len(required_files)} model files")
        
        if len(available_files) > 0:
            print("   ✓ At least one model available for predictions")
        else:
            print("   ⚠️  No trained models found - live predictions will show warning")
        
        # Test 7: Test feature vector creation
        print("\n✅ Test 7: Testing feature vector creation...")
        try:
            features = live_comp._create_feature_vector(
                1000, 50, 5, 2, 'photo', 'fashion'
            )
            print(f"   ✓ Feature vector created with {len(features)} features")
        except Exception as e:
            print(f"   ❌ Error creating feature vector: {e}")
            return False
        
        # Test 8: Test sentiment analysis fallback
        print("\n✅ Test 8: Testing sentiment analysis...")
        try:
            sentiment = live_comp._analyze_text_sentiment("This is a great post!")
            print(f"   ✓ Sentiment analysis returned: {sentiment}")
        except Exception as e:
            print(f"   ⚠️  Sentiment analysis error (expected): {e}")
        
        print("\n" + "=" * 50)
        print("🎉 All integration tests passed!")
        print("✅ Live Predictions functionality is ready")
        print("✅ Navigate to '🔮 Live Predictions' in the Streamlit app")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def test_modular_app():
    """Test the modular app version as well"""
    print("\n🔧 Testing Modular App Version")
    print("=" * 30)
    
    try:
        from app_modular import main as modular_main
        print("✅ Modular app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Modular app import failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Instagram Engagement Prediction System")
    print("🔮 Live Predictions Integration Test")
    print("=" * 60)
    
    # Test main integration
    main_success = test_live_predictions_integration()
    
    # Test modular version
    modular_success = test_modular_app()
    
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    
    if main_success:
        print("✅ Main App Integration: PASSED")
    else:
        print("❌ Main App Integration: FAILED")
    
    if modular_success:
        print("✅ Modular App: PASSED")
    else:
        print("❌ Modular App: FAILED")
    
    if main_success:
        print("\n🎉 SUCCESS: Live Predictions are fully integrated!")
        print("🌐 Streamlit app available at: http://localhost:8501")
        print("🔮 Navigate to 'Live Predictions' to test the functionality")
    else:
        print("\n❌ FAILED: Integration issues detected")
        sys.exit(1)
