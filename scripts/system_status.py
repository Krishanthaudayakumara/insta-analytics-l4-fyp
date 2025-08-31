#!/usr/bin/env python3
"""
System Status Check
Quick verification that all components are ready
"""

import os
import sys
import subprocess

def check_streamlit_status():
    """Check if Streamlit is running"""
    try:
        result = subprocess.run(['curl', '-s', 'http://localhost:8501'], 
                              capture_output=True, text=True, timeout=5)
        if 'Streamlit' in result.stdout or result.returncode == 0:
            return "✅ RUNNING"
        else:
            return "❌ NOT RUNNING"
    except:
        return "❌ NOT ACCESSIBLE"

def check_file_structure():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'src/preprocessing/data_processor.py',
        'src/follower_selection/high_value_selector.py',
        'src/sentiment_analysis/bert_analyzer.py',
        'src/models/model_trainer.py',
        'src/evaluation/model_evaluator.py',
        'src/profiling/profile_generator.py',
        'data/final_with_all_outputs.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def check_dependencies():
    """Check if key dependencies are available"""
    try:
        import pandas
        import numpy
        import sklearn
        import streamlit
        import plotly
        return "✅ INSTALLED"
    except ImportError as e:
        return f"❌ MISSING: {e}"

def main():
    print("=" * 60)
    print("📸 INSTAGRAM ENGAGEMENT MODELING SYSTEM STATUS")
    print("=" * 60)
    
    # Check Streamlit app
    print(f"\n🌐 Streamlit Application: {check_streamlit_status()}")
    if check_streamlit_status() == "✅ RUNNING":
        print("   URL: http://localhost:8501")
    
    # Check file structure
    print(f"\n📁 File Structure:", end=" ")
    missing = check_file_structure()
    if not missing:
        print("✅ COMPLETE")
    else:
        print("❌ INCOMPLETE")
        for file in missing:
            print(f"   Missing: {file}")
    
    # Check dependencies
    print(f"\n📦 Dependencies: {check_dependencies()}")
    
    # Check data
    print(f"\n📊 Dataset:", end=" ")
    if os.path.exists('data/final_with_all_outputs.csv'):
        try:
            import pandas as pd
            df = pd.read_csv('data/final_with_all_outputs.csv')
            print(f"✅ READY ({df.shape[0]:,} rows, {df.shape[1]} columns)")
        except:
            print("❌ CORRUPTED")
    else:
        print("❌ NOT FOUND")
    
    # Overall status
    streamlit_ok = check_streamlit_status() == "✅ RUNNING"
    files_ok = len(check_file_structure()) == 0
    deps_ok = "✅" in check_dependencies()
    data_ok = os.path.exists('data/final_with_all_outputs.csv')
    
    print("\n" + "=" * 60)
    if all([streamlit_ok, files_ok, deps_ok, data_ok]):
        print("🎉 SYSTEM STATUS: ✅ FULLY OPERATIONAL")
        print("\n🚀 READY FOR:")
        print("   • High-value follower identification")
        print("   • ML-driven engagement prediction")
        print("   • BERT sentiment analysis")
        print("   • Individual profile generation")
        print("   • Real-time dashboard interaction")
        print("\n💡 Next Steps:")
        print("   1. Open http://localhost:8501 in your browser")
        print("   2. Upload your Instagram dataset")
        print("   3. Follow the step-by-step pipeline")
        print("   4. Generate personalized engagement profiles")
    else:
        print("⚠️  SYSTEM STATUS: PARTIAL")
        print("\n🔧 Issues to resolve:")
        if not streamlit_ok:
            print("   • Start Streamlit app: streamlit run app.py")
        if not files_ok:
            print("   • Some source files are missing")
        if not deps_ok:
            print("   • Install dependencies: pip install -r requirements.txt")
        if not data_ok:
            print("   • Dataset not found in data/ directory")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
