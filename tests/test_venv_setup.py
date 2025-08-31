#!/usr/bin/env python3
"""
Simple Virtual Environment Test
Quick test to verify venv setup and basic functionality
"""

import sys
import os
print("🧪 Virtual Environment Test")
print("=" * 40)

# Check Python executable
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check if we're in virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✅ Virtual environment detected")
else:
    print("⚠️ Not in virtual environment")

# Test basic imports
print("\n📦 Testing imports...")
try:
    import pandas as pd
    print("✅ Pandas")
except ImportError:
    print("❌ Pandas not available")

try:
    import streamlit as st
    print("✅ Streamlit")
except ImportError:
    print("❌ Streamlit not available")

try:
    import plotly
    print("✅ Plotly")
except ImportError:
    print("❌ Plotly not available")

try:
    import networkx as nx
    print("✅ NetworkX")
except ImportError:
    print("❌ NetworkX not available")

# Test project imports
print("\n🎯 Testing project modules...")
sys.path.append('src')

try:
    from follower_selection.dataset_analyzer import DatasetFollowerAnalyzer
    print("✅ DatasetFollowerAnalyzer")
except ImportError as e:
    print(f"❌ DatasetFollowerAnalyzer: {e}")

try:
    from follower_selection.network_visualizer import FollowerNetworkVisualizer
    print("✅ FollowerNetworkVisualizer")
except ImportError as e:
    print(f"❌ FollowerNetworkVisualizer: {e}")

try:
    from ui.dataset_analysis import DatasetAnalysisComponent
    print("✅ DatasetAnalysisComponent")
except ImportError as e:
    print(f"❌ DatasetAnalysisComponent: {e}")

# Check data availability
print("\n📊 Checking data...")
if os.path.exists("outputs/preprocessed_data.csv"):
    try:
        df = pd.read_csv("outputs/preprocessed_data.csv")
        print(f"✅ Data available: {len(df):,} records")
    except Exception as e:
        print(f"❌ Data load error: {e}")
else:
    print("⚠️ Preprocessed data not found")

print("\n🎉 Test complete!")
