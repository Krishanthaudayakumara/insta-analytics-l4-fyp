"""
UI Components Base Module
Contains base classes and utilities for UI components
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import time
import random
from datetime import datetime


class BaseUIComponent:
    """Base class for UI components"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        
        # Handle None app_instance gracefully
        if app_instance is not None:
            self.data_processor = app_instance.data_processor
            self.clustered_data_processor = app_instance.clustered_data_processor
            self.follower_selector = app_instance.follower_selector
            self.sentiment_analyzer = app_instance.sentiment_analyzer
            self.model_trainer = app_instance.model_trainer
            self.model_evaluator = app_instance.model_evaluator
            self.profile_generator = app_instance.profile_generator
        else:
            # Initialize with None for testing/standalone mode
            self.data_processor = None
            self.clustered_data_processor = None
            self.follower_selector = None
            self.sentiment_analyzer = None
            self.model_trainer = None
            self.model_evaluator = None
            self.profile_generator = None
    
    def show_prerequisites_warning(self, prerequisites):
        """Show warning for missing prerequisites"""
        missing = [name for name, path in prerequisites if not os.path.exists(path)]
        if missing:
            st.warning(f"⚠️ Missing: {', '.join(missing)}")
            return True
        return False
    
    def generate_unique_key(self, base_key, context="default"):
        """Generate unique key for Streamlit components"""
        timestamp = str(time.time()).replace('.', '')
        random_id = random.randint(1000, 9999)
        return f"{base_key}_{context}_{timestamp}_{random_id}"


class UIUtils:
    """Utility functions for UI components"""
    
    @staticmethod
    def check_file_exists(filepath):
        """Check if file exists"""
        return os.path.exists(filepath)
    
    @staticmethod
    def load_json_file(filepath):
        """Load JSON file safely"""
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def save_json_file(data, filepath):
        """Save data to JSON file"""
        try:
            with open(filepath, "w") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            st.error(f"Error saving {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def show_pipeline_status():
        """Show pipeline execution status in sidebar"""
        st.sidebar.markdown("### 📋 Pipeline Status")
        
        statuses = {
            "Data Preprocessed": os.path.exists("outputs/preprocessed_data.csv"),
            "High-Value Followers": os.path.exists("outputs/high_value_followers.json"),
            "Sentiment Analysis": os.path.exists("outputs/sentiment_scores.json"),
            "Models Trained": os.path.exists("outputs/rf_model.pkl"),
            "Models Evaluated": os.path.exists("outputs/metrics.json"),
            "Profiles Generated": os.path.exists("outputs/profiles.json")
        }
        
        for step, completed in statuses.items():
            status_icon = "✅" if completed else "⏳"
            status_class = "status-success" if completed else "status-error"
            st.sidebar.markdown(f'{status_icon} <span class="{status_class}">{step}</span>', 
                              unsafe_allow_html=True)
