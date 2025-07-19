"""
Instagram User Behavior Analysis: Clean Modular Application
Demonstrates component-based architecture with UI separation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import json
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import core modules
from preprocessing.data_processor import DataProcessor
from preprocessing.clustered_data_processor import ClusteredDataProcessor
from follower_selection.high_value_selector import HighValueFollowerSelector
from sentiment_analysis.bert_analyzer import BERTSentimentAnalyzer
from models.model_trainer import ModelTrainer
from evaluation.model_evaluator import ModelEvaluator
from profiling.profile_generator import ProfileGenerator

# Import UI components
from ui.base import BaseUIComponent
from ui.overview import OverviewComponent
from ui.preprocessing import PreprocessingComponent
from ui.follower_selection import FollowerSelectionComponent
from ui.dataset_analysis import DatasetAnalysisComponent
from ui.sentiment_analysis import SentimentAnalysisComponent
from ui.model_training import ModelTrainingComponent
from ui.model_evaluation import ModelEvaluationComponent
from ui.profile_generation import ProfileGenerationComponent
from ui.visualization import VisualizationComponent
from ui.live_prediction import LivePredictionComponent

# Configure page
st.set_page_config(
    page_title="Instagram Engagement Modeling - Modular",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #E4405F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #405DE6;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .component-info {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ModularInstagramEngagementApp:
    """Clean modular Instagram engagement application"""
    
    def __init__(self):
        """Initialize with component-based architecture"""
        # Core processing modules
        self.data_processor = DataProcessor()
        self.clustered_data_processor = ClusteredDataProcessor()
        self.follower_selector = HighValueFollowerSelector()
        self.sentiment_analyzer = BERTSentimentAnalyzer()
        self.model_trainer = ModelTrainer()
        self.model_evaluator = ModelEvaluator()
        self.profile_generator = ProfileGenerator()
        
        # UI Components (component-based architecture)
        self.overview = OverviewComponent(self)
        self.preprocessing = PreprocessingComponent(self)
        self.follower_selection = FollowerSelectionComponent(self)
        self.dataset_analysis = DatasetAnalysisComponent(self)
        self.sentiment_analysis = SentimentAnalysisComponent(self)
        self.model_training = ModelTrainingComponent(self)
        self.model_evaluation = ModelEvaluationComponent(self)
        self.profile_generation = ProfileGenerationComponent(self)
        self.visualization = VisualizationComponent(self)
        self.live_prediction = LivePredictionComponent(self)
        
        # Page configuration
        self.pages = {
            "🏠 Overview": self.overview,
            "🔧 Data Preprocessing": self.preprocessing,
            "👑 Select High-Value Followers": self.follower_selection,
            "🌐 Dataset-Wide Analysis": self.dataset_analysis,
            "🧠 Sentiment Analysis": self.sentiment_analysis,
            "🤖 Train Models": self.model_training,
            "📈 Evaluate Models": self.model_evaluation,
            "👤 Generate Profiles": self.profile_generation,
            "📋 Visualize Results": self.visualization,
            "🔮 Live Predictions": self.live_prediction
        }
    
    def main(self):
        """Main application interface - clean and modular"""
        # Header
        st.markdown('<h1 class="main-header">📸 Instagram User Behavior Analysis</h1>', 
                   unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem;">Clean Modular Architecture</p>', 
                   unsafe_allow_html=True)
        
        # Architecture info
        with st.expander("🏗️ Modular Architecture Info"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🔧 Core Modules:**")
                st.markdown("- Data Processing")
                st.markdown("- ML Model Training")
                st.markdown("- Sentiment Analysis")
                st.markdown("- Profile Generation")
            
            with col2:
                st.markdown("**🎨 UI Components:**")
                st.markdown("- Modular UI Design")
                st.markdown("- Component Separation")
                st.markdown("- Reusable Elements")
                st.markdown("- Clean Interfaces")
            
            with col3:
                st.markdown("**🚀 Features:**")
                st.markdown("- Live Predictions")
                st.markdown("- Real-time Analysis")
                st.markdown("- Interactive Dashboards")
                st.markdown("- Scalable Architecture")
        
        # Sidebar navigation
        self._show_sidebar()
        
        # Main content area
        self._show_main_content()
    
    def _show_sidebar(self):
        """Sidebar with navigation and status"""
        st.sidebar.markdown("### 🧭 Navigation")
        
        # Page selection
        selected_page = st.sidebar.selectbox(
            "Choose a section:",
            list(self.pages.keys()),
            key="page_selector"
        )
        
        # Store selected page
        st.session_state.selected_page = selected_page
        
        # Pipeline status
        self._show_pipeline_status()
        
        # Component info
        self._show_component_info()
    
    def _show_pipeline_status(self):
        """Show pipeline execution status"""
        st.sidebar.markdown("### 📋 Pipeline Status")
        
        # Check file existence for status
        statuses = {
            "📊 Data Preprocessed": os.path.exists("outputs/preprocessed_data.csv"),
            "👑 High-Value Followers": os.path.exists("outputs/high_value_followers.json"),
            "🧠 Sentiment Analysis": os.path.exists("outputs/sentiment_scores.json"),
            "🤖 Models Trained": os.path.exists("outputs/rf_model.pkl"),
            "📈 Models Evaluated": os.path.exists("outputs/metrics.json"),
            "👤 Profiles Generated": os.path.exists("outputs/profiles.json")
        }
        
        for step, completed in statuses.items():
            if completed:
                st.sidebar.markdown(f'<span class="status-success">✅ {step}</span>', 
                                  unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f'<span class="status-error">❌ {step}</span>', 
                                  unsafe_allow_html=True)
    
    def _show_component_info(self):
        """Show component architecture information"""
        st.sidebar.markdown("### 🏗️ Component Info")
        
        # Component count
        st.sidebar.metric("UI Components", len(self.pages))
        
        # Current component info
        if hasattr(st.session_state, 'selected_page'):
            current_component = self.pages[st.session_state.selected_page]
            component_name = current_component.__class__.__name__
            st.sidebar.markdown(f'<div class="component-info">**Active:** {component_name}</div>', 
                              unsafe_allow_html=True)
    
    def _show_main_content(self):
        """Display main content based on selected page"""
        if hasattr(st.session_state, 'selected_page'):
            selected_page = st.session_state.selected_page
            
            # Get the component
            component = self.pages[selected_page]
            
            # Show component info
            st.markdown(f"### {selected_page}")
            
            # Component description
            component_descriptions = {
                "🏠 Overview": "System overview and architecture visualization",
                "🔧 Data Preprocessing": "Data cleaning and preparation pipeline",
                "👑 Select High-Value Followers": "Identify and select valuable followers for specific accounts",
                "🌐 Dataset-Wide Analysis": "Comprehensive analysis across all accounts with network visualizations",
                "🧠 Sentiment Analysis": "Analyze sentiment in comments and interactions",
                "🤖 Train Models": "Train machine learning models for predictions",
                "📈 Evaluate Models": "Evaluate and compare model performance",
                "👤 Generate Profiles": "Generate user engagement profiles",
                "📋 Visualize Results": "Visualize analysis results and insights",
                "🔮 Live Predictions": "Real-time engagement predictions and optimization"
            }
            
            if selected_page in component_descriptions:
                st.markdown(f"*{component_descriptions[selected_page]}*")
            
            st.markdown("---")
            
            # Show the component
            try:
                component.show()
            except Exception as e:
                st.error(f"Error loading component: {str(e)}")
                st.error("Please check if all required dependencies are installed.")
        else:
            # Default to overview
            self.pages["🏠 Overview"].show()
    
    def generate_sample_data(self):
        """Generate sample Instagram data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample data matching the required schema
        data = {
            'post_id': range(n_samples),
            'owner_id': np.random.randint(1000, 9999, n_samples),
            'likes': np.random.exponential(100, n_samples).astype(int),
            'comments_count': np.random.exponential(20, n_samples).astype(int),
            'comment_text': [f"Sample comment {i}" for i in range(n_samples)],
            'comment_owner_username': [f"user_{i%200}" for i in range(n_samples)],
            'comment_likes': np.random.exponential(5, n_samples).astype(int),
            'media_type': np.random.choice(['photo', 'video', 'album'], n_samples),
            'Category': np.random.choice(['fashion', 'travel', 'food', 'lifestyle', 'tech'], n_samples),
            '#Followers': np.random.exponential(1000, n_samples).astype(int),
        }
        
        return pd.DataFrame(data)

# Initialize and run the app
if __name__ == "__main__":
    app = ModularInstagramEngagementApp()
    app.main()
