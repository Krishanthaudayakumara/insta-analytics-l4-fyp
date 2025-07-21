"""
Overview UI Component
Handles the main overview page display
"""

import streamlit as st
import plotly.graph_objects as go
from .base import BaseUIComponent


class OverviewComponent(BaseUIComponent):
    """Overview page component"""
    
    def show(self):
        """Show project overview and capabilities"""
        st.markdown("### 🎯 Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🚀 Key Features:**
            - **Individual Engagement Profiles**: Granular profiles for high-value followers
            - **Advanced ML Models**: Random Forest, XGBoost, LightGBM, TabNet, GNN, BERT
            - **Sentiment Integration**: BERT-based sentiment analysis
            - **High-Value Focus**: Target top 10% followers by engagement/influence
            """)
            
            st.markdown("""
            **🔬 Novel Contributions:**
            - Individual vs. group-based segmentation
            - High-value follower prioritization
            - Sentiment-engagement integration
            - Advanced ML model comparison
            """)
        
        with col2:
            st.markdown("""
            **📊 Features Used:**
            - `media_type`: Content type preference
            - `Category`: Content theme preference  
            - `likes`: Aggregate engagement
            - `comments_count`: Aggregate engagement
            - `comment_text`: Sentiment analysis input
            - `comment_owner_username`: Follower ID
            - `comment_likes`: Engagement frequency
            - `#Followers`: Influence scoring
            """)
        
        # Architecture diagram
        st.markdown("### 🏗️ System Architecture")
        self._show_architecture_diagram()
        
        # Quick start button
        st.markdown("### 🚀 Quick Start")
        if st.button("🎯 Start Full Pipeline", type="primary"):
            st.info("💡 Use the sidebar to navigate through each pipeline step!")
    
    def _show_architecture_diagram(self):
        """Show system architecture flow"""
        fig = go.Figure()
        
        # Create flow diagram
        steps = [
            "Dataset\n(data.csv)",
            "Preprocessing\nModule",
            "Follower Selection\nModule",
            "Sentiment Analysis\nModule", 
            "Model Training\nModule",
            "Model Evaluation\nModule",
            "Profile Generation\nModule",
            "Streamlit\nInterface"
        ]
        
        y_positions = [7, 6, 5, 4, 3, 2, 1, 0]
        
        for i, (step, y) in enumerate(zip(steps, y_positions)):
            fig.add_shape(
                type="rect",
                x0=0, y0=y-0.3, x1=2, y1=y+0.3,
                fillcolor="lightblue",
                line=dict(color="blue", width=2)
            )
            fig.add_annotation(
                x=1, y=y,
                text=step,
                showarrow=False,
                font=dict(size=12, color="black")
            )
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=1, y=y-0.5,
                    ax=1, ay=y-0.7,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="black"
                )
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True, key="architecture_diagram")
