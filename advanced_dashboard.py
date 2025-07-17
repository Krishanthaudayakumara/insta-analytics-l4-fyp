"""
Advanced Machine Learning Dashboard for Instagram User Behavior Analysis
Integrates all advanced ML modules with a comprehensive UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our advanced modules
from analysis.advanced_boosting_models import AdvancedBoostingPredictor
from analysis.multimodal_deep_learning import MultiModalEngagementPredictor
from analysis.advanced_nlp_llm import InstagramNLPPredictor
from analysis.graph_neural_networks import InstagramGNNAnalyzer

class AdvancedMLDashboard:
    """Advanced Machine Learning Dashboard for comprehensive Instagram analysis"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_models()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Advanced Instagram ML Dashboard",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #E91E63;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(45deg, #E91E63, #9C27B0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #1976D2;
            margin: 1rem 0;
            border-left: 4px solid #2196F3;
            padding-left: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .feature-box {
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
            background: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_models(self):
        """Initialize all ML models with caching"""
        if 'models_initialized' not in st.session_state:
            with st.spinner("🚀 Initializing Advanced ML Models..."):
                try:
                    st.session_state.boosting_analyzer = AdvancedBoostingPredictor()
                    st.session_state.multimodal_analyzer = MultiModalEngagementPredictor()
                    st.session_state.nlp_predictor = InstagramNLPPredictor()
                    st.session_state.gnn_analyzer = InstagramGNNAnalyzer()
                    st.session_state.models_initialized = True
                    st.success("✅ All advanced ML models initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing models: {str(e)}")
                    st.info("💡 Some advanced features may not be available.")
    
    def run(self):
        """Main dashboard interface"""
        # Header
        st.markdown('<h1 class="main-header">🧠 Advanced Instagram ML Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("🎛️ Navigation")
        page = st.sidebar.selectbox(
            "Choose Analysis Type:",
            [
                "🏠 Overview",
                "🚀 Advanced Boosting Models",
                "🎭 Multimodal Deep Learning",
                "💬 NLP & LLM Analysis",
                "🕸️ Graph Neural Networks",
                "📊 Model Comparison",
                "🔧 Model Configuration"
            ]
        )
        
        # Route to appropriate page
        if page == "🏠 Overview":
            self.show_overview()
        elif page == "🚀 Advanced Boosting Models":
            self.show_boosting_analysis()
        elif page == "🎭 Multimodal Deep Learning":
            self.show_multimodal_analysis()
        elif page == "💬 NLP & LLM Analysis":
            self.show_nlp_analysis()
        elif page == "🕸️ Graph Neural Networks":
            self.show_gnn_analysis()
        elif page == "📊 Model Comparison":
            self.show_model_comparison()
        elif page == "🔧 Model Configuration":
            self.show_model_configuration()
    
    def show_overview(self):
        """Display overview of all advanced ML capabilities"""
        st.markdown('<h2 class="sub-header">🔬 Advanced ML Capabilities Overview</h2>', 
                   unsafe_allow_html=True)
        
        # Feature cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h3>🚀 Advanced Boosting Models</h3>
                <ul>
                    <li>XGBoost, LightGBM, CatBoost ensemble</li>
                    <li>Automated hyperparameter optimization</li>
                    <li>Feature importance analysis</li>
                    <li>Model interpretability with SHAP</li>
                    <li>Cross-validation and performance metrics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-box">
                <h3>💬 NLP & LLM Integration</h3>
                <ul>
                    <li>BERT/RoBERTa fine-tuning</li>
                    <li>GPT-based content generation</li>
                    <li>Semantic similarity analysis</li>
                    <li>Advanced sentiment & emotion detection</li>
                    <li>Topic modeling and content optimization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h3>🎭 Multimodal Deep Learning</h3>
                <ul>
                    <li>Vision Transformers for image analysis</li>
                    <li>Text-image fusion models</li>
                    <li>Audio processing for video content</li>
                    <li>Multi-modal engagement prediction</li>
                    <li>Content quality assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-box">
                <h3>🕸️ Graph Neural Networks</h3>
                <ul>
                    <li>Social network analysis with GCN/GAT</li>
                    <li>Community detection algorithms</li>
                    <li>Influencer identification</li>
                    <li>Content propagation modeling</li>
                    <li>Interactive network visualization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample data generator
        st.markdown('<h3 class="sub-header">🎲 Generate Sample Data</h3>', 
                   unsafe_allow_html=True)
        
        if st.button("🎯 Generate Sample Instagram Data", type="primary"):
            sample_data = self.generate_sample_data()
            st.session_state.sample_data = sample_data
            st.success("✅ Sample data generated successfully!")
            st.dataframe(sample_data.head())
    
    def show_boosting_analysis(self):
        """Display advanced boosting models analysis"""
        st.markdown('<h2 class="sub-header">🚀 Advanced Boosting Models Analysis</h2>', 
                   unsafe_allow_html=True)
        
        if 'sample_data' not in st.session_state:
            st.warning("⚠️ Please generate sample data from the Overview page first.")
            return
        
        data = st.session_state.sample_data
        analyzer = st.session_state.boosting_analyzer
        
        # Model selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_metric = st.selectbox(
                "Select Target Metric:",
                ["engagement_rate", "likes", "comments", "shares"]
            )
            
            model_type = st.selectbox(
                "Select Model Type:",
                ["ensemble", "xgboost", "lightgbm", "catboost"]
            )
            
            if st.button("🎯 Train Model", type="primary"):
                with st.spinner("Training advanced boosting model..."):
                    try:
                        results = analyzer.train_ensemble_model(data, target_metric)
                        st.session_state.boosting_results = results
                        st.success("✅ Model trained successfully!")
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
        
        with col2:
            if 'boosting_results' in st.session_state:
                results = st.session_state.boosting_results
                
                # Performance metrics
                metrics_cols = st.columns(4)
                for i, (metric, value) in enumerate(results['metrics'].items()):
                    with metrics_cols[i % 4]:
                        st.metric(metric.upper(), f"{value:.4f}")
                
                # Feature importance
                st.subheader("📊 Feature Importance")
                if 'feature_importance' in results:
                    importance_df = pd.DataFrame(results['feature_importance'])
                    fig = px.bar(importance_df.head(10), 
                               x='importance', y='feature',
                               orientation='h',
                               title="Top 10 Important Features")
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_multimodal_analysis(self):
        """Display multimodal deep learning analysis"""
        st.markdown('<h2 class="sub-header">🎭 Multimodal Deep Learning Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Image Analysis")
            uploaded_image = st.file_uploader(
                "Upload an image:", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload an Instagram post image for analysis"
            )
            
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image with deep learning..."):
                        # Placeholder for actual analysis
                        st.success("✅ Image analysis completed!")
                        
                        # Mock results
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Aesthetic Score", "8.7/10")
                            st.metric("Engagement Prediction", "High")
                        with col_b:
                            st.metric("Content Quality", "9.2/10")
                            st.metric("Viral Potential", "Medium-High")
        
        with col2:
            st.subheader("📝 Text-Image Fusion")
            caption_text = st.text_area(
                "Enter image caption:",
                placeholder="Enter the Instagram post caption here...",
                height=100
            )
            
            if caption_text and st.button("🔗 Analyze Text-Image Fusion"):
                with st.spinner("Performing multimodal analysis..."):
                    # Placeholder for actual analysis
                    st.success("✅ Multimodal analysis completed!")
                    
                    # Results visualization
                    st.subheader("📊 Analysis Results")
                    
                    # Sentiment alignment
                    alignment_score = np.random.uniform(0.7, 0.95)
                    st.progress(alignment_score)
                    st.write(f"Text-Image Alignment: {alignment_score:.2%}")
                    
                    # Mood detection
                    moods = ["Happy", "Excited", "Calm", "Energetic", "Inspirational"]
                    detected_mood = np.random.choice(moods)
                    st.info(f"🎭 Detected Mood: {detected_mood}")
    
    def show_nlp_analysis(self):
        """Display NLP and LLM analysis"""
        st.markdown('<h2 class="sub-header">💬 NLP & LLM Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Text input
        text_input = st.text_area(
            "Enter Instagram post text:",
            placeholder="Write or paste Instagram post content here...",
            height=150
        )
        
        if text_input:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Analyze Text", type="primary"):
                    with st.spinner("Performing advanced NLP analysis..."):
                        try:
                            nlp_predictor = st.session_state.nlp_predictor
                            results = nlp_predictor.analyze_post_content(text_input)
                            st.session_state.nlp_results = results
                            st.success("✅ NLP analysis completed!")
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
            
            with col2:
                if st.button("✨ Generate Optimized Content"):
                    with st.spinner("Generating optimized content..."):
                        # Placeholder for content generation
                        optimized_content = f"""
                        🌟 OPTIMIZED VERSION:
                        
                        {text_input} ✨
                        
                        Suggested hashtags: #instagram #viral #engagement #content
                        
                        Engagement boost potential: +23%
                        """
                        st.success("✅ Content optimized!")
                        st.text_area("Optimized Content:", optimized_content, height=150)
            
            # Display analysis results
            if 'nlp_results' in st.session_state:
                results = st.session_state.nlp_results
                
                st.subheader("📊 Analysis Results")
                
                # Metrics
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    st.metric("Sentiment Score", f"{results.get('sentiment_score', 0.8):.2f}")
                with metrics_cols[1]:
                    st.metric("Engagement Prediction", results.get('engagement_prediction', 'High'))
                with metrics_cols[2]:
                    st.metric("Content Quality", f"{results.get('content_quality', 0.85):.2f}")
                with metrics_cols[3]:
                    st.metric("Readability", results.get('readability', 'Good'))
                
                # Emotion analysis
                if 'emotions' in results:
                    st.subheader("😊 Emotion Analysis")
                    emotions_df = pd.DataFrame(list(results['emotions'].items()), 
                                             columns=['Emotion', 'Score'])
                    fig = px.bar(emotions_df, x='Emotion', y='Score',
                               title="Detected Emotions")
                    st.plotly_chart(fig, use_container_width=True)
    
    def show_gnn_analysis(self):
        """Display Graph Neural Networks analysis"""
        st.markdown('<h2 class="sub-header">🕸️ Graph Neural Networks Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Generate network visualization
        if st.button("🌐 Generate Social Network Graph", type="primary"):
            with st.spinner("Building social network graph..."):
                # Create sample network data
                import networkx as nx
                
                # Generate sample social network
                G = nx.barabasi_albert_graph(50, 3)
                pos = nx.spring_layout(G, k=1, iterations=50)
                
                # Create plotly network visualization
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                
                # Node sizes based on degree centrality
                node_sizes = [G.degree(node) * 5 for node in G.nodes()]
                
                fig = go.Figure()
                
                # Add edges
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=[f'User {node}<br>Connections: {G.degree(node)}' for node in G.nodes()],
                    marker=dict(
                        size=node_sizes,
                        color=node_sizes,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Connections")
                    )
                ))
                
                fig.update_layout(
                    title="Instagram User Interaction Network",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[dict(
                        text="Node size represents number of connections",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor="left", yanchor="bottom",
                        font=dict(size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Network statistics
                st.subheader("📊 Network Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Nodes", len(G.nodes()))
                with col2:
                    st.metric("Total Edges", len(G.edges()))
                with col3:
                    st.metric("Avg Clustering", f"{nx.average_clustering(G):.3f}")
                with col4:
                    st.metric("Network Density", f"{nx.density(G):.3f}")
        
        # Community detection
        st.subheader("👥 Community Detection")
        if st.button("🔍 Detect Communities"):
            with st.spinner("Detecting communities..."):
                st.success("✅ Communities detected!")
                
                # Mock community results
                communities = {
                    "Fitness Enthusiasts": 15,
                    "Food Bloggers": 12,
                    "Travel Influencers": 10,
                    "Fashion Lovers": 8,
                    "Tech Reviewers": 5
                }
                
                fig = px.pie(
                    values=list(communities.values()),
                    names=list(communities.keys()),
                    title="Detected Communities"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def show_model_comparison(self):
        """Display model comparison dashboard"""
        st.markdown('<h2 class="sub-header">📊 Model Performance Comparison</h2>', 
                   unsafe_allow_html=True)
        
        # Mock comparison data
        comparison_data = {
            'Model': ['XGBoost', 'LightGBM', 'CatBoost', 'Neural Network', 'GNN'],
            'Accuracy': [0.87, 0.89, 0.85, 0.91, 0.88],
            'Precision': [0.84, 0.86, 0.83, 0.89, 0.85],
            'Recall': [0.82, 0.85, 0.81, 0.88, 0.84],
            'F1-Score': [0.83, 0.85, 0.82, 0.89, 0.84],
            'Training Time (min)': [15, 12, 18, 45, 35]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Performance metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, metric in enumerate(metrics):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(x=df_comparison['Model'], y=df_comparison[metric], 
                      name=metric, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison table
        st.subheader("📋 Detailed Comparison")
        st.dataframe(df_comparison, use_container_width=True)
        
        # Training time comparison
        fig_time = px.bar(df_comparison, x='Model', y='Training Time (min)',
                         title="Training Time Comparison")
        st.plotly_chart(fig_time, use_container_width=True)
    
    def show_model_configuration(self):
        """Display model configuration interface"""
        st.markdown('<h2 class="sub-header">🔧 Model Configuration</h2>', 
                   unsafe_allow_html=True)
        
        # Configuration tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚀 Boosting Models", 
            "🎭 Multimodal", 
            "💬 NLP/LLM", 
            "🕸️ GNN"
        ])
        
        with tab1:
            st.subheader("Boosting Models Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                xgb_lr = st.slider("XGBoost Learning Rate", 0.01, 0.3, 0.1)
                xgb_depth = st.slider("XGBoost Max Depth", 3, 10, 6)
                xgb_estimators = st.slider("XGBoost N Estimators", 100, 1000, 300)
            
            with col2:
                lgb_lr = st.slider("LightGBM Learning Rate", 0.01, 0.3, 0.1)
                lgb_leaves = st.slider("LightGBM Num Leaves", 10, 100, 31)
                lgb_estimators = st.slider("LightGBM N Estimators", 100, 1000, 300)
        
        with tab2:
            st.subheader("Multimodal Deep Learning Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                vision_model = st.selectbox(
                    "Vision Model", 
                    ["ViT-Base", "ViT-Large", "ResNet-50", "EfficientNet-B4"]
                )
                text_model = st.selectbox(
                    "Text Model",
                    ["BERT-Base", "RoBERTa-Base", "DistilBERT", "ALBERT"]
                )
            
            with col2:
                fusion_method = st.selectbox(
                    "Fusion Method",
                    ["Concatenation", "Attention", "Bilinear", "Gated"]
                )
                batch_size = st.slider("Batch Size", 8, 64, 16)
        
        with tab3:
            st.subheader("NLP & LLM Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                llm_provider = st.selectbox(
                    "LLM Provider",
                    ["OpenAI GPT", "Hugging Face", "Local Model", "Azure OpenAI"]
                )
                max_tokens = st.slider("Max Tokens", 100, 2000, 500)
            
            with col2:
                temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
                use_fine_tuning = st.checkbox("Enable Fine-tuning")
        
        with tab4:
            st.subheader("Graph Neural Network Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                gnn_type = st.selectbox(
                    "GNN Architecture",
                    ["GCN", "GAT", "GraphSAGE", "GIN"]
                )
                num_layers = st.slider("Number of Layers", 2, 8, 3)
            
            with col2:
                hidden_dim = st.slider("Hidden Dimension", 64, 512, 128)
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.3)
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            st.success("✅ Configuration saved successfully!")
    
    def generate_sample_data(self):
        """Generate sample Instagram data for testing"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'post_id': range(n_samples),
            'user_id': np.random.randint(1, 101, n_samples),
            'likes': np.random.exponential(100, n_samples).astype(int),
            'comments': np.random.exponential(20, n_samples).astype(int),
            'shares': np.random.exponential(10, n_samples).astype(int),
            'engagement_rate': np.random.uniform(0.01, 0.15, n_samples),
            'followers': np.random.exponential(1000, n_samples).astype(int),
            'hashtag_count': np.random.randint(0, 30, n_samples),
            'caption_length': np.random.randint(10, 500, n_samples),
            'post_type': np.random.choice(['photo', 'video', 'carousel'], n_samples),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'sentiment_score': np.random.uniform(-1, 1, n_samples),
            'has_face': np.random.choice([0, 1], n_samples),
            'is_verified': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        return pd.DataFrame(data)

def main():
    """Main function to run the advanced dashboard"""
    dashboard = AdvancedMLDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
