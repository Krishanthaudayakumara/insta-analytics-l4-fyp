"""
Network Visualization Module for High-Value Follower Analysis
Creates interactive network graphs and visualizations for dataset-wide analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import json
from collections import defaultdict, Counter
import logging


class FollowerNetworkVisualizer:
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def create_network_graph(self, analysis_results, layout='spring', node_size_factor=1.0):
        """
        Create an interactive network graph showing relationships between accounts and followers
        
        Args:
            analysis_results: Results from DatasetFollowerAnalyzer
            layout: Network layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_factor: Factor to scale node sizes
            
        Returns:
            Plotly figure object
        """
        self.logger.info("🕸️ Creating network visualization...")
        
        try:
            # Create NetworkX graph
            G = nx.Graph()
            
            account_results = analysis_results['account_results']
            multi_account_followers = analysis_results['cross_account_analysis']['multi_account_followers']
            
            # Add account nodes
            for username, data in account_results.items():
                G.add_node(
                    username,
                    node_type='account',
                    followers_count=data['followers_count'],
                    high_value_count=data['high_value_count'],
                    size=max(10, data['high_value_count'] * node_size_factor)
                )
            
            # Add high-value follower nodes and edges
            added_followers = set()
            
            for username, data in account_results.items():
                for follower, follower_data in data['high_value_followers'].items():
                    # Add follower node if not already added
                    if follower not in added_followers:
                        is_power_follower = follower in multi_account_followers
                        account_count = len(multi_account_followers.get(follower, {}).get('accounts', []))
                        
                        G.add_node(
                            follower,
                            node_type='follower',
                            total_score=follower_data['total_score'],
                            engagement_score=follower_data['engagement_score'],
                            influence_score=follower_data['influence_score'],
                            is_power_follower=is_power_follower,
                            account_count=account_count,
                            size=max(5, follower_data['total_score'] * 20 * node_size_factor)
                        )
                        added_followers.add(follower)
                    
                    # Add edge between account and follower
                    G.add_edge(
                        username,
                        follower,
                        weight=follower_data['total_score'],
                        edge_type='follows'
                    )
            
            # Choose layout
            if layout == 'spring':
                pos = nx.spring_layout(G, k=3, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Create Plotly traces
            account_nodes = []
            follower_nodes = []
            power_follower_nodes = []
            edges = []
            
            # Prepare node data
            for node, (x, y) in pos.items():
                node_data = G.nodes[node]
                
                if node_data['node_type'] == 'account':
                    account_nodes.append({
                        'x': x, 'y': y,
                        'text': f"@{node}<br>Followers: {node_data['followers_count']}<br>High-Value: {node_data['high_value_count']}",
                        'size': max(15, node_data['size'])
                    })
                else:  # follower
                    node_info = {
                        'x': x, 'y': y,
                        'text': f"@{node}<br>Score: {node_data['total_score']:.3f}<br>Accounts: {node_data['account_count']}",
                        'size': max(8, node_data['size'])
                    }
                    
                    if node_data['is_power_follower']:
                        power_follower_nodes.append(node_info)
                    else:
                        follower_nodes.append(node_info)
            
            # Prepare edge data
            for edge in G.edges(data=True):
                node1, node2, edge_data = edge
                x0, y0 = pos[node1]
                x1, y1 = pos[node2]
                
                edges.extend([
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=max(0.5, edge_data['weight'] * 2), color='rgba(125,125,125,0.3)'),
                        hoverinfo='none',
                        showlegend=False
                    )
                ])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            for edge_trace in edges:
                fig.add_trace(edge_trace)
            
            # Add account nodes
            if account_nodes:
                fig.add_trace(go.Scatter(
                    x=[node['x'] for node in account_nodes],
                    y=[node['y'] for node in account_nodes],
                    mode='markers+text',
                    marker=dict(
                        size=[node['size'] for node in account_nodes],
                        color='#FF6B6B',
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    text=[f"@{node['text'].split('@')[1].split('<br>')[0]}" for node in account_nodes],
                    textposition='middle center',
                    textfont=dict(size=10, color='white'),
                    hovertext=[node['text'] for node in account_nodes],
                    hoverinfo='text',
                    name='Instagram Accounts'
                ))
            
            # Add regular follower nodes
            if follower_nodes:
                fig.add_trace(go.Scatter(
                    x=[node['x'] for node in follower_nodes],
                    y=[node['y'] for node in follower_nodes],
                    mode='markers',
                    marker=dict(
                        size=[node['size'] for node in follower_nodes],
                        color='#4ECDC4',
                        line=dict(width=1, color='white'),
                        opacity=0.7
                    ),
                    hovertext=[node['text'] for node in follower_nodes],
                    hoverinfo='text',
                    name='High-Value Followers'
                ))
            
            # Add power follower nodes
            if power_follower_nodes:
                fig.add_trace(go.Scatter(
                    x=[node['x'] for node in power_follower_nodes],
                    y=[node['y'] for node in power_follower_nodes],
                    mode='markers',
                    marker=dict(
                        size=[node['size'] for node in power_follower_nodes],
                        color='#FFD93D',
                        line=dict(width=2, color='#FF6B6B'),
                        opacity=0.9
                    ),
                    hovertext=[node['text'] for node in power_follower_nodes],
                    hoverinfo='text',
                    name='Power Followers (Multi-Account)'
                ))
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text='Instagram High-Value Follower Network',
                    font=dict(size=16)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Red nodes: Instagram accounts | Teal nodes: High-value followers | Yellow nodes: Power followers (valuable across multiple accounts)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                    font=dict(color="gray", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                width=1000,
                height=700
            )
            
            self.logger.info("✅ Network graph created successfully")
            return fig
            
        except Exception as e:
            self.logger.error(f"❌ Error creating network graph: {str(e)}")
            raise
    
    def create_account_similarity_heatmap(self, analysis_results):
        """Create a heatmap showing similarity between accounts based on shared followers"""
        
        similarity_matrix = analysis_results['cross_account_analysis']['account_similarity']
        
        # Prepare data for heatmap
        accounts = list(similarity_matrix.keys())
        n_accounts = len(accounts)
        
        if n_accounts == 0:
            return go.Figure().add_annotation(text="No data available for similarity analysis")
        
        # Create similarity matrix
        z_data = []
        hover_text = []
        
        for i, acc1 in enumerate(accounts):
            row_data = []
            row_hover = []
            for j, acc2 in enumerate(accounts):
                if i == j:
                    similarity = 1.0
                    shared = "Same account"
                else:
                    sim_data = similarity_matrix[acc1].get(acc2, {'similarity_score': 0, 'shared_followers': 0})
                    similarity = sim_data['similarity_score']
                    shared = f"Shared followers: {sim_data['shared_followers']}"
                
                row_data.append(similarity)
                row_hover.append(f"@{acc1} ↔ @{acc2}<br>Similarity: {similarity:.3f}<br>{shared}")
            
            z_data.append(row_data)
            hover_text.append(row_hover)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f"@{acc}" for acc in accounts],
            y=[f"@{acc}" for acc in accounts],
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_text,
            colorscale='Viridis',
            colorbar=dict(title="Similarity Score")
        ))
        
        fig.update_layout(
            title='Account Similarity Based on Shared High-Value Followers',
            xaxis_title='Instagram Accounts',
            yaxis_title='Instagram Accounts',
            width=800,
            height=600
        )
        
        return fig
    
    def create_insights_dashboard(self, analysis_results):
        """Create a comprehensive insights dashboard"""
        
        insights = analysis_results['insights']
        cross_analysis = analysis_results['cross_account_analysis']
        account_results = analysis_results['account_results']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'High-Value Followers per Account',
                'Top Power Followers',
                'Account Performance Distribution',
                'Cross-Account Engagement Patterns'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. High-value followers per account
        accounts = list(account_results.keys())
        follower_counts = [data['high_value_count'] for data in account_results.values()]
        
        fig.add_trace(
            go.Bar(
                x=accounts,
                y=follower_counts,
                name='High-Value Followers',
                marker_color='#FF6B6B'
            ),
            row=1, col=1
        )
        
        # 2. Top power followers
        power_followers = cross_analysis['multi_account_followers']
        if power_followers:
            top_power = sorted(
                [(name, data['account_count']) for name, data in power_followers.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            if top_power:
                fig.add_trace(
                    go.Bar(
                        x=[f"@{name}" for name, _ in top_power],
                        y=[count for _, count in top_power],
                        name='Account Count',
                        marker_color='#FFD93D'
                    ),
                    row=1, col=2
                )
        
        # 3. Performance distribution
        fig.add_trace(
            go.Histogram(
                x=follower_counts,
                nbinsx=10,
                name='Distribution',
                marker_color='#4ECDC4'
            ),
            row=2, col=1
        )
        
        # 4. Engagement vs Influence pattern
        all_scores = []
        for data in account_results.values():
            for follower_data in data['high_value_followers'].values():
                all_scores.append({
                    'engagement': follower_data['engagement_score'],
                    'influence': follower_data['influence_score'],
                    'total': follower_data['total_score']
                })
        
        if all_scores:
            fig.add_trace(
                go.Scatter(
                    x=[score['engagement'] for score in all_scores],
                    y=[score['influence'] for score in all_scores],
                    mode='markers',
                    marker=dict(
                        size=[score['total'] * 20 for score in all_scores],
                        color=[score['total'] for score in all_scores],
                        colorscale='Viridis',
                        opacity=0.6
                    ),
                    name='Followers',
                    hovertemplate='Engagement: %{x:.3f}<br>Influence: %{y:.3f}<br>Total Score: %{marker.color:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Dataset-Wide High-Value Follower Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Accounts", row=1, col=1)
        fig.update_yaxes(title_text="Follower Count", row=1, col=1)
        
        fig.update_xaxes(title_text="Power Followers", row=1, col=2)
        fig.update_yaxes(title_text="Account Count", row=1, col=2)
        
        fig.update_xaxes(title_text="High-Value Followers Count", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_xaxes(title_text="Engagement Score", row=2, col=2)
        fig.update_yaxes(title_text="Influence Score", row=2, col=2)
        
        return fig
    
    def create_top_performers_chart(self, analysis_results):
        """Create visualization for top performing accounts and followers"""
        
        top_performers = analysis_results['cross_account_analysis']['top_performers']
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top Accounts by High-Value Followers', 'Top Followers by Average Score'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Top accounts by count
        top_accounts = top_performers['top_accounts_by_count'][:10]
        if top_accounts:
            fig.add_trace(
                go.Bar(
                    x=[f"@{name}" for name, _ in top_accounts],
                    y=[count for _, count in top_accounts],
                    name='High-Value Followers',
                    marker_color='#FF6B6B',
                    text=[count for _, count in top_accounts],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # Top followers by score
        top_followers = top_performers['top_followers_by_score'][:10]
        if top_followers:
            fig.add_trace(
                go.Bar(
                    x=[f"@{name}" for name, _ in top_followers],
                    y=[score for _, score in top_followers],
                    name='Average Score',
                    marker_color='#4ECDC4',
                    text=[f"{score:.3f}" for _, score in top_followers],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="Top Performers Analysis",
            showlegend=False,
            height=500
        )
        
        return fig
    
    def save_visualizations(self, analysis_results, output_dir="outputs/visualizations"):
        """Save all visualizations as HTML files"""
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = {
            'network_graph': self.create_network_graph(analysis_results),
            'similarity_heatmap': self.create_account_similarity_heatmap(analysis_results),
            'insights_dashboard': self.create_insights_dashboard(analysis_results),
            'top_performers': self.create_top_performers_chart(analysis_results)
        }
        
        saved_files = []
        for name, fig in visualizations.items():
            file_path = os.path.join(output_dir, f"{name}.html")
            fig.write_html(file_path)
            saved_files.append(file_path)
            self.logger.info(f"📊 Saved {name} to {file_path}")
        
        return saved_files
