"""
High-Value Follower Selection Module
Implements K-Means clustering to identify top 10% high-value followers
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json
import logging

class HighValueFollowerSelector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def select_followers(self, df, top_percent=10, method="K-Means", 
                        engagement_weight=0.7, influence_weight=0.3):
        """
        Select high-value followers using clustering algorithms
        
        Args:
            df: Preprocessed Instagram dataset
            top_percent: Percentage of top followers to select
            method: Clustering method ('K-Means', 'DBSCAN', 'Hierarchical')
            engagement_weight: Weight for engagement features
            influence_weight: Weight for influence features
            
        Returns:
            Dictionary of high-value followers with their scores
        """
        self.logger.info(f"Selecting top {top_percent}% followers using {method}")
        
        # Prepare features for clustering
        cluster_features = self._prepare_clustering_features(df)
        
        # Apply clustering
        if method == "K-Means":
            cluster_labels = self._apply_kmeans(cluster_features)
        elif method == "DBSCAN":
            cluster_labels = self._apply_dbscan(cluster_features)
        elif method == "Hierarchical":
            cluster_labels = self._apply_hierarchical(cluster_features)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Calculate engagement and influence scores
        user_scores = self._calculate_user_scores(
            df, engagement_weight, influence_weight
        )
        
        # Add cluster information
        user_scores['cluster'] = cluster_labels
        
        # Select high-value followers
        high_value_followers = self._select_top_followers(
            user_scores, top_percent
        )
        
        self.logger.info(f"Selected {len(high_value_followers)} high-value followers")
        return high_value_followers
    
    def _prepare_clustering_features(self, df):
        """Prepare features for clustering"""
        self.logger.info("Preparing clustering features...")
        
        # Aggregate user-level features
        user_features = df.groupby('comment_owner_username').agg({
            'comment_likes': ['mean', 'sum', 'count'],
            'engagement_frequency': 'first',
            'influence_score': 'first',
            '#Followers': 'first',
            'content_interaction': 'mean',
            'comment_engagement_ratio': 'mean'
        }).reset_index()
        
        # Flatten column names
        user_features.columns = [
            'username', 'avg_comment_likes', 'total_comment_likes', 'comment_count',
            'engagement_frequency', 'influence_score', 'followers',
            'avg_content_interaction', 'avg_comment_ratio'
        ]
        
        # Select features for clustering
        feature_columns = [
            'avg_comment_likes', 'total_comment_likes', 'comment_count',
            'engagement_frequency', 'influence_score', 'followers',
            'avg_content_interaction', 'avg_comment_ratio'
        ]
        
        # Handle missing values
        user_features[feature_columns] = user_features[feature_columns].fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(user_features[feature_columns])
        
        return scaled_features, user_features
    
    def _apply_kmeans(self, features):
        """Apply K-Means clustering"""
        scaled_features, user_features = features
        
        # Determine optimal number of clusters using elbow method
        optimal_k = self._find_optimal_clusters(scaled_features)
        
        # Apply K-Means
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(scaled_features)
        
        self.logger.info(f"K-Means clustering completed with {optimal_k} clusters")
        return cluster_labels
    
    def _apply_dbscan(self, features):
        """Apply DBSCAN clustering"""
        scaled_features, user_features = features
        
        # Apply DBSCAN with automatic parameter selection
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(scaled_features)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        self.logger.info(f"DBSCAN clustering completed with {n_clusters} clusters")
        return cluster_labels
    
    def _apply_hierarchical(self, features):
        """Apply Hierarchical clustering"""
        from sklearn.cluster import AgglomerativeClustering
        
        scaled_features, user_features = features
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(scaled_features)
        
        # Apply Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
        cluster_labels = hierarchical.fit_predict(scaled_features)
        
        self.logger.info(f"Hierarchical clustering completed with {optimal_k} clusters")
        return cluster_labels
    
    def _find_optimal_clusters(self, features, max_k=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        if len(features) < max_k:
            max_k = min(len(features) - 1, 5)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            inertias.append(kmeans.inertia_)
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                silhouette_scores.append(silhouette_score(features, labels))
            else:
                silhouette_scores.append(0)
        
        # Choose k with highest silhouette score
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 5  # Default
        
        return optimal_k
    
    def _calculate_user_scores(self, df, engagement_weight, influence_weight):
        """Calculate engagement and influence scores for each user"""
        self.logger.info("Calculating user engagement and influence scores...")
        
        # Aggregate user metrics
        user_metrics = df.groupby('comment_owner_username').agg({
            'comment_likes': ['mean', 'sum'],
            'engagement_frequency': 'first',
            'influence_score': 'first',
            'content_interaction': 'mean',
            'comment_engagement_ratio': 'mean',
            '#Followers': 'first'
        }).reset_index()
        
        # Flatten column names
        user_metrics.columns = [
            'username', 'avg_comment_likes', 'total_comment_likes',
            'engagement_frequency', 'influence_score', 'avg_content_interaction',
            'avg_comment_ratio', 'followers'
        ]
        
        # Calculate engagement score (normalized combination of metrics)
        engagement_features = [
            'avg_comment_likes', 'total_comment_likes', 'engagement_frequency',
            'avg_content_interaction', 'avg_comment_ratio'
        ]
        
        # Normalize engagement features
        engagement_scores = user_metrics[engagement_features].fillna(0)
        engagement_scores = (engagement_scores - engagement_scores.min()) / (
            engagement_scores.max() - engagement_scores.min() + 1e-8
        )
        user_metrics['engagement_score'] = engagement_scores.mean(axis=1)
        
        # Normalize influence score (already normalized in preprocessing)
        user_metrics['influence_score'] = user_metrics['influence_score'].fillna(0)
        
        # Calculate total score
        user_metrics['total_score'] = (
            engagement_weight * user_metrics['engagement_score'] +
            influence_weight * user_metrics['influence_score']
        )
        
        return user_metrics
    
    def _select_top_followers(self, user_scores, top_percent):
        """Select top followers based on combined score"""
        # Calculate the number of followers to select
        n_followers = max(1, int(len(user_scores) * top_percent / 100))
        
        # Sort by total score and select top followers
        top_followers_df = user_scores.nlargest(n_followers, 'total_score')
        
        # Convert to dictionary format
        high_value_followers = {}
        for _, row in top_followers_df.iterrows():
            high_value_followers[row['username']] = {
                'engagement_score': float(row['engagement_score']),
                'influence_score': float(row['influence_score']),
                'total_score': float(row['total_score']),
                'followers': int(row['followers']),
                'cluster': int(row['cluster']),
                'avg_comment_likes': float(row['avg_comment_likes']),
                'engagement_frequency': float(row['engagement_frequency'])
            }
        
        return high_value_followers
    
    def analyze_clusters(self, df, cluster_labels):
        """Analyze cluster characteristics"""
        self.logger.info("Analyzing cluster characteristics...")
        
        # Add cluster labels to dataframe
        user_clusters = df.groupby('comment_owner_username').first().reset_index()
        user_clusters['cluster'] = cluster_labels
        
        # Analyze cluster statistics
        cluster_stats = user_clusters.groupby('cluster').agg({
            'engagement_frequency': ['mean', 'std'],
            'influence_score': ['mean', 'std'],
            '#Followers': ['mean', 'std'],
            'comment_likes': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        cluster_stats.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
        
        return cluster_stats
    
    def save_clustering_results(self, high_value_followers, cluster_stats=None, 
                               output_path="outputs/follower_selection_results.json"):
        """Save clustering and selection results"""
        results = {
            "high_value_followers": high_value_followers,
            "selection_summary": {
                "total_selected": len(high_value_followers),
                "selection_criteria": "Top followers by combined engagement and influence score",
                "clustering_method": "K-Means"
            }
        }
        
        if cluster_stats is not None:
            results["cluster_analysis"] = cluster_stats.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Clustering results saved to {output_path}")
    
    def visualize_clusters(self, features, cluster_labels, user_features):
        """Create cluster visualization (optional)"""
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            # Reduce dimensionality for visualization
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # Create scatter plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
            plt.title('User Clusters (PCA visualization)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            
            # Save plot
            plt.savefig('outputs/user_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Cluster visualization saved to outputs/user_clusters.png")
            
        except ImportError:
            self.logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
