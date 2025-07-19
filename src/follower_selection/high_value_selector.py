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
        Select high-value followers using clustering algorithms (legacy method)
        
        Args:
            df: Preprocessed Instagram dataset
            top_percent: Percentage of top followers to select
            method: Clustering method ('K-Means', 'DBSCAN', 'Hierarchical')
            engagement_weight: Weight for engagement features
            influence_weight: Weight for influence features
            
        Returns:
            Dictionary of high-value followers with their scores
        """
        self.logger.info(f"Selecting top {top_percent}% followers using {method} (legacy mode)")
        
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

    def select_high_value_followers(self, data, owner_id, top_percentage=10, 
                                  clustering_method="K-Means", 
                                  engagement_weight=0.7, influence_weight=0.3):
        """
        Select high-value followers for a specific Instagram account
        
        Args:
            data: Full dataset (DataFrame)
            owner_id: Specific Instagram account owner ID to filter by
            top_percentage: Top X% of followers to select (5-25)
            clustering_method: 'K-Means', 'DBSCAN', or 'Hierarchical'
            engagement_weight: Weight for engagement features (0.0-1.0)
            influence_weight: Weight for influence features (0.0-1.0)
            
        Returns:
            Dictionary of high-value followers for the specified owner_id
        """
        try:
            self.logger.info(f"Selecting top {top_percentage}% followers for owner_id: {owner_id}")
            
            # Validate inputs
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            
            if 'owner_id' not in data.columns:
                raise KeyError("Dataset missing 'owner_id' column. Please check your data preprocessing.")
            
            if 'comment_owner_username' not in data.columns:
                raise KeyError("Dataset missing 'comment_owner_username' column. Please check your data preprocessing.")
            
            # Convert owner_id to match dataset type (handle string/int conversion)
            original_owner_id = owner_id
            try:
                # Try to convert to int if it's a string of digits
                if isinstance(owner_id, str) and owner_id.isdigit():
                    owner_id = int(owner_id)
                elif isinstance(owner_id, (int, float)):
                    owner_id = int(owner_id)
            except (ValueError, TypeError):
                pass  # Keep original owner_id
            
            # Filter dataset by specific owner_id
            account_data = data[data['owner_id'] == owner_id].copy()
            
            if account_data.empty:
                # Try the original owner_id if conversion didn't work
                account_data = data[data['owner_id'] == original_owner_id].copy()
                
            if account_data.empty:
                available_ids = data['owner_id'].unique()
                self.logger.error(f"Data type comparison issue - owner_id: {original_owner_id} (type: {type(original_owner_id)}), converted: {owner_id} (type: {type(owner_id)})")
                self.logger.error(f"Sample dataset owner_ids: {available_ids[:5]} (types: {[type(x) for x in available_ids[:5]]})")
                raise ValueError(f"No data found for owner_id: {original_owner_id} (tried both {original_owner_id} and {owner_id}). Available owner_ids: {available_ids}")
            
            self.logger.info(f"Filtered data: {len(account_data)} records for owner {owner_id}")
            
            # Check if we have enough data for clustering
            unique_users = account_data['comment_owner_username'].nunique()
            if unique_users < 2:
                self.logger.warning(f"Only {unique_users} unique followers found for owner {owner_id}. Returning all followers.")
                # Return all followers if too few for clustering
                return self._handle_insufficient_data(account_data, owner_id)
            
            # Prepare features for clustering (account-specific)
            cluster_features = self._prepare_clustering_features(account_data)
            
            # Apply selected clustering method
            if clustering_method == "K-Means":
                cluster_labels = self._apply_kmeans(cluster_features)
            elif clustering_method == "DBSCAN":
                cluster_labels = self._apply_dbscan(cluster_features)
            elif clustering_method == "Hierarchical":
                cluster_labels = self._apply_hierarchical(cluster_features)
            else:
                raise ValueError(f"Unknown clustering method: {clustering_method}")
            
            # Calculate engagement and influence scores (account-specific)
            user_scores = self._calculate_user_scores(
                account_data, engagement_weight, influence_weight
            )
            
            # Add cluster information
            user_scores['cluster'] = cluster_labels
            user_scores['owner_id'] = owner_id  # Track which account these followers belong to
            
            # Select high-value followers
            high_value_followers = self._select_top_followers(
                user_scores, top_percentage
            )
            
            # Save results with account-specific naming
            self._save_account_specific_results(
                high_value_followers, owner_id, account_data, top_percentage, 
                clustering_method, engagement_weight, influence_weight
            )
            
            self.logger.info(f"Selected {len(high_value_followers)} high-value followers for {owner_id}")
            return high_value_followers
            
        except Exception as e:
            self.logger.error(f"Error in follower selection for owner {owner_id}: {str(e)}")
            raise
    
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
    
    def _handle_insufficient_data(self, account_data, owner_id):
        """Handle cases where there's insufficient data for clustering"""
        self.logger.info(f"Handling insufficient data for owner {owner_id}")
        
        # Calculate basic scores for all users
        user_metrics = account_data.groupby('comment_owner_username').agg({
            'comment_likes': ['mean', 'sum'],
            'engagement_frequency': 'first',
            'influence_score': 'first',
            '#Followers': 'first'
        }).reset_index()
        
        # Flatten column names
        user_metrics.columns = [
            'username', 'avg_comment_likes', 'total_comment_likes',
            'engagement_frequency', 'influence_score', 'followers'
        ]
        
        # Fill missing values
        user_metrics = user_metrics.fillna(0)
        
        # Create simple scoring
        high_value_followers = {}
        for _, row in user_metrics.iterrows():
            high_value_followers[row['username']] = {
                'engagement_score': float(row['avg_comment_likes']) / 100.0,  # Normalize
                'influence_score': float(row['influence_score']),
                'total_score': float(row['avg_comment_likes']) / 100.0 + float(row['influence_score']),
                'followers': int(row['followers']),
                'cluster': 0,  # Single cluster
                'avg_comment_likes': float(row['avg_comment_likes']),
                'engagement_frequency': float(row['engagement_frequency'])
            }
        
        return high_value_followers

    def _save_account_specific_results(self, high_value_followers, owner_id, account_data,
                                     top_percentage, clustering_method, 
                                     engagement_weight, influence_weight):
        """Save results with account-specific metadata"""
        # Get username for this owner_id from the account data
        username = "unknown_user"
        try:
            if 'username' in account_data.columns and len(account_data) > 0:
                username = account_data['username'].iloc[0]
            else:
                username = f"user_{owner_id}"
        except:
            username = f"user_{owner_id}"
            
        results = {
            "owner_id": str(owner_id),
            "username": username,
            "selection_metadata": {
                "top_percentage": top_percentage,
                "clustering_method": clustering_method,
                "engagement_weight": engagement_weight,
                "influence_weight": influence_weight,
                "total_selected": len(high_value_followers),
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "high_value_followers": high_value_followers
        }
        
        # Save account-specific file
        account_output_path = f"outputs/high_value_followers_{owner_id}.json"
        with open(account_output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save to general location for backward compatibility
        with open("outputs/high_value_followers.json", 'w') as f:
            json.dump(high_value_followers, f, indent=2)
        
        self.logger.info(f"Results saved to {account_output_path} and outputs/high_value_followers.json")

    def get_available_accounts(self, data):
        """Get list of available owner_ids in the dataset with usernames"""
        if 'owner_id' not in data.columns:
            raise KeyError("Dataset missing 'owner_id' column")
        
        if 'username' not in data.columns:
            raise KeyError("Dataset missing 'username' column")
        
        accounts = data['owner_id'].unique()
        account_stats = []
        
        for account in accounts:
            account_data = data[data['owner_id'] == account]
            followers_count = account_data['comment_owner_username'].nunique()
            interactions_count = len(account_data)
            
            # Get the username for this account (should be consistent for same owner_id)
            username = account_data['username'].iloc[0] if len(account_data) > 0 else f"user_{account}"
            
            account_stats.append({
                'owner_id': account,
                'username': username,
                'unique_followers': followers_count,
                'total_interactions': interactions_count
            })
        
        # Sort by followers count for better UX
        account_stats.sort(key=lambda x: x['unique_followers'], reverse=True)
        return account_stats
