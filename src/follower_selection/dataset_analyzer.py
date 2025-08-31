"""
Dataset-Wide High-Value Follower Analyzer
Analyzes the entire dataset to identify high-value followers across all accounts
and provides network visualizations and insights
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
from .high_value_selector import HighValueFollowerSelector


class DatasetFollowerAnalyzer:
    def __init__(self):
        self.follower_selector = HighValueFollowerSelector()
        self.logger = self._setup_logger()
        self.analysis_results = {}
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def analyze_entire_dataset(self, data, top_percentage=10, clustering_method="K-Means", 
                             engagement_weight=0.7, influence_weight=0.3, 
                             min_followers_per_account=3):
        """
        Analyze the entire dataset to find high-value followers for all accounts
        
        Args:
            data: Full dataset (DataFrame)
            top_percentage: Top X% of followers to select per account
            clustering_method: Clustering method to use
            engagement_weight: Weight for engagement features
            influence_weight: Weight for influence features
            min_followers_per_account: Minimum followers required for analysis
            
        Returns:
            Comprehensive analysis results dictionary
        """
        self.logger.info("🔍 Starting dataset-wide high-value follower analysis...")
        
        try:
            # Get all available accounts
            accounts = self.follower_selector.get_available_accounts(data)
            self.logger.info(f"📊 Analyzing {len(accounts)} accounts in dataset")
            
            # Filter accounts with sufficient followers
            valid_accounts = [acc for acc in accounts 
                            if acc['unique_followers'] >= min_followers_per_account]
            
            self.logger.info(f"📈 {len(valid_accounts)} accounts meet minimum follower criteria")
            
            # Analyze each account
            account_results = {}
            all_high_value_followers = {}
            
            for i, account in enumerate(valid_accounts):
                owner_id = account['owner_id']
                username = account['username']
                
                self.logger.info(f"🎯 Analyzing account {i+1}/{len(valid_accounts)}: @{username}")
                
                try:
                    # Get high-value followers for this account
                    high_value_followers = self.follower_selector.select_high_value_followers(
                        data=data,
                        owner_id=owner_id,
                        top_percentage=top_percentage,
                        clustering_method=clustering_method,
                        engagement_weight=engagement_weight,
                        influence_weight=influence_weight
                    )
                    
                    account_results[username] = {
                        'owner_id': owner_id,
                        'username': username,
                        'followers_count': account['unique_followers'],
                        'interactions_count': account['total_interactions'],
                        'high_value_followers': high_value_followers,
                        'high_value_count': len(high_value_followers)
                    }
                    
                    # Add to global high-value followers tracking
                    for follower, data_point in high_value_followers.items():
                        if follower not in all_high_value_followers:
                            all_high_value_followers[follower] = []
                        all_high_value_followers[follower].append({
                            'account_username': username,
                            'account_owner_id': owner_id,
                            'total_score': data_point['total_score'],
                            'engagement_score': data_point['engagement_score'],
                            'influence_score': data_point['influence_score']
                        })
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to analyze @{username}: {str(e)}")
                    continue
            
            # Perform cross-account analysis
            cross_analysis = self._analyze_cross_account_patterns(account_results, all_high_value_followers)
            
            # Generate insights
            insights = self._generate_insights(account_results, all_high_value_followers, cross_analysis)
            
            # Compile final results
            self.analysis_results = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_accounts_analyzed': len(account_results),
                    'total_accounts_in_dataset': len(accounts),
                    'analysis_parameters': {
                        'top_percentage': top_percentage,
                        'clustering_method': clustering_method,
                        'engagement_weight': engagement_weight,
                        'influence_weight': influence_weight,
                        'min_followers_per_account': min_followers_per_account
                    }
                },
                'account_results': account_results,
                'cross_account_analysis': cross_analysis,
                'insights': insights,
                'high_value_followers_global': all_high_value_followers
            }
            
            # Save results
            self._save_analysis_results()
            
            self.logger.info("✅ Dataset-wide analysis completed successfully!")
            return self.analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in dataset analysis: {str(e)}")
            raise
    
    def _analyze_cross_account_patterns(self, account_results, all_high_value_followers):
        """Analyze patterns across multiple accounts"""
        self.logger.info("🔗 Analyzing cross-account patterns...")
        
        # Find followers who are high-value for multiple accounts
        multi_account_followers = {}
        for follower, accounts in all_high_value_followers.items():
            if len(accounts) > 1:
                multi_account_followers[follower] = {
                    'accounts': accounts,
                    'account_count': len(accounts),
                    'avg_total_score': np.mean([acc['total_score'] for acc in accounts]),
                    'avg_engagement_score': np.mean([acc['engagement_score'] for acc in accounts]),
                    'avg_influence_score': np.mean([acc['influence_score'] for acc in accounts])
                }
        
        # Account similarity based on shared high-value followers
        account_similarity = {}
        account_usernames = list(account_results.keys())
        
        for i, account1 in enumerate(account_usernames):
            account_similarity[account1] = {}
            followers1 = set(account_results[account1]['high_value_followers'].keys())
            
            for j, account2 in enumerate(account_usernames):
                if i != j:
                    followers2 = set(account_results[account2]['high_value_followers'].keys())
                    
                    # Calculate Jaccard similarity
                    intersection = len(followers1.intersection(followers2))
                    union = len(followers1.union(followers2))
                    similarity = intersection / union if union > 0 else 0
                    
                    account_similarity[account1][account2] = {
                        'similarity_score': similarity,
                        'shared_followers': intersection,
                        'total_unique_followers': union
                    }
        
        # Top performers analysis
        top_performers = self._identify_top_performers(account_results)
        
        return {
            'multi_account_followers': multi_account_followers,
            'account_similarity': account_similarity,
            'top_performers': top_performers,
            'network_stats': {
                'total_unique_high_value_followers': len(all_high_value_followers),
                'multi_account_high_value_followers': len(multi_account_followers),
                'cross_pollination_rate': len(multi_account_followers) / len(all_high_value_followers) if all_high_value_followers else 0
            }
        }
    
    def _identify_top_performers(self, account_results):
        """Identify top performing accounts and followers"""
        
        # Top accounts by high-value follower count
        top_accounts_by_count = sorted(
            account_results.items(),
            key=lambda x: x[1]['high_value_count'],
            reverse=True
        )[:10]
        
        # Top accounts by high-value follower ratio
        top_accounts_by_ratio = sorted(
            account_results.items(),
            key=lambda x: x[1]['high_value_count'] / x[1]['followers_count'] if x[1]['followers_count'] > 0 else 0,
            reverse=True
        )[:10]
        
        # Most valuable followers (highest average scores across accounts)
        all_followers_scores = defaultdict(list)
        for account_data in account_results.values():
            for follower, data in account_data['high_value_followers'].items():
                all_followers_scores[follower].append(data['total_score'])
        
        top_followers = sorted(
            [(follower, np.mean(scores)) for follower, scores in all_followers_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:20]
        
        return {
            'top_accounts_by_count': [(name, data['high_value_count']) for name, data in top_accounts_by_count],
            'top_accounts_by_ratio': [(name, data['high_value_count'] / data['followers_count']) for name, data in top_accounts_by_ratio],
            'top_followers_by_score': top_followers
        }
    
    def _generate_insights(self, account_results, all_high_value_followers, cross_analysis):
        """Generate actionable insights from the analysis"""
        
        insights = {
            'summary_statistics': {
                'total_accounts': len(account_results),
                'total_high_value_followers': len(all_high_value_followers),
                'avg_high_value_per_account': np.mean([data['high_value_count'] for data in account_results.values()]),
                'max_high_value_per_account': max([data['high_value_count'] for data in account_results.values()]) if account_results else 0,
                'min_high_value_per_account': min([data['high_value_count'] for data in account_results.values()]) if account_results else 0
            },
            'patterns': {
                'high_cross_pollination': len(cross_analysis['multi_account_followers']) > len(all_high_value_followers) * 0.1,
                'account_clustering_detected': self._detect_account_clusters(cross_analysis['account_similarity']),
                'power_followers_identified': len([f for f, data in cross_analysis['multi_account_followers'].items() if data['account_count'] >= 3])
            },
            'recommendations': self._generate_recommendations(account_results, cross_analysis)
        }
        
        return insights
    
    def _detect_account_clusters(self, similarity_matrix):
        """Detect clusters of similar accounts based on shared followers"""
        # Simple clustering based on high similarity scores
        clusters = []
        processed_accounts = set()
        
        for account1, similarities in similarity_matrix.items():
            if account1 in processed_accounts:
                continue
                
            cluster = [account1]
            for account2, sim_data in similarities.items():
                if sim_data['similarity_score'] > 0.3:  # Threshold for similarity
                    cluster.append(account2)
                    processed_accounts.add(account2)
            
            if len(cluster) > 1:
                clusters.append(cluster)
                processed_accounts.add(account1)
        
        return len(clusters) > 0
    
    def _generate_recommendations(self, account_results, cross_analysis):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Power followers recommendation
        power_followers = [f for f, data in cross_analysis['multi_account_followers'].items() 
                          if data['account_count'] >= 3]
        if power_followers:
            recommendations.append({
                'type': 'power_followers',
                'title': 'Leverage Power Followers',
                'description': f"Found {len(power_followers)} followers who are high-value across multiple accounts. These are key influencers.",
                'action': 'Focus on maintaining relationships with these cross-account valuable followers.'
            })
        
        # Account collaboration recommendation
        high_similarity_pairs = []
        for acc1, similarities in cross_analysis['account_similarity'].items():
            for acc2, sim_data in similarities.items():
                if sim_data['similarity_score'] > 0.4:
                    high_similarity_pairs.append((acc1, acc2, sim_data['similarity_score']))
        
        if high_similarity_pairs:
            recommendations.append({
                'type': 'collaboration',
                'title': 'Account Collaboration Opportunities',
                'description': f"Found {len(high_similarity_pairs)} account pairs with high follower overlap (>40% similarity).",
                'action': 'Consider collaboration or cross-promotion strategies between similar accounts.'
            })
        
        # Growth opportunities
        low_performing_accounts = [name for name, data in account_results.items() 
                                 if data['high_value_count'] < 5]
        if low_performing_accounts:
            recommendations.append({
                'type': 'growth',
                'title': 'Growth Opportunities',
                'description': f"{len(low_performing_accounts)} accounts have fewer than 5 high-value followers.",
                'action': 'Focus on engagement strategies and content optimization for these accounts.'
            })
        
        return recommendations
    
    def _save_analysis_results(self):
        """Save comprehensive analysis results"""
        output_path = "outputs/dataset_follower_analysis.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the results
        import json
        
        def deep_convert(item):
            if isinstance(item, dict):
                return {key: deep_convert(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [deep_convert(element) for element in item]
            else:
                return convert_numpy(item)
        
        serializable_results = deep_convert(self.analysis_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"📁 Analysis results saved to {output_path}")
    
    def get_analysis_summary(self):
        """Get a summary of the analysis results"""
        if not self.analysis_results:
            return None
        
        metadata = self.analysis_results['metadata']
        insights = self.analysis_results['insights']
        cross_analysis = self.analysis_results['cross_account_analysis']
        
        return {
            'accounts_analyzed': metadata['total_accounts_analyzed'],
            'total_high_value_followers': insights['summary_statistics']['total_high_value_followers'],
            'avg_high_value_per_account': round(insights['summary_statistics']['avg_high_value_per_account'], 1),
            'power_followers_count': len(cross_analysis['multi_account_followers']),
            'cross_pollination_rate': round(cross_analysis['network_stats']['cross_pollination_rate'] * 100, 1),
            'recommendations_count': len(insights['recommendations'])
        }
