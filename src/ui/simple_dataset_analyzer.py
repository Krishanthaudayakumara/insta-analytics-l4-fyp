"""
Simple Dataset Analyzer - Fallback for when full modules aren't available
Provides basic dataset-wide analysis functionality without complex dependencies
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from collections import defaultdict, Counter


class SimpleDatasetAnalyzer:
    """Simple dataset analyzer for fallback scenarios"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def analyze_entire_dataset(self, data, top_percentage=10, clustering_method="K-Means", 
                             engagement_weight=0.7, influence_weight=0.3, 
                             min_followers_per_account=3):
        """
        Simple analysis of the entire dataset without complex dependencies
        
        Args:
            data: Full dataset (DataFrame)
            top_percentage: Top X% of followers to select per account
            clustering_method: Clustering method (for compatibility)
            engagement_weight: Weight for engagement features
            influence_weight: Weight for influence features
            min_followers_per_account: Minimum followers required for analysis
            
        Returns:
            Simplified analysis results dictionary
        """
        self.logger.info("🔍 Starting simple dataset-wide analysis...")
        
        try:
            # Get available accounts
            accounts = self._get_available_accounts(data)
            self.logger.info(f"📊 Analyzing {len(accounts)} accounts in dataset")
            
            # Filter accounts with sufficient followers
            valid_accounts = [acc for acc in accounts 
                            if acc['unique_followers'] >= min_followers_per_account]
            
            self.logger.info(f"📈 {len(valid_accounts)} accounts meet minimum follower criteria")
            
            # Analyze each account with simple scoring
            account_results = {}
            all_high_value_followers = {}
            
            for i, account in enumerate(valid_accounts):
                owner_id = account['owner_id']
                username = account['username']
                
                self.logger.info(f"🎯 Analyzing account {i+1}/{len(valid_accounts)}: @{username}")
                
                try:
                    # Get high-value followers for this account using simple method
                    high_value_followers = self._simple_follower_selection(
                        data=data,
                        owner_id=owner_id,
                        top_percentage=top_percentage,
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
            
            # Perform simple cross-account analysis
            cross_analysis = self._simple_cross_account_analysis(account_results, all_high_value_followers)
            
            # Generate simple insights
            insights = self._simple_insights(account_results, all_high_value_followers, cross_analysis)
            
            # Compile final results
            analysis_results = {
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_accounts_analyzed': len(account_results),
                    'total_accounts_in_dataset': len(accounts),
                    'analysis_parameters': {
                        'top_percentage': top_percentage,
                        'clustering_method': clustering_method + " (Simple Mode)",
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
            self._save_analysis_results(analysis_results)
            
            self.logger.info("✅ Simple dataset-wide analysis completed successfully!")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in simple dataset analysis: {str(e)}")
            raise
    
    def _get_available_accounts(self, data):
        """Get list of available accounts with basic stats"""
        accounts = []
        
        for owner_id in data['owner_id'].unique():
            account_data = data[data['owner_id'] == owner_id]
            
            if len(account_data) > 0:
                # Get username
                username = account_data['username'].iloc[0] if 'username' in account_data.columns else f"user_{owner_id}"
                
                # Count unique followers
                unique_followers = account_data['comment_owner_username'].nunique()
                total_interactions = len(account_data)
                
                accounts.append({
                    'owner_id': owner_id,
                    'username': username,
                    'unique_followers': unique_followers,
                    'total_interactions': total_interactions
                })
        
        return accounts
    
    def _simple_follower_selection(self, data, owner_id, top_percentage, engagement_weight, influence_weight):
        """Simple follower selection without clustering"""
        
        # Filter data for this account
        account_data = data[data['owner_id'] == owner_id].copy()
        
        if len(account_data) == 0:
            return {}
        
        # Group by follower
        follower_stats = account_data.groupby('comment_owner_username').agg({
            'comment_likes': ['sum', 'mean', 'count'],
            'likes': 'mean'
        }).round(3)
        
        # Flatten column names
        follower_stats.columns = ['total_comment_likes', 'avg_comment_likes', 'comment_count', 'avg_post_likes']
        
        # Simple scoring without complex features
        # Engagement score: based on comment activity
        max_comments = follower_stats['comment_count'].max() if len(follower_stats) > 0 else 1
        max_comment_likes = follower_stats['total_comment_likes'].max() if len(follower_stats) > 0 else 1
        
        follower_stats['engagement_score'] = (
            (follower_stats['comment_count'] / max_comments) * 0.6 +
            (follower_stats['total_comment_likes'] / max_comment_likes) * 0.4
        ).fillna(0)
        
        # Influence score: based on average metrics
        max_avg_likes = follower_stats['avg_comment_likes'].max() if len(follower_stats) > 0 else 1
        
        follower_stats['influence_score'] = (
            follower_stats['avg_comment_likes'] / max_avg_likes
        ).fillna(0)
        
        # Total score
        follower_stats['total_score'] = (
            follower_stats['engagement_score'] * engagement_weight +
            follower_stats['influence_score'] * influence_weight
        ).fillna(0)
        
        # Select top percentage
        num_to_select = max(1, int(len(follower_stats) * top_percentage / 100))
        top_followers = follower_stats.nlargest(num_to_select, 'total_score')
        
        # Convert to dict format
        high_value_followers = {}
        for follower, row in top_followers.iterrows():
            high_value_followers[follower] = {
                'total_score': float(row['total_score']),
                'engagement_score': float(row['engagement_score']),
                'influence_score': float(row['influence_score']),
                'comment_count': int(row['comment_count']),
                'total_comment_likes': int(row['total_comment_likes']),
                'avg_comment_likes': float(row['avg_comment_likes'])
            }
        
        return high_value_followers
    
    def _simple_cross_account_analysis(self, account_results, all_high_value_followers):
        """Simple cross-account pattern analysis"""
        
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
        
        # Simple account similarity (shared followers)
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
        
        # Top performers
        top_performers = self._simple_top_performers(account_results)
        
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
    
    def _simple_top_performers(self, account_results):
        """Simple top performers identification"""
        
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
        
        # Most valuable followers
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
    
    def _simple_insights(self, account_results, all_high_value_followers, cross_analysis):
        """Generate simple insights"""
        
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
                'account_clustering_detected': len([s for similarities in cross_analysis['account_similarity'].values() 
                                                   for s in similarities.values() if s['similarity_score'] > 0.3]) > 0,
                'power_followers_identified': len([f for f, data in cross_analysis['multi_account_followers'].items() if data['account_count'] >= 3])
            },
            'recommendations': self._simple_recommendations(account_results, cross_analysis)
        }
        
        return insights
    
    def _simple_recommendations(self, account_results, cross_analysis):
        """Generate simple recommendations"""
        recommendations = []
        
        # Power followers recommendation
        power_followers = [f for f, data in cross_analysis['multi_account_followers'].items() 
                          if data['account_count'] >= 3]
        if power_followers:
            recommendations.append({
                'type': 'power_followers',
                'title': 'Leverage Power Followers (Simple Analysis)',
                'description': f"Found {len(power_followers)} followers who are high-value across multiple accounts.",
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
                'title': 'Account Collaboration Opportunities (Simple Analysis)',
                'description': f"Found {len(high_similarity_pairs)} account pairs with high follower overlap.",
                'action': 'Consider collaboration strategies between similar accounts.'
            })
        
        # Growth opportunities
        low_performing_accounts = [name for name, data in account_results.items() 
                                 if data['high_value_count'] < 5]
        if low_performing_accounts:
            recommendations.append({
                'type': 'growth',
                'title': 'Growth Opportunities (Simple Analysis)',
                'description': f"{len(low_performing_accounts)} accounts have fewer than 5 high-value followers.",
                'action': 'Focus on engagement strategies for these accounts.'
            })
        
        return recommendations
    
    def _save_analysis_results(self, analysis_results):
        """Save simple analysis results"""
        output_path = "outputs/simple_dataset_analysis.json"
        
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the results
        def deep_convert(item):
            if isinstance(item, dict):
                return {key: deep_convert(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [deep_convert(element) for element in item]
            else:
                return convert_numpy(item)
        
        serializable_results = deep_convert(analysis_results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"📁 Simple analysis results saved to {output_path}")


class SimpleDatasetAnalyzer:
    """Simple fallback analyzer for dataset-wide analysis"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def analyze_entire_dataset(self, data, top_percentage=10, clustering_method="K-Means", 
                             engagement_weight=0.7, influence_weight=0.3, 
                             min_followers_per_account=3):
        """
        Simple dataset-wide analysis without complex dependencies
        """
        self.logger.info("🔍 Starting simple dataset analysis...")
        
        try:
            # Import the high value selector (which should work)
            import sys
            import os
            
            # Add src to path if not already there
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(current_dir)
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            
            from follower_selection.high_value_selector import HighValueFollowerSelector
            
            selector = HighValueFollowerSelector()
            accounts = selector.get_available_accounts(data)
            
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
                    high_value_followers = selector.select_high_value_followers(
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
            
            # Simple cross-account analysis
            cross_analysis = self._simple_cross_analysis(account_results, all_high_value_followers)
            
            # Generate simple insights
            insights = self._simple_insights(account_results, all_high_value_followers, cross_analysis)
            
            # Compile results
            results = {
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
                    },
                    'analysis_mode': 'simple'
                },
                'account_results': account_results,
                'cross_account_analysis': cross_analysis,
                'insights': insights,
                'high_value_followers_global': all_high_value_followers
            }
            
            # Save results
            self._save_results(results)
            
            self.logger.info("✅ Simple dataset analysis completed successfully!")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in simple dataset analysis: {str(e)}")
            raise
    
    def _simple_cross_analysis(self, account_results, all_high_value_followers):
        """Simple cross-account pattern analysis"""
        
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
        
        # Simple similarity calculation
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
        
        # Top performers
        top_performers = self._simple_top_performers(account_results)
        
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
    
    def _simple_top_performers(self, account_results):
        """Identify top performers"""
        
        # Top accounts by count
        top_accounts_by_count = sorted(
            account_results.items(),
            key=lambda x: x[1]['high_value_count'],
            reverse=True
        )[:10]
        
        # Top accounts by ratio
        top_accounts_by_ratio = sorted(
            account_results.items(),
            key=lambda x: x[1]['high_value_count'] / x[1]['followers_count'] if x[1]['followers_count'] > 0 else 0,
            reverse=True
        )[:10]
        
        # Top followers by score
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
    
    def _simple_insights(self, account_results, all_high_value_followers, cross_analysis):
        """Generate simple insights"""
        
        insights = {
            'summary_statistics': {
                'total_accounts': len(account_results),
                'total_high_value_followers': len(all_high_value_followers),
                'avg_high_value_per_account': np.mean([data['high_value_count'] for data in account_results.values()]) if account_results else 0,
                'max_high_value_per_account': max([data['high_value_count'] for data in account_results.values()]) if account_results else 0,
                'min_high_value_per_account': min([data['high_value_count'] for data in account_results.values()]) if account_results else 0
            },
            'patterns': {
                'high_cross_pollination': len(cross_analysis['multi_account_followers']) > len(all_high_value_followers) * 0.1,
                'power_followers_identified': len([f for f, data in cross_analysis['multi_account_followers'].items() if data['account_count'] >= 3])
            },
            'recommendations': self._simple_recommendations(account_results, cross_analysis)
        }
        
        return insights
    
    def _simple_recommendations(self, account_results, cross_analysis):
        """Generate simple recommendations"""
        recommendations = []
        
        # Power followers recommendation
        power_followers = [f for f, data in cross_analysis['multi_account_followers'].items() 
                          if data['account_count'] >= 3]
        if power_followers:
            recommendations.append({
                'type': 'power_followers',
                'title': 'Leverage Power Followers',
                'description': f"Found {len(power_followers)} followers who are high-value across multiple accounts.",
                'action': 'Focus on maintaining relationships with these cross-account valuable followers.'
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
    
    def _save_results(self, results):
        """Save results to file"""
        output_path = "outputs/dataset_follower_analysis.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        def deep_convert(item):
            if isinstance(item, dict):
                return {key: deep_convert(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [deep_convert(element) for element in item]
            else:
                return convert_numpy(item)
        
        serializable_results = deep_convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"📁 Simple analysis results saved to {output_path}")