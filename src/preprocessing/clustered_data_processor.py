import os
import json
import pandas as pd

class ClusteredDataProcessor:
    def __init__(self, base_path='data/clustered_data'):
        self.base_path = base_path
        self.user_stats = self._load_user_stats()

    def _load_user_stats(self):
        """
        Load user statistics from influencers.csv
        """
        try:
            import pandas as pd
            influencers_path = os.path.join(self.base_path, 'influencers.csv')
            if os.path.exists(influencers_path):
                # Read the CSV, skipping the separator line and handling BOM
                df = pd.read_csv(influencers_path, skiprows=[1], encoding='utf-8-sig')
                # Create a dictionary for quick lookup
                user_stats = {}
                for _, row in df.iterrows():
                    username = row['Username']
                    user_stats[username] = {
                        'followers': int(row['#Followers']) if pd.notna(row['#Followers']) else 0,
                        'followees': int(row['#Followees']) if pd.notna(row['#Followees']) else 0,
                        'posts': int(row['#Posts']) if pd.notna(row['#Posts']) else 0,
                        'category': str(row['Category']) if pd.notna(row['Category']) else 'other'
                    }
                print(f"Loaded user stats for {len(user_stats)} users from influencers.csv")
                return user_stats
            else:
                print(f"Warning: influencers.csv not found at {influencers_path}")
                return {}
        except Exception as e:
            print(f"Warning: Could not load user stats: {e}")
            return {}

    def get_cluster_names(self):
        """
        Scans the base path for cluster directories.
        """
        try:
            return [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        except FileNotFoundError:
            return []

    def process_cluster(self, cluster_name):
        """
        Processes a single cluster, reads all .info files, and returns a pandas DataFrame.
        """
        cluster_path = os.path.join(self.base_path, cluster_name)
        all_post_data = []

        user_folders = [d for d in os.listdir(cluster_path) if os.path.isdir(os.path.join(cluster_path, d))]

        for user_folder in user_folders:
            user_path = os.path.join(cluster_path, user_folder)
            
            # Get all .info files in the user folder
            info_files = [f for f in os.listdir(user_path) if f.endswith('.info')]
            
            for info_file in info_files:
                info_file_path = os.path.join(user_path, info_file)
                
                try:
                    with open(info_file_path, 'r', encoding='utf-8') as f:
                        post_data = json.load(f)
                        
                        # Extract owner information
                        owner = post_data.get('owner', {})
                        username = owner.get('username', user_folder)
                        
                        # Get user statistics from influencers.csv
                        user_stats = self.user_stats.get(username, {})
                        followers_count = user_stats.get('followers', self._estimate_followers_from_cluster(cluster_name))
                        followees_count = user_stats.get('followees', 0)
                        posts_count = user_stats.get('posts', 0)
                        user_category = user_stats.get('category', '')
                        
                        # Extract caption text
                        caption_edges = post_data.get('edge_media_to_caption', {}).get('edges', [])
                        caption = caption_edges[0]['node']['text'] if caption_edges else ""
                        
                        # Extract engagement metrics
                        likes_count = post_data.get('edge_media_preview_like', {}).get('count', 0)
                        comments_count = post_data.get('edge_media_to_parent_comment', {}).get('count', 0)
                        
                        # Extract post metadata
                        shortcode = post_data.get('shortcode', '')
                        post_id = post_data.get('id', '')
                        timestamp = post_data.get('taken_at_timestamp', 0)
                        is_video = post_data.get('is_video', False)
                        
                        # Extract location if available
                        location = post_data.get('location', {})
                        location_name = location.get('name', '') if location else ''
                        
                        # Extract hashtags from caption
                        hashtags = []
                        if caption:
                            words = caption.split()
                            hashtags = [word for word in words if word.startswith('#')]
                        
                        # Extract mentions from caption
                        mentions = []
                        if caption:
                            words = caption.split()
                            mentions = [word for word in words if word.startswith('@')]
                        
                        # Use user category from CSV if available, otherwise classify content
                        content_category = user_category if user_category else self._classify_content(caption, hashtags)
                        
                        # Create row for this post
                        row_data = {
                            'post_id': post_id,
                            'owner_id': owner.get('id', ''),
                            'timestamp': timestamp,
                            'likes': likes_count,
                            'comments_count': comments_count,
                            'caption': caption,
                            'hashtags': ', '.join(hashtags),
                            'location_id': location.get('id', '') if location else '',
                            'media_type': 'video' if is_video else 'image',
                            'username': username,
                            'shortcode': shortcode,
                            'location': location_name,
                            'is_private': owner.get('is_private', False),
                            'is_verified': owner.get('is_verified', False),
                            'mentions': ', '.join(mentions),
                            'Category': content_category,
                            '#Followers': followers_count,
                            '#Followees': followees_count,
                            '#Posts': posts_count
                        }
                        
                        # Extract comments data
                        comment_edges = post_data.get('edge_media_to_parent_comment', {}).get('edges', [])
                        if comment_edges:
                            for comment_edge in comment_edges:
                                comment_node = comment_edge.get('node', {})
                                comment_owner = comment_node.get('owner', {})
                                
                                comment_row = row_data.copy()
                                comment_row.update({
                                    'comment_text': comment_node.get('text', ''),
                                    'comment_owner_username': comment_owner.get('username', ''),
                                    'comment_likes': comment_node.get('edge_liked_by', {}).get('count', 0),
                                    'comment_timestamp': comment_node.get('created_at', 0)
                                })
                                all_post_data.append(comment_row)
                        else:
                            # If no comments, still add the post data
                            row_data.update({
                                'comment_text': '',
                                'comment_owner_username': '',
                                'comment_likes': 0,
                                'comment_timestamp': 0
                            })
                            all_post_data.append(row_data)
                            
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {info_file_path}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing {info_file_path}: {e}")
                    continue
        
        return pd.DataFrame(all_post_data)

    def _estimate_followers_from_cluster(self, cluster_name):
        """
        Estimate follower count based on cluster name if not available in influencers.csv
        """
        cluster_mapping = {
            'followers_1000_to_2500': 1750,  # Average of range
            'followers_2500_to_5000': 3750,
            'followers_5000_to_7500': 6250,
            'followers_7500_to_10000': 8750,
            'followers_10000_to_50000': 30000,
            'followers_greater_than_50000': 75000
        }
        return cluster_mapping.get(cluster_name, 0)

    def generate_csv(self, dataframe, output_path):
        """
        Saves the DataFrame to a CSV file.
        """
        dataframe.to_csv(output_path, index=False)

    def _classify_content(self, caption, hashtags):
        """
        Classify content based on caption and hashtags
        """
        text = (caption + ' ' + ' '.join(hashtags)).lower()
        
        # Define category keywords
        categories = {
            'fashion': ['fashion', 'style', 'outfit', 'ootd', 'clothing', 'dress', 'shoes', 'bag', 'accessories'],
            'beauty': ['beauty', 'makeup', 'skincare', 'cosmetics', 'lipstick', 'foundation', 'hair', 'nails'],
            'fitness': ['fitness', 'gym', 'workout', 'exercise', 'health', 'training', 'muscle', 'cardio'],
            'food': ['food', 'recipe', 'cooking', 'restaurant', 'meal', 'dinner', 'lunch', 'breakfast', 'chef'],
            'travel': ['travel', 'vacation', 'trip', 'explore', 'adventure', 'wanderlust', 'journey', 'tourism'],
            'family': ['family', 'kids', 'children', 'baby', 'mom', 'dad', 'parent', 'motherhood', 'fatherhood'],
            'pet': ['pet', 'dog', 'cat', 'puppy', 'kitten', 'animal', 'pets', 'dogs', 'cats']
        }
        
        # Count keyword matches for each category
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        # Return the category with highest score, or 'other' if no matches
        if max(category_scores.values()) > 0:
            return max(category_scores, key=category_scores.get)
        else:
            return 'other'

    def create_compatible_dataframe(self, dataframe):
        """
        Transform the clustered dataframe to match the expected format for the existing pipeline
        """
        if dataframe.empty:
            return dataframe
            
        # Create a copy to avoid modifying the original
        compatible_df = dataframe.copy()
        
        # Add missing columns with default values
        required_columns = {
            'caption_length': 0,
            'num_hashtags': 0,
            'has_mention': False,
            'has_url': False,
            'follower_adjusted_likes': 0.0,
            'follower_adjusted_comments': 0.0,
            'engagement_rate': 0.0,
            'hashtags_agg': '',
            'engagement_frequency': 0.0,
            'influence_score': 0.0,
            'content_interaction': 0.0,
            'comment_engagement_ratio': 0.0,
            'comment_length': 0,
            'has_emoji': False,
            'category_beauty': False,
            'category_family': False,
            'category_fashion': False,
            'category_fitness': False,
            'category_food': False,
            'category_pet': False,
            'category_travel': False,
            'user_id_encoded': 0,
            'engagement_binary': 0,
            'engagement_probability': 0.0
        }
        
        for col, default_value in required_columns.items():
            if col not in compatible_df.columns:
                compatible_df[col] = default_value
        
        # Calculate basic features
        if 'caption' in compatible_df.columns:
            compatible_df['caption_length'] = compatible_df['caption'].str.len().fillna(0)
            compatible_df['has_mention'] = compatible_df['caption'].str.contains('@', na=False)
            compatible_df['has_url'] = compatible_df['caption'].str.contains('http', na=False)
        
        if 'hashtags' in compatible_df.columns:
            compatible_df['num_hashtags'] = compatible_df['hashtags'].str.count(',').fillna(0) + 1
            compatible_df['hashtags_agg'] = compatible_df['hashtags']
        
        if 'comment_text' in compatible_df.columns:
            compatible_df['comment_length'] = compatible_df['comment_text'].str.len().fillna(0)
            compatible_df['has_emoji'] = compatible_df['comment_text'].str.contains(r'[^\w\s]', regex=True, na=False)
        
        # Set category binary flags
        if 'Category' in compatible_df.columns:
            for category in ['beauty', 'family', 'fashion', 'fitness', 'food', 'pet', 'travel']:
                compatible_df[f'category_{category}'] = (compatible_df['Category'] == category)
        
        return compatible_df
