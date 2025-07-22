"""
Utility functions for managing high-value follower data files.
This module handles both individual account-specific files and consolidated data.
"""

import json
import os
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def get_available_owner_ids() -> List[str]:
    """Get list of available owner IDs from high-value follower files."""
    output_dir = "outputs"
    owner_ids = []
    
    if not os.path.exists(output_dir):
        return owner_ids
    
    for filename in os.listdir(output_dir):
        if filename.startswith("high_value_followers_") and filename.endswith(".json"):
            # Extract owner_id from filename
            owner_id = filename.replace("high_value_followers_", "").replace(".json", "")
            if owner_id.isdigit():
                owner_ids.append(owner_id)
    
    return sorted(owner_ids)

def load_high_value_followers(owner_id: Optional[Union[str, int]] = None) -> Dict:
    """
    Load high-value followers data.
    
    Args:
        owner_id: Specific owner ID to load. If None, loads all available data.
        
    Returns:
        Dictionary with high-value followers data
    """
    output_dir = "outputs"
    
    if owner_id is not None:
        # Load specific owner's data
        owner_id_str = str(owner_id)
        specific_file = os.path.join(output_dir, f"high_value_followers_{owner_id_str}.json")
        
        if os.path.exists(specific_file):
            try:
                with open(specific_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading high-value followers for owner {owner_id}: {e}")
                return {}
        else:
            logger.warning(f"No high-value followers file found for owner {owner_id}")
            return {}
    
    # Load all available data
    all_data = {}
    available_ids = get_available_owner_ids()
    
    for owner_id_str in available_ids:
        try:
            specific_file = os.path.join(output_dir, f"high_value_followers_{owner_id_str}.json")
            with open(specific_file, "r") as f:
                owner_data = json.load(f)
                all_data[owner_id_str] = owner_data
        except Exception as e:
            logger.error(f"Error loading high-value followers for owner {owner_id_str}: {e}")
    
    return all_data

def get_consolidated_high_value_followers() -> Dict:
    """
    Get consolidated high-value followers data from all individual files.
    This replaces the old consolidated file approach.
    
    Returns:
        Dictionary with all high-value followers data consolidated
    """
    all_data = {}
    available_ids = get_available_owner_ids()
    
    for owner_id_str in available_ids:
        try:
            owner_data = load_high_value_followers(owner_id_str)
            if owner_data:
                # Flatten the data structure
                for username, follower_data in owner_data.items():
                    # Add owner_id to follower data for context
                    follower_data_with_owner = follower_data.copy()
                    follower_data_with_owner['source_owner_id'] = owner_id_str
                    all_data[username] = follower_data_with_owner
        except Exception as e:
            logger.error(f"Error consolidating data for owner {owner_id_str}: {e}")
    
    return all_data

def get_owner_id_from_data() -> Optional[str]:
    """
    Get a suitable owner_id from available data.
    Returns the first available owner_id or None if no data exists.
    """
    # Check if preprocessed data exists to get owner_id
    if os.path.exists("outputs/preprocessed_data.csv"):
        try:
            df = pd.read_csv("outputs/preprocessed_data.csv")
            if 'owner_id' in df.columns and not df.empty:
                return str(df['owner_id'].iloc[0])
        except Exception as e:
            logger.error(f"Error reading preprocessed data: {e}")
    
    # Fallback to first available high-value follower file
    available_ids = get_available_owner_ids()
    return available_ids[0] if available_ids else None

def check_high_value_data_exists() -> bool:
    """Check if any high-value follower data exists."""
    return len(get_available_owner_ids()) > 0

def migrate_from_consolidated_file():
    """
    Helper function to handle migration from old consolidated file approach.
    This is mainly for backward compatibility during transition.
    """
    consolidated_path = "outputs/high_value_followers.json"
    if os.path.exists(consolidated_path):
        logger.info("Found old consolidated high_value_followers.json file")
        # Note: We don't delete it as it might be needed temporarily
        # The new approach will use individual files instead
        pass
