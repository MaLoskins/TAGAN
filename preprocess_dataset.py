import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle

def preprocess_social_media_dataset(raw_data_dir, processed_data_dir, window_size=15):
    """
    Preprocess the raw social media dataset for use with TempGAT.
    
    Args:
        raw_data_dir: Directory containing raw data files
        processed_data_dir: Directory to save processed data
        window_size: Size of temporal window in minutes
    """
    print("Loading raw data...")
    users_df = pd.read_csv(os.path.join(raw_data_dir, 'users.csv'))
    interactions_df = pd.read_csv(os.path.join(raw_data_dir, 'interactions.csv'))
    
    # Convert timestamp to datetime
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
    
    # Sort interactions by timestamp
    interactions_df = interactions_df.sort_values('timestamp')
    
    # Extract user features
    print("Extracting user features...")
    user_features = extract_user_features(users_df)
    
    # Create node labels (for node classification task)
    # Here we use community_id as the label
    node_labels = users_df[['user_id', 'community_id']].set_index('user_id').to_dict()['community_id']
    
    # Create temporal graph data
    print(f"Creating temporal graph with {window_size}-minute windows...")
    temporal_data = create_temporal_graph_data(interactions_df, user_features, window_size)
    
    # Add node labels to temporal data
    temporal_data['node_labels'] = node_labels
    
    # Create directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Save processed data
    output_path = os.path.join(processed_data_dir, f'temporal_graph_data_{window_size}min.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(temporal_data, f)
    
    print(f"Saved processed data to {output_path}")
    
    # Also save a CSV version of the interactions for easier inspection
    processed_interactions = interactions_df.copy()
    processed_interactions['window_id'] = processed_interactions['timestamp'].apply(
        lambda x: int((x - interactions_df['timestamp'].min()).total_seconds() / (window_size * 60))
    )
    processed_interactions.to_csv(os.path.join(processed_data_dir, 'processed_interactions.csv'), index=False)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Number of users: {len(users_df)}")
    print(f"Number of interactions: {len(interactions_df)}")
    print(f"Number of temporal windows: {len(temporal_data['snapshots'])}")
    print(f"Time span: {interactions_df['timestamp'].min()} to {interactions_df['timestamp'].max()}")
    print(f"Feature dimension: {len(next(iter(user_features.values())))}")
    
    return temporal_data


def extract_user_features(users_df):
    """
    Extract features for each user.
    
    Args:
        users_df: DataFrame of user profiles
        
    Returns:
        Dictionary mapping user_id to feature vector
    """
    # Select features to use
    feature_columns = []
    
    # Check which standard features are available
    standard_features = ['age', 'activity_level', 'influence']
    for feature in standard_features:
        if feature in users_df.columns:
            feature_columns.append(feature)
    
    # Add topic interest columns if they exist
    for i in range(1, 11):
        topic_col = f'topic_{i}_interest'
        if topic_col in users_df.columns:
            feature_columns.append(topic_col)
    
    # If no standard features are found, look for feature_* columns
    if len(feature_columns) == 0:
        feature_columns = [col for col in users_df.columns if col.startswith('feature_')]
        print(f"Using {len(feature_columns)} feature_* columns from the dataset")
    
    # One-hot encode categorical features if available
    categorical_features = None
    if 'gender' in users_df.columns:
        categorical_features = pd.get_dummies(users_df[['gender']], prefix=['gender'])
    
    # Combine numerical and categorical features
    if categorical_features is not None:
        features_df = pd.concat([
            users_df[feature_columns],
            categorical_features
        ], axis=1)
    else:
        # If no categorical features, just use numerical features
        features_df = users_df[feature_columns].copy()
    
    # Check if we have any feature columns
    if len(feature_columns) == 0:
        # Use any feature_* columns as fallback
        feature_cols = [col for col in users_df.columns if col.startswith('feature_')]
        if feature_cols:
            features_df = users_df[feature_cols].copy()
        else:
            # Create dummy features if no features are available
            print("No features found in users.csv. Creating dummy features.")
            features_df = pd.DataFrame(
                np.random.normal(0, 1, (len(users_df), 16)),
                index=users_df.index
            )
    
    # Normalize numerical features
    for col in feature_columns:
        if col in features_df.columns:
            features_df[col] = (features_df[col] - features_df[col].mean()) / features_df[col].std()
    
    # Create dictionary mapping user_id to feature vector
    user_features = {}
    for _, row in users_df.iterrows():
        user_id = row['user_id']
        # Convert to numpy array with float32 dtype
        feature_vector = np.array(features_df.loc[_].values, dtype=np.float32)
        # Replace NaN values with 0
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        user_features[user_id] = feature_vector
    
    return user_features


def create_temporal_graph_data(interactions_df, user_features, window_size):
    """
    Create temporal graph data from interactions.
    
    Args:
        interactions_df: DataFrame of interactions
        user_features: Dictionary mapping user_id to feature vector
        window_size: Size of temporal window in minutes
        
    Returns:
        Dictionary containing temporal graph data
    """
    # Get time range
    start_time = interactions_df['timestamp'].min()
    end_time = interactions_df['timestamp'].max()
    
    # Calculate number of windows
    total_minutes = (end_time - start_time).total_seconds() / 60
    num_windows = int(total_minutes / window_size) + 1
    
    # Create snapshots
    snapshots = []
    
    for window_id in tqdm(range(num_windows), desc="Creating snapshots"):
        window_start = start_time + timedelta(minutes=window_id * window_size)
        window_end = window_start + timedelta(minutes=window_size)
        
        # Get interactions in this window
        window_mask = (interactions_df['timestamp'] >= window_start) & (interactions_df['timestamp'] < window_end)
        window_interactions = interactions_df[window_mask]
        
        # Skip if no interactions in this window
        if len(window_interactions) == 0:
            continue
        
        # Get active nodes (users) in this window
        active_sources = set(window_interactions['source_id'])
        active_targets = set(window_interactions['target_id'])
        # Remove -1 (used for posts with no target)
        if -1 in active_targets:
            active_targets.remove(-1)
        active_nodes = list(active_sources.union(active_targets))
        
        # Create edges
        edges = []
        for _, row in window_interactions.iterrows():
            source_id = row['source_id']
            target_id = row['target_id']
            
            # Skip posts (no target)
            if target_id == -1:
                continue
                
            edges.append((source_id, target_id))
        
        # Create snapshot
        snapshot = {
            'timestamp': window_start,
            'window_id': window_id,
            'window_start': window_id * window_size,  # Minutes since start
            'window_end': (window_id + 1) * window_size,  # Minutes since start
            'active_nodes': active_nodes,
            'edges': edges
        }
        
        snapshots.append(snapshot)
    
    # Create temporal graph data
    temporal_data = {
        'snapshots': snapshots,
        'user_features': user_features,
        'window_size': window_size,
        'start_time': start_time,
        'end_time': end_time
    }
    
    return temporal_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess social media dataset for TempGAT')
    parser.add_argument('--raw_data_dir', type=str, default='data/raw', help='Directory containing raw data')
    parser.add_argument('--processed_data_dir', type=str, default='data/processed', 
                        help='Directory to save processed data')
    parser.add_argument('--window_size', type=int, default=15, help='Size of temporal window in minutes')
    
    args = parser.parse_args()
    
    # Preprocess dataset
    preprocess_social_media_dataset(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        window_size=args.window_size
    )


if __name__ == '__main__':
    main()