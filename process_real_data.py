import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from tqdm import tqdm
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import community as community_louvain

def process_email_eu_dataset(raw_file, output_dir):
    """
    Process the EU Email Communication Network dataset.
    
    Format: sender receiver timestamp
    """
    print("Processing EU Email Communication Network dataset...")
    
    # Read the raw data
    df = pd.read_csv(raw_file, sep=' ', header=None, names=['source_id', 'target_id', 'timestamp'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Get unique users
    unique_users = sorted(set(df['source_id'].unique()) | set(df['target_id'].unique()))
    print(f"Number of unique users: {len(unique_users)}")
    
    # Create a static graph for community detection
    G = nx.Graph()
    for user in unique_users:
        G.add_node(user)
    
    # Add edges (ignoring timestamps for community detection)
    for _, row in df.iterrows():
        G.add_edge(row['source_id'], row['target_id'])
    
    # Detect communities using Louvain method
    print("Detecting communities...")
    partition = community_louvain.best_partition(G)
    
    # Generate synthetic features for users
    print("Generating user features...")
    feature_dim = 16
    user_features = {}
    
    for user in unique_users:
        # Generate random features
        features = np.random.normal(0, 1, feature_dim)
        # Add some community-based bias to features
        community_id = partition[user]
        community_bias = np.random.normal(community_id, 0.5, feature_dim)
        features += community_bias
        # Normalize
        features = features / np.linalg.norm(features)
        user_features[user] = features
    
    # Create users.csv
    users_df = pd.DataFrame({
        'user_id': unique_users,
        'community_id': [partition[user] for user in unique_users]
    })
    
    # Add features
    for i in range(feature_dim):
        users_df[f'feature_{i}'] = [user_features[user][i] for user in unique_users]
    
    # Save users.csv
    users_file = os.path.join(output_dir, 'users.csv')
    users_df.to_csv(users_file, index=False)
    print(f"Saved user profiles to {users_file}")
    
    # Create interactions.csv
    interactions_df = df.copy()
    interactions_df.columns = ['source_id', 'target_id', 'timestamp']
    
    # Save interactions.csv
    interactions_file = os.path.join(output_dir, 'interactions.csv')
    interactions_df.to_csv(interactions_file, index=False)
    print(f"Saved interactions to {interactions_file}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Number of users: {len(unique_users)}")
    print(f"Number of interactions: {len(df)}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of communities: {len(set(partition.values()))}")
    print(f"Feature dimension: {feature_dim}")
    
    return {
        'users_file': users_file,
        'interactions_file': interactions_file,
        'num_users': len(unique_users),
        'num_interactions': len(df),
        'time_span': (df['timestamp'].min(), df['timestamp'].max()),
        'num_communities': len(set(partition.values())),
        'feature_dim': feature_dim
    }

def process_reddit_dataset(raw_file, output_dir):
    """
    Process the Reddit Hyperlink Network dataset.
    
    Format: TSV with multiple columns including timestamps
    """
    print("Processing Reddit Hyperlink Network dataset...")
    
    # Read the raw data
    df = pd.read_csv(raw_file, sep='\t')
    
    # Extract relevant columns
    df = df[['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']]
    df.columns = ['source_id', 'target_id', 'timestamp']
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Get unique subreddits
    unique_subreddits = sorted(set(df['source_id'].unique()) | set(df['target_id'].unique()))
    print(f"Number of unique subreddits: {len(unique_subreddits)}")
    
    # Create a static graph for community detection
    G = nx.Graph()
    for subreddit in unique_subreddits:
        G.add_node(subreddit)
    
    # Add edges (ignoring timestamps for community detection)
    for _, row in df.iterrows():
        G.add_edge(row['source_id'], row['target_id'])
    
    # Detect communities using Louvain method
    print("Detecting communities...")
    partition = community_louvain.best_partition(G)
    
    # Generate synthetic features for subreddits
    print("Generating subreddit features...")
    feature_dim = 16
    subreddit_features = {}
    
    for subreddit in unique_subreddits:
        # Generate random features
        features = np.random.normal(0, 1, feature_dim)
        # Add some community-based bias to features
        community_id = partition[subreddit]
        community_bias = np.random.normal(community_id, 0.5, feature_dim)
        features += community_bias
        # Normalize
        features = features / np.linalg.norm(features)
        subreddit_features[subreddit] = features
    
    # Create users.csv (subreddits)
    users_df = pd.DataFrame({
        'user_id': unique_subreddits,
        'community_id': [partition[subreddit] for subreddit in unique_subreddits]
    })
    
    # Add features
    for i in range(feature_dim):
        users_df[f'feature_{i}'] = [subreddit_features[subreddit][i] for subreddit in unique_subreddits]
    
    # Save users.csv
    users_file = os.path.join(output_dir, 'users.csv')
    users_df.to_csv(users_file, index=False)
    print(f"Saved subreddit profiles to {users_file}")
    
    # Create interactions.csv
    interactions_df = df.copy()
    
    # Save interactions.csv
    interactions_file = os.path.join(output_dir, 'interactions.csv')
    interactions_df.to_csv(interactions_file, index=False)
    print(f"Saved interactions to {interactions_file}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Number of subreddits: {len(unique_subreddits)}")
    print(f"Number of interactions: {len(df)}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of communities: {len(set(partition.values()))}")
    print(f"Feature dimension: {feature_dim}")
    
    return {
        'users_file': users_file,
        'interactions_file': interactions_file,
        'num_users': len(unique_subreddits),
        'num_interactions': len(df),
        'time_span': (df['timestamp'].min(), df['timestamp'].max()),
        'num_communities': len(set(partition.values())),
        'feature_dim': feature_dim
    }

def process_bitcoin_dataset(raw_file, output_dir):
    """
    Process the Bitcoin OTC Trust Network dataset.
    
    Format: source target rating timestamp
    """
    print("Processing Bitcoin OTC Trust Network dataset...")
    
    # Read the raw data
    df = pd.read_csv(raw_file, sep=',', header=None, names=['source_id', 'target_id', 'rating', 'timestamp'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Get unique users
    unique_users = sorted(set(df['source_id'].unique()) | set(df['target_id'].unique()))
    print(f"Number of unique users: {len(unique_users)}")
    
    # Create a static graph for community detection
    G = nx.Graph()
    for user in unique_users:
        G.add_node(user)
    
    # Add edges (ignoring timestamps for community detection)
    for _, row in df.iterrows():
        # Use positive ratings as edges
        if row['rating'] > 0:
            G.add_edge(row['source_id'], row['target_id'], weight=row['rating'])
    
    # Detect communities using Louvain method
    print("Detecting communities...")
    partition = community_louvain.best_partition(G)
    
    # Generate synthetic features for users
    print("Generating user features...")
    feature_dim = 16
    user_features = {}
    
    for user in unique_users:
        # Generate random features
        features = np.random.normal(0, 1, feature_dim)
        # Add some community-based bias to features
        community_id = partition.get(user, 0)  # Default to community 0 if not in partition
        community_bias = np.random.normal(community_id, 0.5, feature_dim)
        features += community_bias
        # Normalize
        features = features / np.linalg.norm(features)
        user_features[user] = features
    
    # Create users.csv
    users_df = pd.DataFrame({
        'user_id': unique_users,
        'community_id': [partition.get(user, 0) for user in unique_users]  # Default to community 0
    })
    
    # Add features
    for i in range(feature_dim):
        users_df[f'feature_{i}'] = [user_features[user][i] for user in unique_users]
    
    # Save users.csv
    users_file = os.path.join(output_dir, 'users.csv')
    users_df.to_csv(users_file, index=False)
    print(f"Saved user profiles to {users_file}")
    
    # Create interactions.csv
    interactions_df = df[['source_id', 'target_id', 'timestamp']].copy()
    
    # Save interactions.csv
    interactions_file = os.path.join(output_dir, 'interactions.csv')
    interactions_df.to_csv(interactions_file, index=False)
    print(f"Saved interactions to {interactions_file}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Number of users: {len(unique_users)}")
    print(f"Number of interactions: {len(df)}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of communities: {len(set(partition.values()))}")
    print(f"Feature dimension: {feature_dim}")
    
    return {
        'users_file': users_file,
        'interactions_file': interactions_file,
        'num_users': len(unique_users),
        'num_interactions': len(df),
        'time_span': (df['timestamp'].min(), df['timestamp'].max()),
        'num_communities': len(set(partition.values())),
        'feature_dim': feature_dim
    }

def main():
    parser = argparse.ArgumentParser(description='Process real-world datasets for TempGAT')
    parser.add_argument('--dataset', type=str, default='email-eu', 
                        choices=['email-eu', 'reddit', 'bitcoin'],
                        help='Dataset to process')
    parser.add_argument('--raw_dir', type=str, default='data/real_world/raw',
                        help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='data/real_world/processed',
                        help='Directory to save processed data')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of interactions to sample (for large datasets)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process dataset
    if args.dataset == 'email-eu':
        raw_file = os.path.join(args.raw_dir, 'email-Eu-core-temporal.txt')
        dataset_info = process_email_eu_dataset(raw_file, args.output_dir)
    elif args.dataset == 'reddit':
        raw_file = os.path.join(args.raw_dir, 'soc-redditHyperlinks-title.tsv')
        dataset_info = process_reddit_dataset(raw_file, args.output_dir)
    elif args.dataset == 'bitcoin':
        raw_file = os.path.join(args.raw_dir, 'soc-sign-bitcoinotc.csv')
        dataset_info = process_bitcoin_dataset(raw_file, args.output_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create a script to run TempGAT on this dataset
    create_run_script(args.dataset, dataset_info, args.output_dir)
    
    print("\nProcessing completed successfully!")
    print(f"To run TempGAT on this dataset, execute: bash run_tempgat_on_{args.dataset}.sh")

def create_run_script(dataset_name, dataset_info, output_dir):
    """Create a shell script to run TempGAT on the processed dataset."""
    
    # Determine appropriate window size based on time span
    time_span = dataset_info['time_span']
    total_hours = (time_span[1] - time_span[0]).total_seconds() / 3600
    
    # Aim for ~200-300 windows total
    window_minutes = max(15, int(total_hours * 60 / 250))
    
    # Round to nearest 5 minutes
    window_minutes = 5 * round(window_minutes / 5)
    
    # Determine appropriate batch size and sequence length
    num_users = dataset_info['num_users']
    if num_users > 5000:
        batch_size = 4
        sequence_length = 3
    elif num_users > 1000:
        batch_size = 8
        sequence_length = 5
    else:
        batch_size = 16
        sequence_length = 10
    
    # Determine number of epochs based on dataset size
    num_interactions = dataset_info['num_interactions']
    if num_interactions > 100000:
        num_epochs = 5
    elif num_interactions > 10000:
        num_epochs = 10
    else:
        num_epochs = 20
    
    # Create the script content
    script_content = f"""#!/bin/bash

# Script to run TempGAT on the {dataset_name} dataset

# Create directories
mkdir -p results/{dataset_name}

# Preprocess the dataset
echo "=== Preprocessing {dataset_name} dataset ==="
python preprocess_dataset.py \\
  --raw_data_dir {output_dir} \\
  --processed_data_dir data/real_world/processed/{dataset_name} \\
  --window_size {window_minutes}

# Run TempGAT
echo "=== Running TempGAT on {dataset_name} dataset ==="
python run_tempgat_on_social_data.py \\
  --data_path data/real_world/processed/{dataset_name}/temporal_graph_data_{window_minutes}min.pkl \\
  --output_dir results/{dataset_name} \\
  --hidden_dim 64 \\
  --output_dim 32 \\
  --num_heads 8 \\
  --memory_decay 0.9 \\
  --dropout 0.2 \\
  --num_epochs {num_epochs} \\
  --batch_size {batch_size} \\
  --sequence_length {sequence_length} \\
  --learning_rate 0.001 \\
  --scheduler_type plateau \\
  --scheduler_factor 0.5 \\
  --scheduler_patience 3 \\
  --scheduler_min_lr 0.0001 \\
  --task node_classification \\
  --visualize \\
  --seed 42

echo "=== TempGAT run completed ==="
echo "Results saved to results/{dataset_name}"
"""
    
    # Write the script to a file
    script_file = f"run_tempgat_on_{dataset_name}.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_file, 0o755)
    
    print(f"Created run script: {script_file}")

if __name__ == '__main__':
    main()