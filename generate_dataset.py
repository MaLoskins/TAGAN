import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import random
import os
import argparse
from tqdm import tqdm

def generate_social_media_dataset(
    num_users=1000,
    num_days=30,
    avg_daily_interactions=5000,
    num_communities=10,
    seed=42
):
    """
    Generate a realistic social media interaction dataset.
    
    Args:
        num_users: Number of users in the network
        num_days: Number of days to simulate
        avg_daily_interactions: Average number of interactions per day
        num_communities: Number of communities in the network
        seed: Random seed
        
    Returns:
        Tuple of (users_df, interactions_df)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Generating social media dataset with {num_users} users over {num_days} days...")
    
    # Generate user profiles
    print("Generating user profiles...")
    users = []
    
    # Create communities
    community_sizes = np.random.dirichlet(np.ones(num_communities) * 5, 1)[0]
    community_sizes = (community_sizes * num_users).astype(int)
    # Ensure the sum equals num_users
    community_sizes[-1] = num_users - np.sum(community_sizes[:-1])
    
    user_id = 0
    for community_id, size in enumerate(community_sizes):
        for _ in range(size):
            # Generate user attributes
            age = np.random.normal(30, 10)
            age = max(13, min(80, int(age)))  # Clip to realistic range
            
            gender = random.choice(['M', 'F', 'O'])
            
            # Activity level (1-10)
            activity_level = np.random.gamma(2, 1.5)
            activity_level = max(1, min(10, activity_level))
            
            # Influence score (1-100)
            influence = np.random.gamma(2, 10)
            influence = max(1, min(100, influence))
            
            # Topic interests (10 topics, values 0-1)
            interests = np.random.dirichlet(np.ones(10) * 0.5, 1)[0]
            
            # Create user
            user = {
                'user_id': user_id,
                'community_id': community_id,
                'age': age,
                'gender': gender,
                'join_date': datetime(2023, 1, 1) - timedelta(days=np.random.randint(1, 365*3)),
                'activity_level': activity_level,
                'influence': influence
            }
            
            # Add topic interests
            for i, interest in enumerate(interests):
                user[f'topic_{i+1}_interest'] = interest
            
            users.append(user)
            user_id += 1
    
    users_df = pd.DataFrame(users)
    
    # Generate social network structure
    print("Generating social network structure...")
    G = nx.Graph()
    
    # Add nodes with community attributes
    for _, user in users_df.iterrows():
        G.add_node(user['user_id'], 
                  community=user['community_id'],
                  activity=user['activity_level'],
                  influence=user['influence'])
    
    # Add edges (connections) with higher probability within communities
    for u in tqdm(G.nodes(), desc="Creating connections"):
        u_community = G.nodes[u]['community']
        u_activity = G.nodes[u]['activity']
        
        for v in G.nodes():
            if u == v:
                continue
                
            v_community = G.nodes[v]['community']
            v_activity = G.nodes[v]['activity']
            
            # Higher probability of connection within same community
            if u_community == v_community:
                p = 0.1 * (u_activity / 10) * (v_activity / 10)
            else:
                p = 0.01 * (u_activity / 10) * (v_activity / 10)
                
            if np.random.random() < p:
                G.add_edge(u, v)
    
    print(f"Generated network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate temporal interactions
    print("Generating temporal interactions...")
    interactions = []
    
    # Simulation parameters
    start_date = datetime(2023, 1, 1)
    end_date = start_date + timedelta(days=num_days)
    
    # Daily and weekly patterns
    hourly_weights = np.array([
        0.01, 0.005, 0.001, 0.001, 0.001, 0.005,  # 0-5 AM
        0.01, 0.03, 0.05, 0.06, 0.07, 0.08,       # 6-11 AM
        0.09, 0.09, 0.08, 0.07, 0.08, 0.09,       # 12-5 PM
        0.1, 0.1, 0.09, 0.07, 0.05, 0.02          # 6-11 PM
    ])
    hourly_weights = hourly_weights / hourly_weights.sum()
    
    day_weights = np.array([0.9, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3])  # Mon-Sun
    
    # Interaction types
    interaction_types = ['post', 'comment', 'like', 'share']
    interaction_weights = [0.2, 0.3, 0.4, 0.1]  # Probabilities for each type
    
    # Generate interactions
    current_date = start_date
    interaction_id = 0
    
    while current_date < end_date:
        # Determine number of interactions for this day
        day_of_week = current_date.weekday()
        daily_interactions = int(avg_daily_interactions * day_weights[day_of_week] * 
                               (1 + 0.1 * np.random.randn()))  # Add some noise
        
        # Generate interactions for each hour
        for hour in range(24):
            # Number of interactions in this hour
            hour_interactions = int(daily_interactions * hourly_weights[hour])
            
            if hour_interactions == 0:
                continue
                
            # Generate interactions
            for _ in range(hour_interactions):
                # Randomly select source user with probability proportional to activity level
                activity_levels = users_df['activity_level'].values
                source_probs = activity_levels / activity_levels.sum()
                source_id = np.random.choice(users_df['user_id'].values, p=source_probs)
                
                # Determine interaction type
                interaction_type = np.random.choice(interaction_types, p=interaction_weights)
                
                # For posts, there's no target user
                if interaction_type == 'post':
                    target_id = None
                else:
                    # Select target from source's connections or with small probability, any user
                    if source_id in G and np.random.random() < 0.9:
                        neighbors = list(G.neighbors(source_id))
                        if neighbors:
                            target_id = random.choice(neighbors)
                        else:
                            # If no connections, interact with random user
                            target_id = np.random.choice(users_df['user_id'].values)
                    else:
                        # Interact with random user
                        target_id = np.random.choice(users_df['user_id'].values)
                        
                    # Ensure source != target
                    while target_id == source_id:
                        target_id = np.random.choice(users_df['user_id'].values)
                
                # Create timestamp
                timestamp = current_date + timedelta(hours=hour, 
                                                   minutes=np.random.randint(0, 60),
                                                   seconds=np.random.randint(0, 60))
                
                # Create interaction
                interaction = {
                    'interaction_id': interaction_id,
                    'timestamp': timestamp,
                    'source_id': source_id,
                    'target_id': target_id,
                    'interaction_type': interaction_type
                }
                
                # Add content features for posts and comments
                if interaction_type in ['post', 'comment']:
                    # Get user's interests
                    user_row = users_df[users_df['user_id'] == source_id].iloc[0]
                    interests = [user_row[f'topic_{i+1}_interest'] for i in range(10)]
                    
                    # Generate content features based on user interests
                    for i in range(10):
                        # Higher interest means content is more likely to be about that topic
                        content_value = np.random.beta(1 + interests[i] * 5, 1 + (1-interests[i]) * 5)
                        interaction[f'content_topic_{i+1}'] = content_value
                
                interactions.append(interaction)
                interaction_id += 1
        
        # Move to next day
        current_date += timedelta(days=1)
    
    interactions_df = pd.DataFrame(interactions)
    
    # Fill NaN values in target_id for posts
    interactions_df['target_id'] = interactions_df['target_id'].fillna(-1).astype(int)
    
    print(f"Generated {len(interactions_df)} interactions")
    
    return users_df, interactions_df


def save_dataset(users_df, interactions_df, output_dir):
    """
    Save the generated dataset to CSV files.
    
    Args:
        users_df: DataFrame of user profiles
        interactions_df: DataFrame of interactions
        output_dir: Directory to save the files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save user profiles
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)
    print(f"Saved user profiles to {os.path.join(output_dir, 'users.csv')}")
    
    # Save interactions
    interactions_df.to_csv(os.path.join(output_dir, 'interactions.csv'), index=False)
    print(f"Saved interactions to {os.path.join(output_dir, 'interactions.csv')}")


def main():
    parser = argparse.ArgumentParser(description='Generate social media dataset')
    parser.add_argument('--num_users', type=int, default=1000, help='Number of users')
    parser.add_argument('--num_days', type=int, default=30, help='Number of days to simulate')
    parser.add_argument('--avg_daily_interactions', type=int, default=5000, 
                        help='Average number of interactions per day')
    parser.add_argument('--num_communities', type=int, default=10, help='Number of communities')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Generate dataset
    users_df, interactions_df = generate_social_media_dataset(
        num_users=args.num_users,
        num_days=args.num_days,
        avg_daily_interactions=args.avg_daily_interactions,
        num_communities=args.num_communities,
        seed=args.seed
    )
    
    # Save dataset
    save_dataset(users_df, interactions_df, args.output_dir)


if __name__ == '__main__':
    main()