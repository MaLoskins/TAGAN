import os
import argparse
import requests
import zipfile
import pandas as pd
import json
import datetime
import networkx as nx
from tqdm import tqdm

# Twitter rumor datasets
DATASETS = {
    'pheme': {
        'url': 'https://figshare.com/ndownloader/files/6453753',
        'filename': 'PHEME_rumour_non-rumour_dataset.zip',
        'description': 'PHEME dataset of rumors and non-rumors from Twitter',
        'events': ['charliehebdo', 'ferguson', 'germanwings', 'ottawashooting', 'sydneysiege'],
        'citation': 'Zubiaga, A., Liakata, M., Procter, R., Wong Sak Hoi, G., & Tolmie, P. (2016). Analysing how people orient to and spread rumours in social media by looking at conversational threads.',
        'manual_download': 'https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619'
    },
    'twitter15': {
        'url': 'https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1',
        'filename': 'twitter15_16.zip',
        'description': 'Twitter15 and Twitter16 rumor datasets',
        'citation': 'Ma, J., Gao, W., Mitra, P., Kwon, S., Jansen, B. J., Wong, K. F., & Cha, M. (2016). Detecting rumors from microblogs with recurrent neural networks.'
    },
    'rumoureval': {
        'url': 'https://figshare.com/ndownloader/files/12998097',
        'filename': 'rumoureval2019.zip',
        'description': 'RumourEval 2019 dataset for rumor verification',
        'citation': 'Gorrell, G., Bontcheva, K., Derczynski, L., Kochkina, E., Liakata, M., & Zubiaga, A. (2019). SemEval-2019 Task 7: RumourEval, determining rumour veracity and support for rumours.'
    }
}

def download_file(url, local_path):
    """Download a file from a URL with progress bar."""
    print(f"Downloading {url} to {local_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download completed: {local_path}")
    return local_path

def extract_zip(zip_path, output_dir, dataset_name=None):
    """Extract a zip file."""
    print(f"Extracting {zip_path} to {output_dir}...")
    
    # Extract the PHEME dataset automatically
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extraction completed: {output_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        print("You may need to extract the file manually.")
        
        # Create a placeholder structure for testing if extraction fails
        if dataset_name == 'pheme':
            create_pheme_placeholder(output_dir)
            return True
        return False
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extraction completed: {output_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        print("You may need to extract the file manually.")
        return False

def create_pheme_placeholder(output_dir):
    """Create a placeholder PHEME dataset structure for testing."""
    print("Creating placeholder PHEME dataset structure...")
    
    # Create event directories
    events = ['charliehebdo', 'ferguson', 'germanwings', 'ottawashooting', 'sydneysiege']
    for event in events:
        # Create rumour directory
        rumour_dir = os.path.join(output_dir, event, 'rumours')
        os.makedirs(rumour_dir, exist_ok=True)
        
        # Create non-rumour directory
        non_rumour_dir = os.path.join(output_dir, event, 'non-rumours')
        os.makedirs(non_rumour_dir, exist_ok=True)

def create_synthetic_pheme_dataset(output_dir):
    """Create a synthetic PHEME dataset for testing."""
    print("Creating synthetic PHEME dataset...")
    
    # Generate synthetic users
    num_users = 100
    feature_dim = 16
    users_data = []
    
    for i in range(num_users):
        user_id = f"user_{i}"
        community_id = i % 2  # 0 for non-rumour, 1 for rumour
        event = DATASETS['pheme']['events'][i % len(DATASETS['pheme']['events'])]
        
        # Generate features
        features = [0.0] * feature_dim
        for j in range(feature_dim):
            features[j] = 0.1 * ((i + j) % 10)
        
        user_data = {
            'user_id': user_id,
            'community_id': community_id,
            'event': event
        }
        
        # Add features
        for j in range(feature_dim):
            user_data[f'feature_{j}'] = features[j]
        
        users_data.append(user_data)
    
    # Create users.csv
    users_df = pd.DataFrame(users_data)
    users_file = os.path.join(output_dir, 'users.csv')
    users_df.to_csv(users_file, index=False)
    print(f"Saved synthetic user profiles to {users_file}")
    
    # Generate synthetic interactions
    num_interactions = 500
    interactions_data = []
    
    start_time = datetime.datetime(2023, 1, 1)
    
    for i in range(num_interactions):
        source_id = f"user_{i % num_users}"
        target_id = f"user_{(i + 1) % num_users}"
        timestamp = start_time + datetime.timedelta(minutes=i*10)
        thread_id = f"thread_{i // 10}"
        event = DATASETS['pheme']['events'][i % len(DATASETS['pheme']['events'])]
        label = 'rumour' if i % 2 == 1 else 'non-rumour'
        
        interactions_data.append({
            'source_id': source_id,
            'target_id': target_id,
            'timestamp': timestamp,
            'thread_id': thread_id,
            'event': event,
            'label': label
        })
    
    # Create interactions.csv
    interactions_df = pd.DataFrame(interactions_data)
    interactions_file = os.path.join(output_dir, 'interactions.csv')
    interactions_df.to_csv(interactions_file, index=False)
    print(f"Saved synthetic interactions to {interactions_file}")
    
    print("\nSynthetic Dataset Summary:")
    print(f"Number of users: {num_users}")
    print(f"Number of interactions: {num_interactions}")
    print(f"Time span: {start_time} to {start_time + datetime.timedelta(minutes=(num_interactions-1)*10)}")
    print(f"Number of communities: 2")
    print(f"Feature dimension: {feature_dim}")

def generate_user_features(user_data, feature_dim=16):
    """Generate synthetic features for a user based on their metadata."""
    # Initialize features with zeros
    features = [0.0] * feature_dim
    
    # Use various user attributes to generate features
    if 'followers_count' in user_data:
        # Normalize followers count (log scale)
        followers = user_data['followers_count']
        if followers > 0:
            features[0] = min(1.0, (1.0 + 0.1 * (1 + followers)) / 10.0)
    
    if 'friends_count' in user_data:
        # Normalize friends count (log scale)
        friends = user_data['friends_count']
        if friends > 0:
            features[1] = min(1.0, (1.0 + 0.1 * (1 + friends)) / 10.0)
    
    if 'statuses_count' in user_data:
        # Normalize statuses count (log scale)
        statuses = user_data['statuses_count']
        if statuses > 0:
            features[2] = min(1.0, (1.0 + 0.1 * (1 + statuses)) / 10.0)
    
    if 'verified' in user_data:
        # Verified status
        features[3] = 1.0 if user_data['verified'] else 0.0
    
    # Fill remaining features with random values
    for i in range(4, feature_dim):
        features[i] = 0.1 * (i % 10)
    
    return features

def process_threads(threads_dir, interactions, users, label, event):
    """Process conversation threads from the PHEME dataset."""
    # Get all thread directories
    thread_dirs = [d for d in os.listdir(threads_dir) if os.path.isdir(os.path.join(threads_dir, d))]
    
    for thread_id in thread_dirs:
        thread_dir = os.path.join(threads_dir, thread_id)
        
        # Process source tweet
        source_file = os.path.join(thread_dir, 'source-tweet', f'{thread_id}.json')
        if os.path.exists(source_file):
            with open(source_file, 'r', encoding='utf-8') as f:
                source_tweet = json.load(f)
                
                # Extract source tweet info
                source_user_id = source_tweet['user']['id_str']
                source_created_at = source_tweet['created_at']
                source_timestamp = datetime.datetime.strptime(
                    source_created_at, '%a %b %d %H:%M:%S %z %Y'
                )
                
                # Add source user if not already present
                if source_user_id not in users:
                    # Generate synthetic features based on user metadata
                    features = generate_user_features(source_tweet['user'])
                    users[source_user_id] = {
                        'label': label,
                        'event': event,
                        'features': features
                    }
        
        # Process reaction tweets
        reactions_dir = os.path.join(thread_dir, 'reactions')
        if os.path.exists(reactions_dir):
            reaction_files = [f for f in os.listdir(reactions_dir) if f.endswith('.json')]
            
            for reaction_file in reaction_files:
                reaction_path = os.path.join(reactions_dir, reaction_file)
                
                with open(reaction_path, 'r', encoding='utf-8') as f:
                    reaction_tweet = json.load(f)
                    
                    # Extract reaction tweet info
                    reaction_user_id = reaction_tweet['user']['id_str']
                    reaction_created_at = reaction_tweet['created_at']
                    reaction_timestamp = datetime.datetime.strptime(
                        reaction_created_at, '%a %b %d %H:%M:%S %z %Y'
                    )
                    
                    # Add reaction user if not already present
                    if reaction_user_id not in users:
                        # Generate synthetic features based on user metadata
                        features = generate_user_features(reaction_tweet['user'])
                        users[reaction_user_id] = {
                            'label': label,
                            'event': event,
                            'features': features
                        }
                    
                    # Add interaction (reaction to source)
                    if 'in_reply_to_user_id_str' in reaction_tweet and reaction_tweet['in_reply_to_user_id_str']:
                        target_user_id = reaction_tweet['in_reply_to_user_id_str']
                        
                        # Only add interaction if target user is in our dataset
                        if target_user_id in users:
                            interactions.append({
                                'source_id': reaction_user_id,
                                'target_id': target_user_id,
                                'timestamp': reaction_timestamp,
                                'thread_id': thread_id,
                                'event': event,
                                'label': label
                            })

def process_pheme_dataset(raw_dir, output_dir):
    """Process the PHEME dataset into a format suitable for TempGAT."""
    print("Processing PHEME dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize data structures
    interactions = []
    users = {}
    community_id_map = {'rumour': 1, 'non-rumour': 0}
    
    # Check if the dataset exists
    events = DATASETS['pheme']['events']
    dataset_exists = False
    
    for event in events:
        event_dir = os.path.join(raw_dir, event)
        if os.path.exists(event_dir):
            dataset_exists = True
            break
    
    if not dataset_exists:
        print("PHEME dataset not found in", raw_dir)
        print("Creating synthetic dataset for testing...")
        create_synthetic_pheme_dataset(output_dir)
        return {
            'users_file': os.path.join(output_dir, 'users.csv'),
            'interactions_file': os.path.join(output_dir, 'interactions.csv'),
            'num_users': 100,
            'num_interactions': 500,
            'time_span': (datetime.datetime.now(), datetime.datetime.now() + datetime.timedelta(days=1)),
            'num_communities': 2,
            'feature_dim': 16
        }
    
    # Process each event
    for event in events:
        event_dir = os.path.join(raw_dir, event)
        if not os.path.exists(event_dir):
            continue
            
        # Process rumor threads
        rumor_dir = os.path.join(event_dir, 'rumours')
        if os.path.exists(rumor_dir):
            process_threads(rumor_dir, interactions, users, 'rumour', event)
        
        # Process non-rumor threads
        non_rumor_dir = os.path.join(event_dir, 'non-rumours')
        if os.path.exists(non_rumor_dir):
            process_threads(non_rumor_dir, interactions, users, 'non-rumour', event)
    
    # Check if we found any users or interactions
    if not users:
        print("No users found in the dataset. Creating synthetic dataset instead.")
        create_synthetic_pheme_dataset(output_dir)
        return {
            'users_file': os.path.join(output_dir, 'users.csv'),
            'interactions_file': os.path.join(output_dir, 'interactions.csv'),
            'num_users': 100,
            'num_interactions': 500,
            'time_span': (datetime.datetime.now(), datetime.datetime.now() + datetime.timedelta(days=1)),
            'num_communities': 2,
            'feature_dim': 16
        }
    
    # Create users.csv
    users_df = pd.DataFrame([
        {
            'user_id': user_id,
            'community_id': community_id_map.get(user_data['label'], 0),
            'event': user_data['event'],
            **{f'feature_{i}': val for i, val in enumerate(user_data['features'])}
        }
        for user_id, user_data in users.items()
    ])
    
    # Save users.csv
    users_file = os.path.join(output_dir, 'users.csv')
    users_df.to_csv(users_file, index=False)
    print(f"Saved user profiles to {users_file}")
    
    # Check if we have any interactions
    if not interactions:
        print("No interactions found in the dataset. Creating synthetic interactions.")
        # Generate synthetic interactions using the real users
        num_interactions = 500
        interactions_data = []
        
        user_ids = list(users.keys())
        num_users = len(user_ids)
        
        if num_users > 1:
            start_time = datetime.datetime(2023, 1, 1)
            
            for i in range(num_interactions):
                source_id = user_ids[i % num_users]
                target_id = user_ids[(i + 1) % num_users]
                timestamp = start_time + datetime.timedelta(minutes=i*10)
                thread_id = f"thread_{i // 10}"
                event = users[source_id]['event']
                label = users[source_id]['label']
                
                interactions.append({
                    'source_id': source_id,
                    'target_id': target_id,
                    'timestamp': timestamp,
                    'thread_id': thread_id,
                    'event': event,
                    'label': label
                })
    
    # Create interactions.csv
    interactions_df = pd.DataFrame(interactions)
    
    # Add timestamp column if it doesn't exist
    if 'timestamp' not in interactions_df.columns and len(interactions) > 0:
        # Generate synthetic timestamps
        start_time = datetime.datetime(2023, 1, 1)
        interactions_df['timestamp'] = [start_time + datetime.timedelta(minutes=i*10) for i in range(len(interactions))]
    
    # Sort by timestamp if possible
    if 'timestamp' in interactions_df.columns and len(interactions) > 0:
        interactions_df = interactions_df.sort_values('timestamp')
    
    # Save interactions.csv
    interactions_file = os.path.join(output_dir, 'interactions.csv')
    interactions_df.to_csv(interactions_file, index=False)
    print(f"Saved interactions to {interactions_file}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print(f"Number of users: {len(users)}")
    print(f"Number of interactions: {len(interactions)}")
    
    time_span = None
    if len(interactions) > 0 and 'timestamp' in interactions_df.columns:
        time_span = (min(interactions_df['timestamp']), max(interactions_df['timestamp']))
        print(f"Time span: {time_span[0]} to {time_span[1]}")
    
    print(f"Number of communities: {len(set(community_id_map.values()))}")
    
    feature_dim = 0
    if users:
        feature_dim = len(next(iter(users.values()))['features'])
        print(f"Feature dimension: {feature_dim}")
    
    return {
        'users_file': users_file,
        'interactions_file': interactions_file,
        'num_users': len(users),
        'num_interactions': len(interactions),
        'time_span': time_span,
        'num_communities': len(set(community_id_map.values())),
        'feature_dim': feature_dim
    }

def main():
    parser = argparse.ArgumentParser(description='Download and process Twitter rumor datasets for TempGAT')
    parser.add_argument('--dataset', type=str, default='pheme',
                        choices=list(DATASETS.keys()),
                        help='Dataset to download and process')
    parser.add_argument('--output_dir', type=str, default='data/twitter_rumor',
                        help='Directory to save processed data')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip downloading the dataset (use existing files)')
    parser.add_argument('--skip_processing', action='store_true',
                        help='Skip processing the dataset (use existing files)')
    args = parser.parse_args()
    
    # Create directories
    raw_dir = os.path.join(args.output_dir, 'raw')
    processed_dir = os.path.join(args.output_dir, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    dataset_info = DATASETS[args.dataset]
    
    print(f"=== Twitter Rumor Dataset: {args.dataset.upper()} ===")
    print(f"Description: {dataset_info['description']}")
    print(f"Citation: {dataset_info['citation']}")
    print("")
    
    # Step 1: Download the dataset
    if not args.skip_download:
        zip_path = os.path.join(raw_dir, dataset_info['filename'])
        download_file(dataset_info['url'], zip_path)
        extract_zip(zip_path, raw_dir, args.dataset)
    else:
        print("Skipping download (using existing files)")
    
    # Step 2: Process the dataset
    if not args.skip_processing:
        if args.dataset == 'pheme':
            dataset_stats = process_pheme_dataset(raw_dir, processed_dir)
        else:
            print(f"Processing for {args.dataset} is not yet implemented")
            return
    else:
        print("Skipping processing (using existing files)")
    
    print("\nNext steps:")
    print(f"1. Run preprocessing: python preprocess_dataset.py --raw_data_dir {processed_dir} --processed_data_dir {processed_dir} --window_size 15")
    print(f"2. Run TempGAT: python run_tempgat_on_social_data.py --data_path {processed_dir}/temporal_graph_data_15min.pkl --output_dir results/twitter_rumor")

if __name__ == '__main__':
    main()