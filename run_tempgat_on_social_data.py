import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import argparse
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm

from src.data.data import TemporalGraph
from src.model.model import TempGAT
from src.training.trainer import TemporalTrainer
from src.metrics.metrics import computational_efficiency_metrics, evaluate_long_term_dependencies
from src.utils.utils import logger


class SocialMediaTemporalGraph(TemporalGraph):
    """
    Extension of TemporalGraph for social media data.
    """
    
    @classmethod
    def from_processed_data(cls, processed_data):
        """
        Create a temporal graph from preprocessed social media data.
        
        Args:
            processed_data: Dictionary containing processed temporal graph data
            
        Returns:
            A TemporalGraph instance
        """
        # Create a new temporal graph
        temporal_graph = cls(window_size=processed_data['window_size'])
        
        # Debug logging to check the snapshots before assignment
        print(f"DEBUG: Processing snapshots of type: {type(processed_data['snapshots'])}")
        if processed_data['snapshots'] and len(processed_data['snapshots']) > 0:
            print(f"DEBUG: First snapshot in from_processed_data: {type(processed_data['snapshots'][0])}")
            
        # Check if snapshots are strings and try to convert them to dictionaries
        if processed_data['snapshots'] and isinstance(processed_data['snapshots'][0], str):
            print("DEBUG: Detected string snapshots, attempting to convert to dictionaries...")
            try:
                import json
                converted_snapshots = []
                for snapshot_str in processed_data['snapshots']:
                    try:
                        snapshot_dict = json.loads(snapshot_str)
                        converted_snapshots.append(snapshot_dict)
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: Error decoding snapshot JSON: {e}")
                        # Skip this snapshot
                processed_data['snapshots'] = converted_snapshots
                print(f"DEBUG: Converted {len(converted_snapshots)} snapshots to dictionaries")
            except Exception as e:
                print(f"DEBUG: Error converting snapshots: {e}")
        
        # Set attributes
        temporal_graph.snapshots = processed_data['snapshots']
        
        # Process node features to ensure they're numpy arrays
        user_features = {}
        for node_id, features in processed_data['user_features'].items():
            user_features[node_id] = np.array(features, dtype=np.float32)
        
        temporal_graph.node_features = user_features
        
        # Create node ID mapping
        all_nodes = set()
        for snapshot in temporal_graph.snapshots:
            if isinstance(snapshot, dict) and 'active_nodes' in snapshot:
                all_nodes.update(snapshot['active_nodes'])
            else:
                print(f"DEBUG: Skipping invalid snapshot: {type(snapshot)}")
        
        for i, node_id in enumerate(sorted(all_nodes)):
            temporal_graph.node_id_map[node_id] = i
            temporal_graph.reverse_node_id_map[i] = node_id
        
        temporal_graph.num_nodes = len(temporal_graph.node_id_map)
        
        # Set feature dimension
        if temporal_graph.node_features:
            first_node = next(iter(temporal_graph.node_features))
            temporal_graph.feature_dim = len(temporal_graph.node_features[first_node])
        
        # Add node labels if available
        if 'node_labels' in processed_data:
            temporal_graph.node_labels = processed_data['node_labels']
        
        return temporal_graph


def load_processed_data(data_path):
    """
    Load preprocessed temporal graph data.
    
    Args:
        data_path: Path to processed data file
        
    Returns:
        Processed data dictionary
    """
    with open(data_path, 'rb') as f:
        processed_data = pickle.load(f)
    
    # Debug logging to check the loaded data structure
    print(f"DEBUG: Loaded processed data type: {type(processed_data)}")
    if 'snapshots' in processed_data:
        print(f"DEBUG: Snapshots type: {type(processed_data['snapshots'])}")
        if processed_data['snapshots'] and len(processed_data['snapshots']) > 0:
            print(f"DEBUG: First snapshot type: {type(processed_data['snapshots'][0])}")
            if isinstance(processed_data['snapshots'][0], str):
                print(f"DEBUG: First snapshot content (first 100 chars): {processed_data['snapshots'][0][:100]}")
    
    return processed_data


def visualize_snapshot(temporal_graph, snapshot_idx=0, title=None):
    """
    Visualize a snapshot of the temporal graph.
    
    Args:
        temporal_graph: TemporalGraph instance
        snapshot_idx: Index of the snapshot to visualize
        title: Title for the plot
    """
    if snapshot_idx >= len(temporal_graph.snapshots):
        print(f"Snapshot index {snapshot_idx} out of range. Max index: {len(temporal_graph.snapshots)-1}")
        return
    
    snapshot = temporal_graph.snapshots[snapshot_idx]
    active_nodes = snapshot['active_nodes']
    edges = snapshot['edges']
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_id in active_nodes:
        # Get node label if available
        if hasattr(temporal_graph, 'node_labels') and node_id in temporal_graph.node_labels:
            label = temporal_graph.node_labels[node_id]
            G.add_node(node_id, label=label)
        else:
            G.add_node(node_id)
    
    # Add edges
    for source, target in edges:
        G.add_edge(source, target)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Use different colors for different communities if labels are available
    if hasattr(temporal_graph, 'node_labels'):
        # Get unique labels
        labels = [temporal_graph.node_labels.get(node_id, 0) for node_id in G.nodes()]
        unique_labels = sorted(set(labels))
        
        # Create color map
        cmap = plt.cm.get_cmap('tab10', len(unique_labels))
        node_colors = [cmap(unique_labels.index(label)) for label in labels]
        
        # Create position layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with colors
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=cmap(i), markersize=10, 
                                     label=f'Community {label}')
                          for i, label in enumerate(unique_labels)]
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        # Draw without colors
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=100, 
                font_size=8, font_weight='bold', arrows=True)
    
    if title:
        plt.title(title)
    else:
        plt.title(f"Snapshot at {snapshot['timestamp']}")
    
    plt.tight_layout()
    plt.show()


def visualize_community_embeddings(model, temporal_graph, snapshot_idx=-1):
    """
    Visualize node embeddings colored by community.
    
    Args:
        model: Trained TempGAT model
        temporal_graph: TemporalGraph instance
        snapshot_idx: Index of the snapshot to visualize (-1 for last snapshot)
    """
    if not hasattr(temporal_graph, 'node_labels'):
        print("No community labels available for visualization")
        return
    
    if snapshot_idx < 0:
        snapshot_idx = len(temporal_graph.snapshots) + snapshot_idx
    
    if snapshot_idx >= len(temporal_graph.snapshots) or snapshot_idx < 0:
        print(f"Snapshot index {snapshot_idx} out of range. Max index: {len(temporal_graph.snapshots)-1}")
        return
    
    # Get snapshot
    snapshot = temporal_graph.snapshots[snapshot_idx]
    
    # Get embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(temporal_graph, [snapshot])
    
    # Convert to numpy (detach first to remove gradient requirements)
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Get community labels for active nodes
    active_nodes = snapshot['active_nodes']
    community_labels = [temporal_graph.node_labels.get(node_id, 0) for node_id in active_nodes]
    unique_labels = sorted(set(community_labels))
    
    # Check if we have enough samples for visualization
    n_samples = embeddings_np.shape[0]
    
    if n_samples < 5:
        print(f"Not enough samples ({n_samples}) for visualization. Need at least 5 samples.")
        return
    
    # Apply dimensionality reduction
    if n_samples >= 30:  # t-SNE works well with more samples
        print(f"Using t-SNE for visualization with {n_samples} samples")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        embeddings_2d = tsne.fit_transform(embeddings_np)
    else:
        # Use PCA for smaller sample sizes
        print(f"Using PCA for visualization with {n_samples} samples")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Create color map
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    
    # Plot each community with a different color
    for i, label in enumerate(unique_labels):
        mask = np.array(community_labels) == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=[cmap(i)], label=f'Community {label}', alpha=0.7, s=100)
    
    plt.title(f"Node Embeddings by Community at {snapshot['timestamp']}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def run_tempgat_on_social_data(data_path, output_dir, params):
    """
    Run TempGAT on social media data.
    
    Args:
        data_path: Path to processed data file
        output_dir: Directory to save results
        params: Dictionary of model parameters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed data
    print(f"Loading processed data from {data_path}...")
    processed_data = load_processed_data(data_path)
    
    # Create temporal graph
    print("Creating temporal graph...")
    temporal_graph = SocialMediaTemporalGraph.from_processed_data(processed_data)
    
    print(f"Created temporal graph with {len(temporal_graph.snapshots)} snapshots")
    print(f"Total nodes: {temporal_graph.num_nodes}")
    print(f"Feature dimension: {temporal_graph.feature_dim}")
    
    # Visualize a snapshot
    if params['visualize']:
        print("Visualizing first snapshot...")
        visualize_snapshot(temporal_graph, snapshot_idx=0)
    
    # Create model
    print("Creating TempGAT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TempGAT(
        input_dim=temporal_graph.feature_dim,
        hidden_dim=params['hidden_dim'],
        output_dim=params['output_dim'],
        num_heads=params['num_heads'],
        memory_decay_factor=params['memory_decay'],
        dropout=params['dropout'],
        max_memory_size=1000
    ).to(device)
    
    # Split snapshots into train, validation, and test sets
    num_snapshots = len(temporal_graph.snapshots)
    train_size = int(0.7 * num_snapshots)
    val_size = int(0.15 * num_snapshots)
    test_size = num_snapshots - train_size - val_size
    
    train_snapshots = temporal_graph.snapshots[:train_size]
    val_snapshots = temporal_graph.snapshots[train_size:train_size+val_size]
    test_snapshots = temporal_graph.snapshots[train_size+val_size:]
    
    print(f"Train snapshots: {len(train_snapshots)}")
    print(f"Validation snapshots: {len(val_snapshots)}")
    print(f"Test snapshots: {len(test_snapshots)}")
    
    # Create a new temporal graph with only training snapshots for the trainer
    train_temporal_graph = SocialMediaTemporalGraph(window_size=temporal_graph.window_size)
    train_temporal_graph.snapshots = train_snapshots
    train_temporal_graph.node_features = temporal_graph.node_features
    train_temporal_graph.node_id_map = temporal_graph.node_id_map
    train_temporal_graph.reverse_node_id_map = temporal_graph.reverse_node_id_map
    train_temporal_graph.num_nodes = temporal_graph.num_nodes
    train_temporal_graph.feature_dim = temporal_graph.feature_dim
    if hasattr(temporal_graph, 'node_labels'):
        train_temporal_graph.node_labels = temporal_graph.node_labels
    
    # Create trainer
    print("Creating trainer...")
    trainer = TemporalTrainer(model, train_temporal_graph, device=device)
    
    # Train model
    print("Training model...")
    start_time = time.time()
    history = trainer.train(
        num_epochs=params['num_epochs'],
        batch_size=params['batch_size'],
        sequence_length=params['sequence_length'],
        learning_rate=params['learning_rate'],
        task=params['task'],
        verbose=True,
        scheduler_type=params.get('scheduler_type', 'plateau') if params.get('scheduler_type') != 'none' else None,
        scheduler_params={
            'factor': params.get('scheduler_factor', 0.5),        # Reduce LR by specified factor
            'patience': params.get('scheduler_patience', 3),      # Wait specified epochs before reducing LR
            'min_lr': params.get('scheduler_min_lr', 1e-6)        # Don't reduce LR below this value
        }
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training history
    if params['visualize']:
        print("Plotting training history...")
        trainer.plot_training_history()
    
    # Save model
    model_path = os.path.join(output_dir, 'tempgat_model.pt')
    trainer.save_model(model_path)
    print(f"Saved model to {model_path}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    # Create sequences for evaluation
    test_sequences = []
    seq_length = params['sequence_length']
    
    # Debug the test snapshots
    print(f"Test snapshots type: {type(test_snapshots)}")
    if test_snapshots and len(test_snapshots) > 0:
        print(f"First test snapshot type: {type(test_snapshots[0])}")
    
    # Ensure test_snapshots contains dictionaries
    valid_test_snapshots = []
    for snapshot in test_snapshots:
        if isinstance(snapshot, dict):
            valid_test_snapshots.append(snapshot)
        else:
            print(f"Skipping invalid test snapshot of type: {type(snapshot)}")
    
    print(f"Valid test snapshots: {len(valid_test_snapshots)}")
    
    # Create sequences from valid snapshots
    for i in range(0, max(0, len(valid_test_snapshots) - seq_length + 1)):
        # Create a proper sequence of snapshot dictionaries
        sequence = valid_test_snapshots[i:i+seq_length]
        if len(sequence) == seq_length:
            # Verify that all items in the sequence are dictionaries
            if all(isinstance(s, dict) for s in sequence):
                test_sequences.append(sequence)
            else:
                print(f"Skipping sequence with non-dictionary items at index {i}")
    
    print(f"Created {len(test_sequences)} test sequences")
    
    # Evaluate each sequence
    test_losses = []
    test_metrics = []
    
    for sequence in tqdm(test_sequences, desc="Evaluating test sequences"):
        try:
            # Double-check that sequence is a list of dictionaries
            if isinstance(sequence, list) and all(isinstance(s, dict) and 'active_nodes' in s for s in sequence):
                # Pass the sequence directly to evaluate, not nested in another list
                loss, metrics = trainer.evaluate([sequence], task=params['task'])
                test_losses.append(loss)
                test_metrics.append(metrics)
            else:
                print(f"Skipping invalid sequence: {type(sequence)}")
                if isinstance(sequence, list):
                    for i, s in enumerate(sequence):
                        print(f"  Item {i} type: {type(s)}")
        except Exception as e:
            print(f"Error evaluating sequence: {e}")
    
    # Compute average metrics if we have any
    if test_losses:
        avg_test_loss = np.mean(test_losses)
        print(f"Test Loss: {avg_test_loss:.4f}")
        
        if test_metrics:
            avg_test_metrics = {}
            for key in test_metrics[0].keys():
                avg_test_metrics[key] = np.mean([m[key] for m in test_metrics])
            
            for key, value in avg_test_metrics.items():
                print(f"Test {key}: {value:.4f}")
        else:
            print("No test metrics available")
    else:
        print("No test results available. All test sequences were invalid.")
        
        # Create a simple test sequence for demonstration
        print("Creating a simple test sequence for demonstration...")
        
        # Get a single snapshot from the test set
        if test_snapshots:
            # Make sure we have a valid snapshot (dictionary)
            valid_snapshot = None
            for snapshot in test_snapshots:
                if isinstance(snapshot, dict) and 'active_nodes' in snapshot:
                    valid_snapshot = snapshot
                    break
            
            if valid_snapshot:
                # Create a proper sequence with a single snapshot
                simple_sequence = [valid_snapshot]
                
                try:
                    # Evaluate on this simple sequence - pass the sequence directly, not nested
                    loss, metrics = trainer.evaluate([simple_sequence], task=params['task'])
                    print(f"Simple test loss: {loss:.4f}")
                    for key, value in metrics.items():
                        print(f"Simple test {key}: {value:.4f}")
                except Exception as e:
                    print(f"Error evaluating simple sequence: {e}")
            else:
                print("No valid snapshots found for testing")
    
    # Compute efficiency metrics
    print("Computing efficiency metrics...")
    efficiency_metrics = computational_efficiency_metrics(
        model, 
        temporal_graph, 
        test_sequences[0], 
        device=device
    )
    
    print("Computational efficiency metrics:")
    print(f"  Total nodes processed: {efficiency_metrics['total_nodes']}")
    print(f"  Inference time: {efficiency_metrics['inference_time']:.4f} seconds")
    print(f"  Nodes per second: {efficiency_metrics['nodes_per_second']:.2f}")
    if device.type == 'cuda':
        print(f"  Peak memory usage: {efficiency_metrics['peak_memory_mb']:.2f} MB")
    
    # Evaluate long-term dependencies
    print("Evaluating long-term dependencies...")
    long_term_results = evaluate_long_term_dependencies(
        model,
        temporal_graph,
        test_sequences,
        time_gaps=[1, 2, 3, 5],
        task=params['task'],
        device=device
    )
    
    print("Long-term dependency results:")
    for i, gap in enumerate(long_term_results['time_gaps']):
        print(f"  Time gap {gap}:")
        for key, value in long_term_results['metrics'][i].items():
            print(f"    {key}: {value:.4f}")
    
    # Visualize community embeddings
    if params['visualize'] and hasattr(temporal_graph, 'node_labels'):
        print("Visualizing community embeddings...")
        visualize_community_embeddings(model, temporal_graph)
    
    print("Done!")
    
    return model, trainer, temporal_graph


def main():
    parser = argparse.ArgumentParser(description='Run TempGAT on social media data')
    parser.add_argument('--data_path', type=str, default='data/processed/temporal_graph_data_15min.pkl',
                        help='Path to processed data file')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=32, help='Output dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--memory_decay', type=float, default=0.9, help='Memory decay factor')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--task', type=str, default='node_classification', 
                        choices=['node_classification', 'link_prediction'], 
                        help='Task to train for')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--scheduler_type', type=str, default='plateau',
                        choices=['plateau', 'cosine', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor by which to reduce learning rate (for plateau scheduler)')
    parser.add_argument('--scheduler_patience', type=int, default=3,
                        help='Number of epochs with no improvement before reducing LR (for plateau scheduler)')
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Convert args to dictionary
    params = vars(args)
    
    # Run TempGAT on social media data
    run_tempgat_on_social_data(args.data_path, args.output_dir, params)


if __name__ == '__main__':
    main()