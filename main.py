import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import argparse
import time

from data import TemporalGraph, create_temporal_batches
from memory import MemoryBank
from model import TempGAT
from trainer import TemporalTrainer
from metrics import (
    temporal_node_classification_metrics,
    temporal_link_prediction_metrics,
    computational_efficiency_metrics,
    compare_with_full_graph_gat,
    evaluate_long_term_dependencies,
    evaluate_scalability
)


def generate_synthetic_data(num_nodes=100, num_timesteps=50, window_size=5, seed=42):
    """
    Generate synthetic temporal graph data for testing.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_timesteps: Number of timesteps
        window_size: Size of the temporal window in minutes
        seed: Random seed
        
    Returns:
        DataFrame of interactions
    """
    np.random.seed(seed)
    
    # Generate node features
    node_features = np.random.randn(num_nodes, 10)  # 10-dimensional features
    
    # Generate temporal interactions
    interactions = []
    
    for t in range(num_timesteps):
        # Determine active nodes for this timestep
        active_prob = 0.3  # Probability of a node being active
        active_nodes = np.random.choice(
            num_nodes, 
            size=int(num_nodes * active_prob), 
            replace=False
        )
        
        # Generate interactions between active nodes
        for i in range(len(active_nodes)):
            source = active_nodes[i]
            
            # Each node interacts with ~3 other nodes on average
            num_interactions = np.random.poisson(3)
            targets = np.random.choice(
                active_nodes, 
                size=min(num_interactions, len(active_nodes)), 
                replace=False
            )
            
            for target in targets:
                if source != target:
                    # Create interaction with timestamp
                    timestamp = pd.Timestamp('2023-01-01') + pd.Timedelta(minutes=t*window_size)
                    
                    # Add features
                    features = {}
                    for j in range(10):
                        features[f'feature{j+1}'] = node_features[source, j]
                    
                    # Create interaction record
                    interaction = {
                        'timestamp': timestamp,
                        'source_id': source,
                        'target_id': target,
                        **features
                    }
                    
                    interactions.append(interaction)
    
    # Create DataFrame
    df = pd.DataFrame(interactions)
    
    return df


def visualize_temporal_graph(temporal_graph, snapshot_idx=0):
    """
    Visualize a snapshot of the temporal graph.
    
    Args:
        temporal_graph: TemporalGraph instance
        snapshot_idx: Index of the snapshot to visualize
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
        G.add_node(node_id)
    
    # Add edges
    for source, target in edges:
        G.add_edge(source, target)
    
    # Plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, 
            font_size=10, font_weight='bold', arrows=True)
    
    plt.title(f"Snapshot at {snapshot['timestamp']}")
    plt.show()


def visualize_node_embeddings(model, temporal_graph, snapshot_idx=-1):
    """
    Visualize node embeddings using t-SNE or PCA depending on the number of samples.
    
    Args:
        model: Trained TempGAT model
        temporal_graph: TemporalGraph instance
        snapshot_idx: Index of the snapshot to visualize (-1 for last snapshot)
    """
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
    
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    # Check if we have enough samples for t-SNE
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
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
    
    # Add node labels
    for i, node_id in enumerate(snapshot['active_nodes']):
        plt.annotate(str(node_id), (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)
    
    plt.title(f"t-SNE Visualization of Node Embeddings at {snapshot['timestamp']}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()


def main():
    """Main function to demonstrate TempGAT usage."""
    parser = argparse.ArgumentParser(description='TempGAT Demo')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--data_path', type=str, default=None, help='Path to real data')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes for synthetic data')
    parser.add_argument('--num_timesteps', type=int, default=50, help='Number of timesteps for synthetic data')
    parser.add_argument('--window_size', type=int, default=15, help='Window size in minutes')
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
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load or generate data
    if args.synthetic:
        print("Generating synthetic data...")
        interactions_df = generate_synthetic_data(
            num_nodes=args.num_nodes,
            num_timesteps=args.num_timesteps,
            window_size=args.window_size,
            seed=args.seed
        )
    elif args.data_path:
        print(f"Loading data from {args.data_path}...")
        interactions_df = pd.read_csv(args.data_path)
    else:
        print("No data source specified. Using synthetic data...")
        interactions_df = generate_synthetic_data(
            num_nodes=args.num_nodes,
            num_timesteps=args.num_timesteps,
            window_size=args.window_size,
            seed=args.seed
        )
    
    # Create temporal graph
    print("Creating temporal graph...")
    feature_columns = [f'feature{i+1}' for i in range(10)]
    temporal_graph = TemporalGraph.from_interactions(
        interactions_df,
        time_column='timestamp',
        source_column='source_id',
        target_column='target_id',
        features_columns=feature_columns,
        window_size=args.window_size
    )
    
    print(f"Created temporal graph with {len(temporal_graph.snapshots)} snapshots")
    print(f"Total nodes: {temporal_graph.num_nodes}")
    print(f"Feature dimension: {temporal_graph.feature_dim}")
    
    # Visualize a snapshot
    if args.visualize:
        print("Visualizing first snapshot...")
        visualize_temporal_graph(temporal_graph, snapshot_idx=0)
    
    # Create model
    print("Creating TempGAT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TempGAT(
        input_dim=temporal_graph.feature_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        memory_decay_factor=args.memory_decay,
        dropout=args.dropout
    ).to(device)
    
    # Create trainer
    print("Creating trainer...")
    trainer = TemporalTrainer(model, temporal_graph, device=device)
    
    # Train model
    print("Training model...")
    start_time = time.time()
    
    history = trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        task=args.task,
        verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot training history
    if args.visualize:
        print("Plotting training history...")
        trainer.plot_training_history()
    
    # Evaluate model
    print("Evaluating model...")
    
    # Get a sequence of snapshots for evaluation
    eval_sequence = temporal_graph.snapshots[-args.sequence_length:]
    
    # Compute efficiency metrics
    efficiency_metrics = computational_efficiency_metrics(
        model, 
        temporal_graph, 
        eval_sequence, 
        device=device
    )
    
    print("Computational efficiency metrics:")
    print(f"  Total nodes processed: {efficiency_metrics['total_nodes']}")
    print(f"  Inference time: {efficiency_metrics['inference_time']:.4f} seconds")
    print(f"  Nodes per second: {efficiency_metrics['nodes_per_second']:.2f}")
    if device.type == 'cuda':
        print(f"  Peak memory usage: {efficiency_metrics['peak_memory_mb']:.2f} MB")
    
    # Visualize node embeddings
    if args.visualize:
        print("Visualizing node embeddings...")
        visualize_node_embeddings(model, temporal_graph)
    
    print("Done!")


if __name__ == "__main__":
    main()