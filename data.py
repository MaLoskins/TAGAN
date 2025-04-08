import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict
import scipy.sparse as sp


class TemporalGraph:
    """
    Container for temporal graph data with methods for snapshot creation.
    
    This class handles conversion from raw temporal interaction data to a sequence
    of snapshot graphs, supporting variable-sized adjacency matrices between snapshots.
    """
    
    def __init__(self, window_size: int = 15):
        """
        Initialize a temporal graph container.
        
        Args:
            window_size: Size of the temporal window in minutes
        """
        self.window_size = window_size  # in minutes
        self.snapshots = []
        self.node_id_map = {}  # Maps external node IDs to internal consecutive IDs
        self.reverse_node_id_map = {}  # Maps internal IDs back to external IDs
        self.feature_dim = None
        self.num_nodes = 0
        self.node_features = {}  # Maps node_id to features
        
    @classmethod
    def from_interactions(cls, 
                         interactions_df: pd.DataFrame,
                         time_column: str = 'timestamp',
                         source_column: str = 'source_id',
                         target_column: str = 'target_id',
                         features_columns: List[str] = None,
                         window_size: int = 15) -> 'TemporalGraph':
        """
        Create a temporal graph from a DataFrame of interactions.
        
        Args:
            interactions_df: DataFrame containing temporal interactions
            time_column: Name of the column containing timestamps
            source_column: Name of the column containing source node IDs
            target_column: Name of the column containing target node IDs
            features_columns: List of column names to use as node features
            window_size: Size of the temporal window in minutes
            
        Returns:
            A TemporalGraph instance
        """
        # Create a new temporal graph
        temporal_graph = cls(window_size=window_size)
        
        # Sort interactions by time
        interactions_df = interactions_df.sort_values(by=time_column)
        
        # Extract timestamps and convert to minutes since start
        timestamps = pd.to_datetime(interactions_df[time_column])
        start_time = timestamps.min()
        minutes_since_start = ((timestamps - start_time).dt.total_seconds() / 60).astype(int)
        
        # Create node ID mapping
        all_nodes = set(interactions_df[source_column]).union(set(interactions_df[target_column]))
        for i, node_id in enumerate(sorted(all_nodes)):
            temporal_graph.node_id_map[node_id] = i
            temporal_graph.reverse_node_id_map[i] = node_id
        
        temporal_graph.num_nodes = len(temporal_graph.node_id_map)
        
        # Extract node features if provided
        if features_columns:
            # Initialize feature dimension
            temporal_graph.feature_dim = len(features_columns)
            
            # Extract features for each node
            node_features = {}
            for node_id in all_nodes:
                # Get rows where this node appears as source or target
                source_rows = interactions_df[interactions_df[source_column] == node_id]
                target_rows = interactions_df[interactions_df[target_column] == node_id]
                
                # Combine and take the most recent feature values
                combined = pd.concat([source_rows, target_rows]).sort_values(by=time_column)
                if not combined.empty:
                    features = combined[features_columns].iloc[-1].values
                    node_features[temporal_graph.node_id_map[node_id]] = features
            
            temporal_graph.node_features = node_features
        else:
            # If no features provided, use one-hot encoding
            temporal_graph.feature_dim = temporal_graph.num_nodes
            for node_id in all_nodes:
                internal_id = temporal_graph.node_id_map[node_id]
                one_hot = np.zeros(temporal_graph.num_nodes)
                one_hot[internal_id] = 1.0
                temporal_graph.node_features[internal_id] = one_hot
        
        # Create snapshots
        max_minutes = minutes_since_start.max()
        num_snapshots = (max_minutes // window_size) + 1
        
        for snapshot_idx in range(num_snapshots):
            start_minute = snapshot_idx * window_size
            end_minute = (snapshot_idx + 1) * window_size
            
            # Get interactions in this time window
            window_mask = (minutes_since_start >= start_minute) & (minutes_since_start < end_minute)
            window_interactions = interactions_df[window_mask]
            
            # Create snapshot if there are interactions in this window
            if not window_interactions.empty:
                # Get active nodes in this window
                active_sources = set(window_interactions[source_column])
                active_targets = set(window_interactions[target_column])
                active_nodes = active_sources.union(active_targets)
                
                # Map to internal IDs
                active_node_ids = [temporal_graph.node_id_map[node] for node in active_nodes]
                
                # Create adjacency matrix for this snapshot
                edges = []
                for _, row in window_interactions.iterrows():
                    source_id = temporal_graph.node_id_map[row[source_column]]
                    target_id = temporal_graph.node_id_map[row[target_column]]
                    edges.append((source_id, target_id))
                
                # Create snapshot
                snapshot = {
                    'timestamp': start_time + pd.Timedelta(minutes=start_minute),
                    'active_nodes': active_node_ids,
                    'edges': edges,
                    'window_start': start_minute,
                    'window_end': end_minute
                }
                
                temporal_graph.snapshots.append(snapshot)
        
        return temporal_graph
    
    def get_snapshot_sequence(self, start_time, end_time):
        """
        Get a sequence of snapshots between the specified times.
        
        Args:
            start_time: Start time for the sequence
            end_time: End time for the sequence
            
        Returns:
            List of snapshots between start_time and end_time
        """
        # Convert times to minutes since start if they're datetime objects
        if isinstance(start_time, pd.Timestamp):
            start_time = int((start_time - self.snapshots[0]['timestamp']).total_seconds() / 60)
        if isinstance(end_time, pd.Timestamp):
            end_time = int((end_time - self.snapshots[0]['timestamp']).total_seconds() / 60)
        
        # Find snapshots in the time range
        sequence = []
        for snapshot in self.snapshots:
            if snapshot['window_end'] > start_time and snapshot['window_start'] < end_time:
                sequence.append(snapshot)
        
        return sequence
    
    def get_node_features(self, node_ids):
        """
        Get features for the specified nodes.
        
        Args:
            node_ids: List of node IDs
            
        Returns:
            Tensor of node features
        """
        features = []
        for node_id in node_ids:
            if node_id in self.node_features:
                features.append(self.node_features[node_id])
            else:
                # Use zeros for nodes without features
                features.append(np.zeros(self.feature_dim))
        
        return torch.FloatTensor(np.array(features))
    
    def create_adjacency_matrix(self, snapshot):
        """
        Create an adjacency matrix for the given snapshot.
        
        Args:
            snapshot: A snapshot dictionary
            
        Returns:
            Sparse adjacency matrix
        """
        active_nodes = snapshot['active_nodes']
        node_to_idx = {node: i for i, node in enumerate(active_nodes)}
        
        # Create sparse adjacency matrix
        rows, cols = [], []
        for source, target in snapshot['edges']:
            if source in node_to_idx and target in node_to_idx:
                rows.append(node_to_idx[source])
                cols.append(node_to_idx[target])
        
        n = len(active_nodes)
        if not rows:  # Handle empty edge list
            return sp.csr_matrix((n, n))
        
        data = np.ones(len(rows))
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        
        return adj
    
    def create_symmetric_mask(self, adjacency_matrix):
        """
        Convert potentially asymmetric adjacency matrix to a symmetric masked version
        that preserves directional information for attention computation.
        
        Args:
            adjacency_matrix: Sparse adjacency matrix
            
        Returns:
            Symmetric mask tensor
        """
        # Make symmetric by taking max of A and A^T
        adj_t = adjacency_matrix.transpose()
        symmetric_adj = adjacency_matrix.maximum(adj_t)
        
        # Convert to dense tensor
        mask = torch.FloatTensor(symmetric_adj.todense())
        
        # Create mask where 1 indicates connection exists in either direction
        mask = (mask > 0).float()
        
        return mask


def create_temporal_batches(temporal_graph, batch_size, sequence_length):
    """
    Create training batches that preserve temporal dependencies.
    
    Args:
        temporal_graph: TemporalGraph instance
        batch_size: Number of sequences per batch
        sequence_length: Number of snapshots per sequence
        
    Returns:
        List of batches, where each batch contains sequences of snapshot graphs
    """
    snapshots = temporal_graph.snapshots
    num_snapshots = len(snapshots)
    
    # Adjust sequence_length if there are not enough snapshots
    if num_snapshots < sequence_length:
        print(f"Warning: Adjusting sequence_length from {sequence_length} to {num_snapshots} due to insufficient snapshots")
        sequence_length = max(1, num_snapshots)
    
    # Create sequences
    sequences = []
    for i in range(0, num_snapshots - sequence_length + 1):
        sequences.append(snapshots[i:i+sequence_length])
    
    # If no sequences were created but we have snapshots, create at least one sequence
    if not sequences and num_snapshots > 0:
        sequences.append(snapshots[:num_snapshots])
    
    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        if batch:  # Ensure batch is not empty
            batches.append(batch)
    
    # Ensure we have at least one batch if we have sequences
    if not batches and sequences:
        batches.append(sequences)
    
    return batches