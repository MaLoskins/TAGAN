import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
import heapq
from collections import defaultdict


class MemoryBank:
    """
    Repository for storing embeddings of inactive nodes.
    
    This class efficiently stores and retrieves node embeddings by node ID,
    implements a time-decay mechanism for long-stored embeddings,
    and handles pruning of rarely-accessed nodes.
    """
    
    def __init__(self, 
                decay_factor: float = 0.9, 
                max_size: int = 10000,
                pruning_threshold: int = 100):
        """
        Initialize a memory bank for storing node embeddings.
        
        Args:
            decay_factor: Factor to decay embeddings over time (0-1)
            max_size: Maximum number of nodes to store in memory
            pruning_threshold: Minimum number of nodes before pruning is triggered
        """
        self.node_embeddings = {}  # Maps node_id to (embedding, last_updated, access_count)
        self.decay_factor = decay_factor
        self.max_size = max_size
        self.pruning_threshold = pruning_threshold
        self.current_timestamp = 0
    
    def store_node(self, node_id: int, embedding: torch.Tensor, timestamp: int) -> None:
        """
        Store a node embedding in the memory bank.
        
        Args:
            node_id: ID of the node
            embedding: Node embedding tensor
            timestamp: Current timestamp
        """
        # Update current timestamp
        self.current_timestamp = max(self.current_timestamp, timestamp)
        
        # Check if node already exists
        if node_id in self.node_embeddings:
            _, last_updated, access_count = self.node_embeddings[node_id]
            self.node_embeddings[node_id] = (embedding.detach().clone(), timestamp, access_count + 1)
        else:
            self.node_embeddings[node_id] = (embedding.detach().clone(), timestamp, 1)
        
        # Check if pruning is needed
        if len(self.node_embeddings) > self.max_size:
            self.prune_memory_bank(timestamp)
    
    def retrieve_node(self, node_id: int, current_timestamp: int) -> Optional[torch.Tensor]:
        """
        Retrieve and potentially time-decay a node's embedding.
        
        Args:
            node_id: ID of the node
            current_timestamp: Current timestamp
            
        Returns:
            Decayed embedding or None if node doesn't exist
        """
        # Update current timestamp
        self.current_timestamp = max(self.current_timestamp, current_timestamp)
        
        if node_id not in self.node_embeddings:
            return None
        
        embedding, last_updated, access_count = self.node_embeddings[node_id]
        
        # Apply time decay based on time difference
        time_diff = current_timestamp - last_updated
        decay = self.decay_factor ** time_diff if time_diff > 0 else 1.0
        decayed_embedding = embedding * decay
        
        # Update access count
        self.node_embeddings[node_id] = (embedding, last_updated, access_count + 1)
        
        return decayed_embedding
    
    def prune_memory_bank(self, current_timestamp: int, max_size: Optional[int] = None) -> None:
        """
        Remove least important nodes from memory bank.
        
        Importance is determined by a combination of:
        - Recency (time since last update)
        - Frequency (access count)
        
        Args:
            current_timestamp: Current timestamp
            max_size: Maximum size to prune to (defaults to self.max_size)
        """
        if max_size is None:
            max_size = self.max_size
        
        # Only prune if we're above the pruning threshold
        if len(self.node_embeddings) <= self.pruning_threshold:
            return
        
        # Calculate importance scores for each node
        # Score = access_count / (time_since_last_update + 1)
        scores = []
        for node_id, (_, last_updated, access_count) in self.node_embeddings.items():
            time_since_update = current_timestamp - last_updated
            importance = access_count / (time_since_update + 1)
            scores.append((importance, node_id))
        
        # Sort by importance (ascending)
        scores.sort()
        
        # Keep only the top max_size nodes
        nodes_to_keep = set(node_id for _, node_id in scores[-max_size:])
        
        # Remove nodes not in the keep set
        self.node_embeddings = {
            node_id: data for node_id, data in self.node_embeddings.items()
            if node_id in nodes_to_keep
        }
    
    def get_all_nodes(self) -> List[int]:
        """
        Get all node IDs in the memory bank.
        
        Returns:
            List of node IDs
        """
        return list(self.node_embeddings.keys())
    
    def get_node_count(self) -> int:
        """
        Get the number of nodes in the memory bank.
        
        Returns:
            Number of nodes
        """
        return len(self.node_embeddings)
    
    def clear(self) -> None:
        """Clear all nodes from the memory bank."""
        self.node_embeddings.clear()
    
    def batch_retrieve_nodes(self, node_ids: List[int], current_timestamp: int) -> torch.Tensor:
        """
        Retrieve embeddings for multiple nodes at once.
        
        Args:
            node_ids: List of node IDs
            current_timestamp: Current timestamp
            
        Returns:
            Tensor of node embeddings, with zeros for missing nodes
        """
        if not node_ids:
            return torch.tensor([])
        
        # Get first embedding to determine dimension
        for node_id in self.node_embeddings:
            embedding_dim = self.node_embeddings[node_id][0].shape[0]
            break
        else:
            # If memory bank is empty, assume a default dimension
            embedding_dim = 64
        
        # Retrieve embeddings
        embeddings = []
        for node_id in node_ids:
            embedding = self.retrieve_node(node_id, current_timestamp)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                embeddings.append(torch.zeros(embedding_dim))
        
        return torch.stack(embeddings) if embeddings else torch.tensor([])
    
    def batch_store_nodes(self, node_ids: List[int], embeddings: torch.Tensor, timestamp: int) -> None:
        """
        Store embeddings for multiple nodes at once.
        
        Args:
            node_ids: List of node IDs
            embeddings: Tensor of node embeddings
            timestamp: Current timestamp
        """
        for i, node_id in enumerate(node_ids):
            self.store_node(node_id, embeddings[i], timestamp)


def propagate_between_snapshots(previous_snapshot, current_snapshot, memory_bank):
    """
    Transfer relevant node states between snapshots and memory.
    
    Args:
        previous_snapshot: Dictionary containing previous snapshot data
        current_snapshot: Dictionary containing current snapshot data
        memory_bank: MemoryBank instance
        
    Returns:
        Dictionary mapping current snapshot node indices to embeddings from memory
    """
    # Get active nodes from both snapshots
    prev_active = set(previous_snapshot['active_nodes'])
    curr_active = set(current_snapshot['active_nodes'])
    
    # Identify nodes that have become inactive
    newly_inactive = prev_active - curr_active
    
    # Identify nodes that have become active
    newly_active = curr_active - prev_active
    
    # Identify nodes that remain active
    still_active = prev_active.intersection(curr_active)
    
    # Map for retrieving embeddings from memory
    memory_embeddings = {}
    
    # Retrieve embeddings for newly active nodes from memory
    timestamp = current_snapshot['window_start']
    for node_id in newly_active:
        embedding = memory_bank.retrieve_node(node_id, timestamp)
        if embedding is not None:
            # Map to the index in the current snapshot
            curr_idx = current_snapshot['active_nodes'].index(node_id)
            memory_embeddings[curr_idx] = embedding
    
    return memory_embeddings


def handle_empty_snapshot(current_timestamp, memory_bank, sample_size=10):
    """
    Strategy for continuing model operation during inactive periods.
    
    Args:
        current_timestamp: Current timestamp
        memory_bank: MemoryBank instance
        sample_size: Number of nodes to sample from memory
        
    Returns:
        Dictionary containing a synthetic snapshot
    """
    # Get all nodes from memory
    all_nodes = memory_bank.get_all_nodes()
    
    if not all_nodes:
        # If memory is empty, return an empty snapshot
        return {
            'timestamp': current_timestamp,
            'active_nodes': [],
            'edges': [],
            'window_start': current_timestamp,
            'window_end': current_timestamp + 1
        }
    
    # Sample nodes from memory
    if len(all_nodes) <= sample_size:
        sampled_nodes = all_nodes
    else:
        sampled_nodes = np.random.choice(all_nodes, size=sample_size, replace=False).tolist()
    
    # Create a synthetic snapshot with no edges
    snapshot = {
        'timestamp': current_timestamp,
        'active_nodes': sampled_nodes,
        'edges': [],  # No edges in an empty snapshot
        'window_start': current_timestamp,
        'window_end': current_timestamp + 1
    }
    
    return snapshot


def initialize_new_node(node_features):
    """
    Create initial embedding for a previously unseen node.
    
    Args:
        node_features: Features for the new node
        
    Returns:
        Initial embedding for the node
    """
    # Convert features to tensor if not already
    if not isinstance(node_features, torch.Tensor):
        node_features = torch.FloatTensor(node_features)
    
    # Use features directly as initial embedding
    # This could be enhanced with more sophisticated initialization
    return node_features