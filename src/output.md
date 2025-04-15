# CODEBASE

## Directory Tree:

### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src

```
C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src
├── __init__.py
├── metrics/
│   ├── __init__.py
│   └── metrics.py
├── model/
│   ├── __init__.py
│   ├── memory.py
│   └── model.py
├── training/
│   ├── __init__.py
│   └── trainer.py
└── utils/
    ├── __init__.py
    └── utils.py
```

## Code Files


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\__init__.py

```
# Corrected version in src/__init__.py using relative imports
from .model import *
from .data import *       # Uncomment or fix this import if a 'data' subpackage exists
from .utils import *
from .training import *
from .metrics import *


__all__ = []
__all__.extend(src.model.__all__)
__all__.extend(src.data.__all__)
__all__.extend(src.utils.__all__)
__all__.extend(src.training.__all__)
__all__.extend(src.metrics.__all__)
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\metrics\__init__.py

```
from src.metrics.metrics import (
    temporal_node_classification_metrics,
    temporal_link_prediction_metrics,
    computational_efficiency_metrics,
    compare_with_full_graph_gat,
    evaluate_long_term_dependencies,
    evaluate_scalability
)

__all__ = [
    'temporal_node_classification_metrics',
    'temporal_link_prediction_metrics',
    'computational_efficiency_metrics',
    'compare_with_full_graph_gat',
    'evaluate_long_term_dependencies',
    'evaluate_scalability'
]
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\metrics\metrics.py

```
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import time
from src.utils.utils import logger


def temporal_node_classification_metrics(predictions: List[torch.Tensor], 
                                        labels: List[torch.Tensor], 
                                        masks: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute metrics for temporal node classification.
    
    Args:
        predictions: List of prediction tensors for each snapshot
        labels: List of label tensors for each snapshot
        masks: List of mask tensors indicating which nodes to evaluate
        
    Returns:
        Dictionary of metrics
    """
    all_preds = []
    all_labels = []
    
    # Collect predictions and labels across all snapshots
    for i in range(len(predictions)):
        preds = predictions[i]
        snapshot_labels = labels[i]
        mask = masks[i]
        
        # Apply mask
        masked_preds = preds[mask]
        masked_labels = snapshot_labels[mask]
        
        # Convert to numpy
        if isinstance(masked_preds, torch.Tensor):
            masked_preds = masked_preds.detach().cpu().numpy()
        if isinstance(masked_labels, torch.Tensor):
            masked_labels = masked_labels.detach().cpu().numpy()
        
        all_preds.append(masked_preds)
        all_labels.append(masked_labels)
    
    # Concatenate predictions and labels
    if all_preds and all_labels:
        try:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            
            # Check if shapes match
            if all_preds.shape != all_labels.shape:
                print(f"Warning: Prediction shape {all_preds.shape} doesn't match label shape {all_labels.shape}")
                # Try to reshape if possible
                if all_preds.size == all_labels.size:
                    all_preds = all_preds.reshape(all_labels.shape)
            
            # Ensure types are compatible
            if all_preds.dtype != all_labels.dtype:
                print(f"Converting predictions from {all_preds.dtype} to {all_labels.dtype}")
                all_preds = all_preds.astype(all_labels.dtype)
            
            # Compute metrics
            try:
                accuracy = accuracy_score(all_labels, all_preds)
            except Exception as e:
                print(f"Error computing accuracy: {e}")
                accuracy = 0.0
                
            try:
                macro_f1 = f1_score(all_labels, all_preds, average='macro')
            except Exception as e:
                print(f"Error computing macro F1: {e}")
                macro_f1 = 0.0
                
            try:
                micro_f1 = f1_score(all_labels, all_preds, average='micro')
            except Exception as e:
                print(f"Error computing micro F1: {e}")
                micro_f1 = 0.0
            
            return {
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1
            }
        except Exception as e:
            print(f"Error in temporal_node_classification_metrics: {e}")
            return {
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'micro_f1': 0.0
            }
    else:
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0
        }


def temporal_link_prediction_metrics(edge_scores: List[torch.Tensor], 
                                    true_edges: List[torch.Tensor], 
                                    negative_edges: List[torch.Tensor]) -> Dict[str, float]:
    """
    Compute metrics for temporal link prediction.
    
    Args:
        edge_scores: List of edge score tensors for each snapshot
        true_edges: List of true edge tensors for each snapshot
        negative_edges: List of negative edge tensors for each snapshot
        
    Returns:
        Dictionary of metrics
    """
    all_scores = []
    all_labels = []
    
    # Collect scores and labels across all snapshots
    for i in range(len(edge_scores)):
        scores = edge_scores[i]
        pos_edges = true_edges[i]
        neg_edges = negative_edges[i]
        
        # Convert to numpy
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        
        # Create labels (1 for positive, 0 for negative)
        labels = np.zeros(len(pos_edges) + len(neg_edges))
        labels[:len(pos_edges)] = 1.0
        
        all_scores.append(scores)
        all_labels.append(labels)
    
    # Concatenate scores and labels
    if all_scores and all_labels:
        try:
            all_scores = np.concatenate(all_scores)
            all_labels = np.concatenate(all_labels)
            
            # Check if we have enough data for meaningful metrics
            if len(np.unique(all_labels)) < 2:
                print("Warning: Not enough class diversity for AUC/AP calculation")
                return {
                    'auc': 0.0,
                    'ap': 0.0
                }
            
            # Compute metrics
            try:
                auc = roc_auc_score(all_labels, all_scores)
            except Exception as e:
                print(f"Error computing AUC: {e}")
                auc = 0.0
                
            try:
                ap = average_precision_score(all_labels, all_scores)
            except Exception as e:
                print(f"Error computing AP: {e}")
                ap = 0.0
            
            return {
                'auc': auc,
                'ap': ap
            }
        except Exception as e:
            print(f"Error in temporal_link_prediction_metrics: {e}")
            return {
                'auc': 0.0,
                'ap': 0.0
            }
    else:
        return {
            'auc': 0.0,
            'ap': 0.0
        }


def computational_efficiency_metrics(model, temporal_graph, snapshot_sequence, 
                                    device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    Compute computational efficiency metrics.
    
    Args:
        model: TempGAT model
        temporal_graph: TemporalGraph instance
        snapshot_sequence: List of snapshot dictionaries
        device: Device to use for computation
        
    Returns:
        Dictionary of metrics
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Count total number of nodes processed
    total_nodes = sum(len(snapshot['active_nodes']) for snapshot in snapshot_sequence)
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        _ = model(temporal_graph, snapshot_sequence)
    
    inference_time = time.time() - start_time
    
    # Compute nodes per second
    nodes_per_second = total_nodes / inference_time if inference_time > 0 else 0
    
    # Compute memory usage
    if device == 'cuda':
        # Get peak memory usage in MB
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak_memory = 0  # Not available for CPU
    
    return {
        'total_nodes': total_nodes,
        'inference_time': inference_time,
        'nodes_per_second': nodes_per_second,
        'peak_memory_mb': peak_memory
    }


def compare_with_full_graph_gat(tempgat_model, full_gat_model, temporal_graph, snapshot_sequence, 
                               device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, float]:
    """
    Compare TempGAT with full-graph GAT in terms of computational efficiency.
    
    Args:
        tempgat_model: TempGAT model
        full_gat_model: Full-graph GAT model
        temporal_graph: TemporalGraph instance
        snapshot_sequence: List of snapshot dictionaries
        device: Device to use for computation
        
    Returns:
        Dictionary of comparison metrics
    """
    # Measure TempGAT efficiency
    tempgat_metrics = computational_efficiency_metrics(
        tempgat_model, 
        temporal_graph, 
        snapshot_sequence, 
        device
    )
    
    # Create a full graph from all snapshots
    all_nodes = set()
    all_edges = []
    
    for snapshot in snapshot_sequence:
        all_nodes.update(snapshot['active_nodes'])
        all_edges.extend(snapshot['edges'])
    
    # Create a synthetic full graph snapshot
    full_graph_snapshot = {
        'timestamp': snapshot_sequence[0]['timestamp'],
        'active_nodes': list(all_nodes),
        'edges': all_edges,
        'window_start': snapshot_sequence[0]['window_start'],
        'window_end': snapshot_sequence[-1]['window_end']
    }
    
    # Measure full GAT efficiency
    start_time = time.time()
    
    with torch.no_grad():
        # Process with full GAT (this is a placeholder, actual implementation would depend on full_gat_model)
        # For a fair comparison, we would need to implement a standard GAT model
        pass
    
    full_gat_time = time.time() - start_time
    
    # Compute speedup
    speedup = full_gat_time / tempgat_metrics['inference_time'] if tempgat_metrics['inference_time'] > 0 else 0
    
    return {
        'tempgat_nodes_per_second': tempgat_metrics['nodes_per_second'],
        'tempgat_memory_mb': tempgat_metrics['peak_memory_mb'],
        'speedup_factor': speedup
    }


def evaluate_long_term_dependencies(model, temporal_graph, snapshot_sequences, 
                                   time_gaps=[1, 5, 10, 20], 
                                   task='node_classification',
                                   device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, List[float]]:
    """
    Evaluate how well the model captures long-term dependencies.
    
    Args:
        model: TempGAT model
        temporal_graph: TemporalGraph instance
        snapshot_sequences: List of snapshot sequences for evaluation
        time_gaps: List of time gaps to evaluate
        task: Task to evaluate ('node_classification' or 'link_prediction')
        device: Device to use for computation
        
    Returns:
        Dictionary of metrics for each time gap
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    results = {
        'time_gaps': time_gaps,
        'metrics': []
    }
    
    for gap in time_gaps:
        gap_metrics = []
        
        for sequence in snapshot_sequences:
            # Split sequence into history and future
            history = sequence[:-gap]
            future = sequence[-gap:]
            
            if not history or not future:
                continue
            
            # Process history to build memory
            with torch.no_grad():
                _ = model(temporal_graph, history)
            
            # Evaluate on future
            if task == 'node_classification':
                # Get predictions for future snapshots
                future_preds = []
                future_labels = []
                future_masks = []
                
                for snapshot in future:
                    # Get active nodes
                    active_nodes = snapshot['active_nodes']
                    
                    # Get predictions
                    with torch.no_grad():
                        preds = model.predict(temporal_graph, [snapshot], task='node_classification')
                    
                    # Get labels (placeholder - replace with actual labels)
                    labels = torch.zeros(len(active_nodes), dtype=torch.long).to(device)
                    
                    # Create mask for nodes with labels
                    mask = torch.ones(len(active_nodes), dtype=torch.bool).to(device)
                    
                    future_preds.append(preds)
                    future_labels.append(labels)
                    future_masks.append(mask)
                
                # Check if we have valid predictions and labels
                if future_preds and future_labels and future_masks:
                    try:
                        # Compute metrics
                        metrics = temporal_node_classification_metrics(
                            future_preds,
                            future_labels,
                            future_masks
                        )
                    except Exception as e:
                        print(f"Error computing node classification metrics: {e}")
                        metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0}
                else:
                    metrics = {'accuracy': 0.0, 'macro_f1': 0.0, 'micro_f1': 0.0}
                
            elif task == 'link_prediction':
                # Get predictions for future snapshots
                future_scores = []
                future_true_edges = []
                future_neg_edges = []
                
                for snapshot in future:
                    try:
                        # Get true edges
                        true_edges = torch.tensor(snapshot['edges'], dtype=torch.long).to(device)
                        
                        # Generate negative samples (placeholder - replace with actual negative sampling)
                        neg_edges = torch.zeros_like(true_edges).to(device)
                        
                        # Get predictions
                        with torch.no_grad():
                            scores = model.predict(temporal_graph, [snapshot], task='link_prediction')
                            
                        future_scores.append(scores)
                        future_true_edges.append(true_edges)
                        future_neg_edges.append(neg_edges)
                    except Exception as e:
                        print(f"Error processing snapshot for link prediction: {e}")
                    
                    future_scores.append(scores)
                    future_true_edges.append(true_edges)
                    future_neg_edges.append(neg_edges)
                
                # Compute metrics
                metrics = temporal_link_prediction_metrics(
                    future_scores, 
                    future_true_edges, 
                    future_neg_edges
                )
            
            gap_metrics.append(metrics)
        
        # Average metrics for this gap
        avg_metrics = {}
        if gap_metrics:
            for key in gap_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in gap_metrics) / len(gap_metrics)
        
        results['metrics'].append(avg_metrics)
    
    return results


def evaluate_scalability(model, temporal_graph, snapshot_sequences, 
                        node_counts=[100, 1000, 10000, 100000],
                        device='cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, List[float]]:
    """
    Evaluate model scalability with increasing graph sizes.
    
    Args:
        model: TempGAT model
        temporal_graph: TemporalGraph instance
        snapshot_sequences: List of snapshot sequences for evaluation
        node_counts: List of node counts to evaluate
        device: Device to use for computation
        
    Returns:
        Dictionary of metrics for each node count
    """
    # Move model to device
    model = model.to(device)
    model.eval()
    
    results = {
        'node_counts': node_counts,
        'inference_times': [],
        'memory_usage': []
    }
    
    for count in node_counts:
        # Create synthetic snapshots with specified node count
        synthetic_snapshots = []
        
        for i in range(5):  # Create 5 snapshots
            # Sample nodes
            if count <= len(temporal_graph.node_id_map):
                nodes = np.random.choice(
                    list(temporal_graph.node_id_map.values()), 
                    size=count, 
                    replace=False
                ).tolist()
            else:
                # If requested count is larger than available nodes, use all nodes
                nodes = list(temporal_graph.node_id_map.values())
            
            # Create synthetic edges (random connections)
            edges = []
            for _ in range(count * 5):  # 5 edges per node on average
                source = np.random.choice(nodes)
                target = np.random.choice(nodes)
                if source != target:
                    edges.append((source, target))
            
            # Create snapshot
            snapshot = {
                'timestamp': i,
                'active_nodes': nodes,
                'edges': edges,
                'window_start': i,
                'window_end': i + 1
            }
            
            synthetic_snapshots.append(snapshot)
        
        # Measure inference time and memory usage
        start_time = time.time()
        
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(temporal_graph, synthetic_snapshots)
        
        inference_time = time.time() - start_time
        
        # Get memory usage
        if device == 'cuda':
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_memory = 0  # Not available for CPU
        
        results['inference_times'].append(inference_time)
        results['memory_usage'].append(peak_memory)
    
    return results
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\model\__init__.py

```
from src.model.model import TempGAT, SnapshotGAT, TemporalAttention, GraphAttentionLayer
from src.model.memory import MemoryBank, propagate_between_snapshots, handle_empty_snapshot, initialize_new_node

__all__ = [
    'TempGAT',
    'SnapshotGAT',
    'TemporalAttention',
    'GraphAttentionLayer',
    'MemoryBank',
    'propagate_between_snapshots',
    'handle_empty_snapshot',
    'initialize_new_node'
]
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\model\memory.py

```
import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
import heapq
from collections import defaultdict
import inspect
from src.utils.utils import manage_memory, logger


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
        # Set pruning percentage - remove this percentage of nodes when max_size is reached
        self.pruning_percentage = 0.3  # Remove 30% of nodes when pruning
        # Set buffer percentage - how much over max_size before pruning
        self.buffer_percentage = 0.03  # Prune when 3% over max_size
    
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
        # Log memory bank size less frequently to reduce output
        if len(self.node_embeddings) % 5000 == 0:
            print(f"[MemoryBank] Size: {len(self.node_embeddings)} nodes")
        
        # Ensure embedding is fully detached from computation graph and on CPU
        # This helps prevent memory leaks by ensuring the tensor doesn't retain connections
        # to the computation graph that created it
        cpu_embedding = embedding.detach().cpu().clone()
        
        # Check if node already exists
        if node_id in self.node_embeddings:
            # Delete old embedding explicitly to help with memory management
            old_embedding, last_updated, access_count = self.node_embeddings[node_id]
            del old_embedding
            self.node_embeddings[node_id] = (cpu_embedding, timestamp, access_count + 1)
        else:
            self.node_embeddings[node_id] = (cpu_embedding, timestamp, 1)
        
        # Check if pruning is needed - use a smaller buffer to be more aggressive
        if len(self.node_embeddings) > self.max_size * (1 + self.buffer_percentage):
            # Only log when pruning large memory banks
            if self.max_size > 5000:
                print(f"[MemoryBank] Pruning at {len(self.node_embeddings)} nodes")
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
        
        # Move to the device of the caller if needed
        # This is determined by checking if we're in a PyTorch module's forward pass
        device = None
        for frame in inspect.stack():
            if 'self' in frame[0].f_locals:
                obj = frame[0].f_locals['self']
                if isinstance(obj, torch.nn.Module) and hasattr(obj, 'parameters'):
                    try:
                        # Get device from the first parameter
                        for param in obj.parameters():
                            device = param.device
                            break
                    except:
                        pass
            if device is not None:
                break
        
        # If we found a device, move the embedding to it
        if device is not None:
            embedding = embedding.to(device)
        
        # Apply decay
        decayed_embedding = embedding * decay
        
        # Update access count
        self.node_embeddings[node_id] = (embedding.cpu() if device is not None else embedding,
                                        last_updated, access_count + 1)
        
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
        
        # Calculate target size based on pruning_percentage
        # This prevents constant pruning of just a few nodes at a time
        target_size = int(max_size * (1 - self.pruning_percentage))
        
        # Keep only the top target_size nodes
        nodes_to_keep = set(node_id for _, node_id in scores[-target_size:])
        
        # Remove nodes not in the keep set
        before_size = len(self.node_embeddings)
        
        # Explicitly delete embeddings to help with memory management
        for node_id in list(self.node_embeddings.keys()):
            if node_id not in nodes_to_keep:
                # Delete the embedding tensor explicitly
                embedding, _, _ = self.node_embeddings[node_id]
                del embedding
                # Remove from dictionary
                del self.node_embeddings[node_id]
        
        # Only log for large memory banks
        if self.max_size > 5000:
            logger.info(f"Pruned: {before_size} -> {len(self.node_embeddings)} nodes")
            
        # Use centralized memory management
        manage_memory(force_gc=True, log_memory=self.max_size > 5000)
    
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
        
        # Determine target device
        device = None
        for frame in inspect.stack():
            if 'self' in frame[0].f_locals:
                obj = frame[0].f_locals['self']
                if isinstance(obj, torch.nn.Module) and hasattr(obj, 'parameters'):
                    try:
                        # Get device from the first parameter
                        for param in obj.parameters():
                            device = param.device
                            break
                    except:
                        pass
            if device is not None:
                break
        
        # Retrieve embeddings
        embeddings = []
        for node_id in node_ids:
            embedding = self.retrieve_node(node_id, current_timestamp)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Create zeros tensor on the appropriate device
                zeros = torch.zeros(embedding_dim)
                if device is not None:
                    zeros = zeros.to(device)
                embeddings.append(zeros)
        
        # Stack embeddings and ensure they're on the right device
        if embeddings:
            stacked = torch.stack(embeddings)
            if device is not None:
                stacked = stacked.to(device)
            return stacked
        else:
            empty = torch.tensor([])
            if device is not None:
                empty = empty.to(device)
            return empty
    
    def batch_store_nodes(self, node_ids: List[int], embeddings: torch.Tensor, timestamp: int) -> None:
        """
        Store embeddings for multiple nodes at once.
        
        Args:
            node_ids: List of node IDs
            embeddings: Tensor of node embeddings
            timestamp: Current timestamp
        """
        # Process in batches to reduce memory pressure
        batch_size = 32  # Process 32 nodes at a time
        
        for batch_start in range(0, len(node_ids), batch_size):
            batch_end = min(batch_start + batch_size, len(node_ids))
            batch_node_ids = node_ids[batch_start:batch_end]
            batch_embeddings = embeddings[batch_start:batch_end]
            
            # Store each node in the batch
            for i, node_id in enumerate(batch_node_ids):
                self.store_node(node_id, batch_embeddings[i], timestamp)
                
            # Use centralized memory management every 4 batches
            if batch_end % 128 == 0:  # Every 4 batches
                manage_memory(force_gc=True, log_memory=False)


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
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\model\model.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
import scipy.sparse as sp

from src.model.memory import MemoryBank, propagate_between_snapshots, handle_empty_snapshot, initialize_new_node
from src.utils.utils import create_symmetric_mask, masked_attention, sparse_to_torch_sparse, manage_memory, validate_snapshot, logger


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) implementation.
    """
    
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                dropout: float = 0.6, 
                alpha: float = 0.2, 
                concat: bool = True):
        """
        Initialize a graph attention layer.
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            concat: Whether to concatenate or average multi-head attention outputs
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention parameters
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, 
               input: torch.Tensor, 
               adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GAT layer.
        
        Args:
            input: Input features [N, in_features]
            adj: Adjacency matrix [N, N]
            
        Returns:
            Output features [N, out_features]
        """
        # Linear transformation
        h = torch.mm(input, self.W)  # [N, out_features]
        N = h.size(0)
        
        # Prepare for attention
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                            h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        
        # Compute attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, h)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class SnapshotGAT(nn.Module):
    """
    GAT implementation for individual snapshots.
    
    Modified GAT that handles variable-sized inputs through masking
    and maintains multi-head attention mechanism from original GAT.
    """
    
    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                output_dim: int, 
                num_heads: int = 8, 
                dropout: float = 0.6, 
                alpha: float = 0.2):
        """
        Initialize a snapshot GAT model.
        
        Args:
            input_dim: Size of input features
            hidden_dim: Size of hidden features
            output_dim: Size of output features
            num_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
        """
        super(SnapshotGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # First layer with multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                input_dim, 
                hidden_dim, 
                dropout=dropout, 
                alpha=alpha, 
                concat=True
            ) for _ in range(num_heads)
        ])
        
        # Output layer
        self.out_att = GraphAttentionLayer(
            hidden_dim * num_heads, 
            output_dim, 
            dropout=dropout, 
            alpha=alpha, 
            concat=False
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
               features: torch.Tensor, 
               adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the snapshot GAT.
        
        Args:
            features: Node features [N, input_dim]
            adj: Adjacency matrix [N, N]
            
        Returns:
            Node embeddings [N, output_dim]
        """
        # Apply dropout to input features
        x = self.dropout(features)
        
        # Apply first layer with multiple attention heads
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.dropout(x)
        
        # Apply output layer
        x = self.out_att(x, adj)
        
        return x
    
    def masked_forward(self,
                       features: torch.Tensor,
                       adj: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masking for asymmetric propagation.
        
        Args:
            features: Node features [N, input_dim]
            adj: Adjacency matrix [N, N]
            mask: Attention mask [N, N]
            
        Returns:
            Node embeddings [N, output_dim]
        """
        # Log input sizes
        N = features.size(0)
        if N > 1000:  # Only log for large inputs
            logger.info(f"Processing snapshot with {N} nodes")
            manage_memory(log_memory=True)
        
        # Apply dropout to input features
        x = self.dropout(features)
        
        # Apply first layer with multiple attention heads and masking
        outputs = []
        for att in self.attentions:
            # Use masked attention for each head
            h = torch.mm(x, att.W)  # [N, hidden_dim]
            N = h.size(0)
            
            # Log memory usage before large tensor operations
            if N > 1000:
                logger.debug("Before attention input creation")
                manage_memory(log_memory=True)
                
            # Prepare for attention
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
                                h.repeat(N, 1)], dim=1).view(N, N, 2 * att.out_features)
            
            if N > 1000:
                logger.debug(f"After attention input creation (size {a_input.shape})")
                manage_memory(log_memory=True)
            
            # Compute attention coefficients
            e = att.leakyrelu(torch.matmul(a_input, att.a).squeeze(2))
            
            # Apply mask
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(mask > 0, e, zero_vec)
            
            # Apply softmax to get attention weights
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, att.dropout, training=self.training)
            
            # Apply attention to features
            h_prime = torch.matmul(attention, h)
            
            if N > 1000:
                logger.debug("After attention computation")
                manage_memory(log_memory=True)
                
            outputs.append(F.elu(h_prime))
        
        # Concatenate outputs from all heads
        x = torch.cat(outputs, dim=1)
        x = self.dropout(x)
        
        # Apply output layer with masking
        h = torch.mm(x, self.out_att.W)  # [N, output_dim]
        N = h.size(0)
        
        # Prepare for attention
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                            h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_att.out_features)
        
        # Compute attention coefficients
        e = self.out_att.leakyrelu(torch.matmul(a_input, self.out_att.a).squeeze(2))
        
        # Apply mask
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(mask > 0, e, zero_vec)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.out_att.dropout, training=self.training)
        
        # Apply attention to features
        x = torch.matmul(attention, h)
        
        return x


class TemporalAttention(nn.Module):
    """
    Temporal attention layer for attending to node states across time.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.6, alpha=0.2):
        """
        Initialize temporal attention layer.
        
        Args:
            hidden_dim: Dimension of hidden features
            num_heads: Number of attention heads
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
        """
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        
        # Multi-head attention parameters
        self.W = nn.Parameter(torch.zeros(size=(num_heads, hidden_dim, hidden_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention parameters
        self.a = nn.Parameter(torch.zeros(size=(num_heads, 2 * hidden_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, sequence_embeddings):
        """
        Apply temporal attention to sequence of node embeddings.
        
        Args:
            sequence_embeddings: List of node embedding tensors [T, N, F]
                where T is sequence length, N is number of nodes, F is feature dimension
                
        Returns:
            Temporally attended node embeddings
        """
        # Stack embeddings along time dimension
        if not sequence_embeddings:
            return None
            
        # Get dimensions
        seq_len = len(sequence_embeddings)
        if seq_len <= 1:
            return sequence_embeddings[-1] if seq_len > 0 else None
            
        # Process each node separately
        device = sequence_embeddings[0].device
        node_counts = [emb.size(0) for emb in sequence_embeddings]
        max_nodes = max(node_counts)
        feature_dim = sequence_embeddings[0].size(1)
        
        # Only log for very large node counts to reduce output
        if max_nodes > 5000:
            logger.info(f"Processing {max_nodes} nodes in temporal attention")
            manage_memory(log_memory=True)
        
        # Create padded tensor for all embeddings
        padded_embeddings = torch.zeros(seq_len, max_nodes, feature_dim).to(device)
        masks = torch.zeros(seq_len, max_nodes, dtype=torch.bool).to(device)
        
        # Fill padded tensor with embeddings and create masks
        for t, embeddings in enumerate(sequence_embeddings):
            num_nodes = embeddings.size(0)
            padded_embeddings[t, :num_nodes] = embeddings
            masks[t, :num_nodes] = True
        
        # Memory-optimized attention implementation
        outputs = []
        for head in range(self.num_heads):
            # Transform embeddings
            transformed = torch.matmul(padded_embeddings, self.W[head])  # [T, N, F]
            
            # Process in smaller chunks to reduce memory usage
            # Use even smaller chunks for very large graphs
            if max_nodes > 5000:
                chunk_size = min(500, max(1, 3000 // max_nodes))
            else:
                chunk_size = min(1000, max(1, 5000 // max_nodes))
            
            # Initialize output tensor for this head on CPU to save GPU memory
            head_output = torch.zeros(max_nodes, self.hidden_dim, device='cpu')
            
            # Process nodes in chunks
            for chunk_start in range(0, max_nodes, chunk_size):
                chunk_end = min(chunk_start + chunk_size, max_nodes)
                
                # Only process nodes that exist in the last time step
                if not masks[-1, chunk_start:chunk_end].any():
                    continue
                
                # Compute attention for this chunk of nodes - use CPU for intermediate results
                chunk_attention = torch.zeros(seq_len, chunk_end - chunk_start, self.hidden_dim, device='cpu')
                
                for t1 in range(seq_len):
                    # Skip if no nodes in this time step
                    if not masks[t1, chunk_start:chunk_end].any():
                        continue
                        
                    # Compute attention weights between this time step and all others
                    # Use CPU for large intermediate tensors
                    weights = torch.zeros(seq_len, chunk_end - chunk_start, max_nodes, device='cpu')
                    
                    for t2 in range(seq_len):
                        # Skip if no nodes in this time step
                        if not masks[t2].any():
                            continue
                            
                        # Move only the necessary tensors to GPU for computation
                        query = transformed[t1, chunk_start:chunk_end].to(device)  # [chunk_size, F]
                        key = transformed[t2].to(device)  # [N, F]
                        
                        # Compute dot product attention
                        scores = torch.matmul(query, key.transpose(0, 1))  # [chunk_size, N]
                        
                        # Apply mask
                        mask_t1 = masks[t1, chunk_start:chunk_end].unsqueeze(1)  # [chunk_size, 1]
                        mask_t2 = masks[t2].unsqueeze(0)  # [1, N]
                        combined_mask = mask_t1 & mask_t2  # [chunk_size, N]
                        
                        scores = scores.masked_fill(~combined_mask, -9e15)
                        
                        # Apply softmax and move back to CPU
                        weights[t2] = F.softmax(scores, dim=1).cpu()
                        
                        # Free GPU memory immediately
                        del query, key, scores
                        if max_nodes > 5000:
                            manage_memory(force_gc=False)
                
                # Apply attention weights to values
                for t2 in range(seq_len):
                    if not masks[t2].any():
                        continue
                    # Move to device, compute, then back to CPU
                    w = weights[t2].to(device)
                    t = transformed[t2].to(device)
                    result = torch.matmul(w, t).cpu()
                    chunk_attention[t1] += result
                    del w, t, result
                
                # Apply time-based attention (simple average for now)
                # This is a simplified version of the full temporal attention
                valid_time_steps = masks[:, chunk_start:chunk_end].sum(dim=0) > 0
                chunk_result = chunk_attention.mean(dim=0)
                
                # Store result for this chunk
                head_output[chunk_start:chunk_end] = chunk_result
                
                # Use centralized memory management after each chunk
                manage_memory(force_gc=True)
                
                # Clear large tensors explicitly
                del chunk_attention, weights
            
            # Apply activation and add to outputs - move to device only at the end
            # This ensures we only have one copy in GPU memory
            device_output = F.elu(head_output.to(device))
            outputs.append(device_output.cpu())  # Store on CPU to save GPU memory
            del device_output
        
        # Combine outputs from all heads
        if outputs:
            # Move to device only for the final computation
            device_outputs = [out.to(device) for out in outputs]
            
            # Stack and average across heads
            all_head_outputs = torch.stack(device_outputs)  # [num_heads, N, F]
            combined_output = all_head_outputs.mean(dim=0)  # [N, F]
            
            # Only keep embeddings for nodes that exist in the last time step
            last_mask = masks[-1]
            final_output = combined_output[last_mask]
            
            # Clean up
            del device_outputs, all_head_outputs, combined_output
            
            return final_output
        else:
            # Fallback if no outputs were generated
            return sequence_embeddings[-1]


class TempGAT(nn.Module):
    """
    Main model architecture for Temporal Graph Attention Network.
    
    Composed of SnapshotGAT, MemoryBank, temporal attention, and temporal propagation logic.
    """
    
    def __init__(self,
                input_dim: int,
                hidden_dim: int,
                output_dim: int,
                num_heads: int = 8,
                memory_decay_factor: float = 0.9,
                dropout: float = 0.6,
                alpha: float = 0.2,
                max_memory_size: int = 10000,
                temporal_attention_heads: int = 4,
                use_temporal_attention: bool = True):
        """
        Initialize a TempGAT model.
        
        Args:
            input_dim: Size of input features
            hidden_dim: Size of hidden features
            output_dim: Size of output features
            num_heads: Number of attention heads
            memory_decay_factor: Factor to decay embeddings over time
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
            max_memory_size: Maximum number of nodes to store in memory
            temporal_attention_heads: Number of attention heads for temporal attention
            use_temporal_attention: Whether to use temporal attention mechanism
        """
        super(TempGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_temporal_attention = use_temporal_attention
        
        # Snapshot GAT model
        self.snapshot_gat = SnapshotGAT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            alpha=alpha
        )
        
        # Temporal attention layer
        if use_temporal_attention:
            self.temporal_attention = TemporalAttention(
                hidden_dim=output_dim,
                num_heads=temporal_attention_heads,
                dropout=dropout,
                alpha=alpha
            )
        
        # Memory bank
        self.memory_bank = MemoryBank(
            decay_factor=memory_decay_factor,
            max_size=max_memory_size
        )
        
        # Node initialization layer (for new nodes)
        self.node_init = nn.Linear(input_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                temporal_graph,
                snapshot_sequence: List[Dict],
                return_embeddings: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Dict]]]:
        """
        Forward pass of the TempGAT model.
        
        Args:
            temporal_graph: TemporalGraph instance
            snapshot_sequence: List of snapshot dictionaries
            return_embeddings: Whether to return node embeddings
            
        Returns:
            Predictions for the last snapshot, or tuple of (predictions, embeddings)
        """
        all_embeddings = []
        
        # Log memory usage at start of forward pass only for large sequences
        if len(snapshot_sequence) > 10:
            logger.info(f"Forward pass with {len(snapshot_sequence)} snapshots")
            manage_memory(log_memory=True)
        
        # Process each snapshot in sequence
        prev_snapshot = None
        
        for i, snapshot in enumerate(snapshot_sequence):
            try:
                # Validate and standardize the snapshot
                snapshot = validate_snapshot(snapshot)
                
                # Log memory usage less frequently to reduce output
                if i % 20 == 0 and self.memory_bank.get_node_count() > 5000:
                    logger.info(f"Processing snapshot {i}/{len(snapshot_sequence)}, "
                          f"Memory bank: {self.memory_bank.get_node_count()} nodes")
                    manage_memory(log_memory=True)
                
                # Handle empty snapshots
                if not snapshot['active_nodes']:
                    snapshot = handle_empty_snapshot(
                        snapshot['window_start'],
                        self.memory_bank
                    )
            except Exception as e:
                logger.warning(f"Error processing snapshot {i}: {e}")
                continue
            
            # Get active nodes and features
            active_nodes = snapshot['active_nodes']
            features = temporal_graph.get_node_features(active_nodes)
            
            # Create adjacency matrix
            adj = temporal_graph.create_adjacency_matrix(snapshot)
            
            # Create symmetric mask
            mask = create_symmetric_mask(adj)
            
            # Convert to PyTorch tensors
            features = torch.FloatTensor(features)
            mask = torch.FloatTensor(mask)
            
            # Move to device if available
            device = next(self.parameters()).device
            features = features.to(device)
            mask = mask.to(device)
            
            # Initialize embeddings for active nodes
            initial_embeddings = torch.zeros(len(active_nodes), self.output_dim).to(device)
            
            # Propagate embeddings from previous snapshot and memory
            if prev_snapshot is not None:
                memory_embeddings = propagate_between_snapshots(
                    prev_snapshot, 
                    snapshot, 
                    self.memory_bank
                )
                
                # Update initial embeddings with memory embeddings
                for idx, embedding in memory_embeddings.items():
                    initial_embeddings[idx] = embedding.to(device)
            
            # Initialize new nodes
            for i, node_id in enumerate(active_nodes):
                if torch.sum(initial_embeddings[i]) == 0:
                    # This is a new node or one without memory
                    initial_embeddings[i] = self.node_init(features[i])
            
            # Process snapshot with GAT
            if len(active_nodes) > 1:  # Need at least 2 nodes for attention
                embeddings = self.snapshot_gat.masked_forward(features, mask, mask)
                
                # Combine with initial embeddings (residual connection)
                embeddings = embeddings + initial_embeddings
            else:
                # For single node or empty snapshot, just use initial embeddings
                embeddings = initial_embeddings
                
            # Ensure embeddings have requires_grad=True for backpropagation
            if not embeddings.requires_grad:
                embeddings = embeddings.detach().clone().requires_grad_(True)
            
            # Store embeddings in memory bank
            timestamp = snapshot['window_start']
            
            # Use batch_store_nodes for better memory management
            self.memory_bank.batch_store_nodes(active_nodes, embeddings.detach(), timestamp)
            
            # Save snapshot embeddings - detach to avoid memory leaks
            snapshot_result = {
                'timestamp': snapshot['timestamp'],
                'active_nodes': active_nodes,
                'embeddings': embeddings.detach()  # Detach to avoid memory leaks
            }
            all_embeddings.append(snapshot_result)
            
            # Update previous snapshot
            prev_snapshot = {
                'timestamp': snapshot['timestamp'],
                'active_nodes': active_nodes.copy(),  # Make a copy to avoid reference issues
                'window_start': snapshot['window_start'],
                'window_end': snapshot['window_end'] if 'window_end' in snapshot else snapshot['window_start'] + 1
            }
            
            # Use centralized memory management periodically
            if i % 10 == 0:
                manage_memory(force_gc=True, log_memory=True)
        
        # Apply temporal attention if enabled and we have multiple snapshots
        if self.use_temporal_attention and len(all_embeddings) > 1:
            # Only log for large node counts to reduce output
            large_node_count = False
            if all_embeddings and 'active_nodes' in all_embeddings[-1]:
                node_count = len(all_embeddings[-1]['active_nodes'])
                large_node_count = node_count > 5000
                
            if large_node_count:
                logger.info(f"Temporal attention for {node_count} nodes")
                manage_memory(log_memory=True)
            
            # Extract embeddings from all snapshots
            sequence_embeddings = [snapshot['embeddings'].detach() for snapshot in all_embeddings]
            
            # Apply temporal attention
            temporal_embeddings = self.temporal_attention(sequence_embeddings)
            
            # Update the last snapshot's embeddings with temporally attended embeddings
            if temporal_embeddings is not None:
                all_embeddings[-1]['embeddings'] = temporal_embeddings
                
            # Use centralized memory management after temporal attention
            manage_memory(force_gc=True, log_memory=True)
        
        # Return embeddings for the last snapshot
        if all_embeddings:
            last_embeddings = all_embeddings[-1]['embeddings']
            
            # Clear all_embeddings except the last one to free memory
            if len(all_embeddings) > 1:
                last_snapshot = all_embeddings[-1]
                all_embeddings.clear()
                all_embeddings.append(last_snapshot)
            
            if return_embeddings:
                return last_embeddings, all_embeddings
            else:
                return last_embeddings
        else:
            # If no snapshots were processed successfully, return empty tensor
            print("Warning: No snapshots were processed successfully")
            device = next(self.parameters()).device
            empty_embeddings = torch.zeros(1, self.output_dim).to(device)
            
            if return_embeddings:
                return empty_embeddings, []
            else:
                return empty_embeddings
    
    def predict(self, 
               temporal_graph, 
               snapshot_sequence: List[Dict], 
               task: str = 'node_classification') -> torch.Tensor:
        """
        Make predictions using the TempGAT model.
        
        Args:
            temporal_graph: TemporalGraph instance
            snapshot_sequence: List of snapshot dictionaries
            task: Prediction task ('node_classification' or 'link_prediction')
            
        Returns:
            Predictions tensor
        """
        # Get embeddings from forward pass
        embeddings, all_snapshot_embeddings = self.forward(
            temporal_graph, 
            snapshot_sequence, 
            return_embeddings=True
        )
        
        if task == 'node_classification':
            # Apply softmax for node classification
            predictions = F.softmax(embeddings, dim=1)
            return predictions
        
        elif task == 'link_prediction':
            # Get the last snapshot
            last_snapshot = snapshot_sequence[-1]
            active_nodes = last_snapshot['active_nodes']
            
            # Create all possible node pairs
            node_pairs = []
            scores = []
            
            for i, source in enumerate(active_nodes):
                for j, target in enumerate(active_nodes):
                    if i != j:  # Exclude self-loops
                        # Compute edge score as dot product of embeddings
                        score = torch.dot(embeddings[i], embeddings[j])
                        node_pairs.append((source, target))
                        scores.append(score)
            
            # Convert to tensor
            if scores:
                edge_scores = torch.stack(scores)
                return edge_scores
            else:
                return torch.tensor([])
        
        else:
            raise ValueError(f"Unknown task: {task}")
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\training\__init__.py

```
from src.training.trainer import TemporalTrainer

__all__ = [
    'TemporalTrainer'
]
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\training\trainer.py

```
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from src.data.data import TemporalGraph, create_temporal_batches
from src.model.model import TempGAT
from src.utils.utils import node_classification_loss, link_prediction_loss, manage_memory, logger


class TemporalTrainer:
    """
    Trainer for TempGAT models.
    
    Handles training, validation, and evaluation of TempGAT models
    on temporal graph data.
    """
    
    def __init__(self, 
                model: TempGAT, 
                temporal_graph: TemporalGraph,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize a temporal trainer.
        
        Args:
            model: TempGAT model
            temporal_graph: TemporalGraph instance
            device: Device to use for training ('cuda' or 'cpu')
        """
        self.model = model
        self.temporal_graph = temporal_graph
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    def train(self,
             num_epochs: int = 200,
             batch_size: int = 32,
             sequence_length: int = 10,
             learning_rate: float = 0.001,
             weight_decay: float = 5e-4,
             val_ratio: float = 0.2,
             task: str = 'node_classification',
             patience: int = 10,
             verbose: bool = True,
             scheduler_type: str = 'plateau',  # 'plateau', 'cosine', or None
             scheduler_params: Dict = None) -> Dict:
        """
        Train the TempGAT model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Number of sequences per batch
            sequence_length: Number of snapshots per sequence
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            val_ratio: Ratio of data to use for validation
            task: Task to train for ('node_classification' or 'link_prediction')
            patience: Number of epochs to wait for improvement before early stopping
            verbose: Whether to print progress
            scheduler_type: Type of learning rate scheduler to use:
                - 'plateau': ReduceLROnPlateau (reduces LR when metric plateaus)
                - 'cosine': CosineAnnealingLR (cosine annealing schedule)
                - None: No scheduler
            scheduler_params: Parameters for the scheduler:
                - For 'plateau': {'factor': 0.1, 'patience': 5, 'min_lr': 1e-6}
                - For 'cosine': {'T_max': num_epochs, 'eta_min': 0}
            
        Returns:
            Training history
        """
        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = None
        if scheduler_type:
            if scheduler_params is None:
                scheduler_params = {}
                
            if scheduler_type == 'plateau':
                # Default parameters for ReduceLROnPlateau
                default_params = {
                    'factor': 0.1,       # Factor by which to reduce LR
                    'patience': 5,       # Number of epochs with no improvement
                    'min_lr': 1e-6       # Minimum LR
                }
                # Update with user-provided parameters
                default_params.update(scheduler_params)
                
                # Remove 'verbose' if it exists in the parameters
                if 'verbose' in default_params:
                    del default_params['verbose']
                
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',  # Minimize validation loss
                    **default_params
                )
                
            elif scheduler_type == 'cosine':
                # Default parameters for CosineAnnealingLR
                default_params = {
                    'T_max': num_epochs,  # Maximum number of iterations
                    'eta_min': 0          # Minimum learning rate
                }
                # Update with user-provided parameters
                default_params.update(scheduler_params)
                
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    **default_params
                )
                
            else:
                print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
        
        # Create batches
        all_batches = create_temporal_batches(
            self.temporal_graph,
            batch_size,
            sequence_length
        )
        
        # Split into train and validation
        num_val_batches = max(1, int(len(all_batches) * val_ratio))
        train_batches = all_batches[:-num_val_batches]
        val_batches = all_batches[-num_val_batches:]
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Add learning rate tracking to history
        self.history['learning_rates'] = []
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            start_time = time.time()
            
            # Process each batch
            for batch_idx, batch in enumerate(tqdm(train_batches) if verbose else train_batches):
                # Log memory usage less frequently to reduce output
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_batches)}")
                
                # Process each sequence in the batch
                batch_loss = 0.0
                
                for seq_idx, sequence in enumerate(batch):
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(self.temporal_graph, sequence)
                    
                    # Compute loss based on task
                    if task == 'node_classification':
                        # Get the last snapshot
                        last_snapshot = sequence[-1]
                        active_nodes = last_snapshot['active_nodes']
                        
                        # Get labels for active nodes
                        # Check if temporal_graph has node_labels
                        if hasattr(self.temporal_graph, 'node_labels'):
                            # Get labels for active nodes
                            node_labels = []
                            for node_id in active_nodes:
                                if node_id in self.temporal_graph.node_labels:
                                    node_labels.append(self.temporal_graph.node_labels[node_id])
                                else:
                                    # Use 0 as default label
                                    node_labels.append(0)
                            
                            # Convert to tensor
                            labels = torch.tensor(node_labels, dtype=torch.long).to(self.device)
                        else:
                            # Use placeholder labels if no node_labels available
                            labels = torch.zeros(len(active_nodes), dtype=torch.long).to(self.device)
                        
                        # Create mask for nodes with labels
                        mask = torch.ones(len(active_nodes), dtype=torch.bool).to(self.device)
                        
                        # Compute loss
                        loss = node_classification_loss(outputs, labels, mask)
                        
                    elif task == 'link_prediction':
                        # Get the last snapshot
                        last_snapshot = sequence[-1]
                        
                        # Get true edges
                        true_edges = torch.tensor(last_snapshot['edges'], dtype=torch.long).to(self.device)
                        
                        # Generate negative samples (placeholder - replace with actual negative sampling)
                        # In a real implementation, you would sample negative edges
                        negative_samples = torch.zeros_like(true_edges).to(self.device)
                        
                        # Compute edge scores
                        edge_scores = self.model.predict(
                            self.temporal_graph, 
                            sequence, 
                            task='link_prediction'
                        )
                        
                        # Compute loss
                        loss = link_prediction_loss(edge_scores, true_edges, negative_samples)
                        
                    else:
                        raise ValueError(f"Unknown task: {task}")
                    
                    # Backward pass and optimization
                    loss.backward()
                    
                    optimizer.step()
                    
                    # Accumulate loss
                    batch_loss += loss.item()
                
                # Average loss for the batch
                batch_loss /= len(batch)
                train_loss += batch_loss
                
                # Use centralized memory management periodically
                if batch_idx % 20 == 0:
                    manage_memory(force_gc=True, log_memory=False)
            
            # Average loss for the epoch
            if len(train_batches) > 0:
                train_loss /= len(train_batches)
                self.history['train_loss'].append(train_loss)
            else:
                logger.warning("No training batches were created. Check your data and batch parameters.")
                self.history['train_loss'].append(0.0)
                
            # Use centralized memory management at the end of each epoch
            logger.info(f"End of epoch {epoch+1}")
            manage_memory(force_gc=True, log_memory=True)
            
            # Validation phase
            val_loss, val_metrics = self.evaluate(val_batches, task)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Print progress
            if verbose:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s - "
                      f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                
                if task == 'node_classification':
                    print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
                elif task == 'link_prediction':
                    print(f"Val AUC: {val_metrics['auc']:.4f} - Val AP: {val_metrics['ap']:.4f}")
                
                # Print current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                if verbose:
                    print(f"Learning rate: {current_lr:.6f}")
            
            # Update scheduler
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    # ReduceLROnPlateau needs validation loss
                    scheduler.step(val_loss)
                elif scheduler_type == 'cosine':
                    # CosineAnnealingLR steps every epoch
                    scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.history
    
    def evaluate(self, 
                batches: List, 
                task: str = 'node_classification') -> Tuple[float, Dict]:
        """
        Evaluate the model on the given batches.
        
        Args:
            batches: List of batches to evaluate on
            task: Task to evaluate ('node_classification' or 'link_prediction')
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            # Process each batch
            for batch_idx, batch in enumerate(batches):
                # Process each sequence in the batch
                batch_loss = 0.0
                
                for sequence in batch:
                    # Forward pass
                    outputs = self.model(self.temporal_graph, sequence)
                    
                    # Compute loss and metrics based on task
                    if task == 'node_classification':
                        # Get the last snapshot
                        last_snapshot = sequence[-1]
                        active_nodes = last_snapshot['active_nodes']
                        
                        # Get labels for active nodes
                        # Check if temporal_graph has node_labels
                        if hasattr(self.temporal_graph, 'node_labels'):
                            # Get labels for active nodes
                            node_labels = []
                            for node_id in active_nodes:
                                if node_id in self.temporal_graph.node_labels:
                                    node_labels.append(self.temporal_graph.node_labels[node_id])
                                else:
                                    # Use 0 as default label
                                    node_labels.append(0)
                            
                            # Convert to tensor
                            labels = torch.tensor(node_labels, dtype=torch.long).to(self.device)
                        else:
                            # Use placeholder labels if no node_labels available
                            labels = torch.zeros(len(active_nodes), dtype=torch.long).to(self.device)
                        
                        # Create mask for nodes with labels
                        mask = torch.ones(len(active_nodes), dtype=torch.bool).to(self.device)
                        
                        # Compute loss
                        loss = node_classification_loss(outputs, labels, mask)
                        
                        # Get predictions
                        _, preds = torch.max(outputs, 1)
                        
                        # Collect predictions and labels for metrics
                        all_predictions.append(preds[mask].cpu().numpy())
                        all_labels.append(labels[mask].cpu().numpy())
                        
                    elif task == 'link_prediction':
                        # Get the last snapshot
                        last_snapshot = sequence[-1]
                        
                        # Get true edges
                        true_edges = torch.tensor(last_snapshot['edges'], dtype=torch.long).to(self.device)
                        
                        # Generate negative samples (placeholder - replace with actual negative sampling)
                        # In a real implementation, you would sample negative edges
                        negative_samples = torch.zeros_like(true_edges).to(self.device)
                        
                        # Compute edge scores
                        edge_scores = self.model.predict(
                            self.temporal_graph, 
                            sequence, 
                            task='link_prediction'
                        )
                        
                        # Compute loss
                        loss = link_prediction_loss(edge_scores, true_edges, negative_samples)
                        
                        # Collect predictions and labels for metrics
                        all_predictions.append(edge_scores.cpu().numpy())
                        
                        # Create labels (1 for positive, 0 for negative)
                        labels = torch.zeros(true_edges.size(0) + negative_samples.size(0))
                        labels[:true_edges.size(0)] = 1.0
                        all_labels.append(labels.numpy())
                        
                    else:
                        raise ValueError(f"Unknown task: {task}")
                    
                    # Accumulate loss
                    batch_loss += loss.item()
                
                # Average loss for the batch
                batch_loss /= len(batch)
                total_loss += batch_loss
        # Average loss
        if len(batches) > 0:
            avg_loss = total_loss / len(batches)
        else:
            logger.warning("No evaluation batches were created. Check your data and batch parameters.")
            avg_loss = 0.0
        
        # Use centralized memory management after evaluation
        logger.debug("After evaluation")
        manage_memory(force_gc=True, log_memory=True)
        
        
        # Compute metrics
        metrics = {}
        
        if all_predictions and all_labels:
            # Concatenate predictions and labels
            all_predictions = np.concatenate(all_predictions)
            all_labels = np.concatenate(all_labels)
            
            if task == 'node_classification':
                # Compute accuracy
                accuracy = accuracy_score(all_labels, all_predictions)
                metrics['accuracy'] = accuracy
                
            elif task == 'link_prediction':
                # Compute AUC and AP
                auc = roc_auc_score(all_labels, all_predictions)
                ap = average_precision_score(all_labels, all_predictions)
                metrics['auc'] = auc
                metrics['ap'] = ap
        
        return avg_loss, metrics
    
    def plot_training_history(self):
        """Plot the training history."""
        # Determine number of subplots needed
        has_learning_rates = 'learning_rates' in self.history and self.history['learning_rates']
        num_plots = 3 if has_learning_rates else 2
        
        plt.figure(figsize=(15, 4))
        
        # Plot loss
        plt.subplot(1, num_plots, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot metrics
        plt.subplot(1, num_plots, 2)
        
        if self.history['val_metrics'] and 'accuracy' in self.history['val_metrics'][0]:
            # Plot accuracy for node classification
            accuracies = [metrics['accuracy'] for metrics in self.history['val_metrics']]
            plt.plot(accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            
        elif self.history['val_metrics'] and 'auc' in self.history['val_metrics'][0]:
            # Plot AUC and AP for link prediction
            aucs = [metrics['auc'] for metrics in self.history['val_metrics']]
            aps = [metrics['ap'] for metrics in self.history['val_metrics']]
            plt.plot(aucs, label='AUC')
            plt.plot(aps, label='AP')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Validation Metrics')
            plt.legend()
        
        # Plot learning rates if available
        if has_learning_rates:
            plt.subplot(1, num_plots, 3)
            plt.plot(self.history['learning_rates'], label='Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.yscale('log')  # Log scale for better visualization
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path: str):
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']
        self.model.to(self.device)
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\utils\__init__.py

```
from src.utils.utils import (
    create_symmetric_mask,
    masked_attention,
    sparse_to_torch_sparse,
    normalize_adjacency,
    add_self_loops,
    node_classification_loss,
    link_prediction_loss,
    manage_memory,
    validate_snapshot,
    create_empty_snapshot,
    logger
)

__all__ = [
    'create_symmetric_mask',
    'masked_attention',
    'sparse_to_torch_sparse',
    'normalize_adjacency',
    'add_self_loops',
    'node_classification_loss',
    'link_prediction_loss',
    'manage_memory',
    'validate_snapshot',
    'create_empty_snapshot',
    'logger'
]
```


### C:\Users\matth\Desktop\1-FRESH_RESEARCH\TAGAN_OPENAIo3\src\utils\utils.py

```
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import logging
import gc
from typing import Optional, Tuple, Dict, List, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TempGAT")

# Memory management utilities
def manage_memory(force_gc: bool = False, log_memory: bool = False) -> None:
    """
    Centralized memory management function.
    
    Args:
        force_gc: Whether to force garbage collection
        log_memory: Whether to log memory usage
    """
    if force_gc:
        gc.collect()
        
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
        
    if log_memory and hasattr(torch.cuda, 'memory_allocated'):
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2 if hasattr(torch.cuda, 'memory_reserved') else 0
        logger.info(f"GPU Memory: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# Graph utilities
def create_symmetric_mask(adjacency_matrix):
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

def masked_attention(query, key, value, mask, dropout=None):
    """
    Compute attention scores with masking to handle asymmetric propagation.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Attention mask
        dropout: Dropout layer (optional)
        
    Returns:
        Output tensor and attention weights
    """
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(key.size(-1))
    
    # Apply mask
    if mask is not None:
        # Set masked positions to -inf
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

def sparse_to_torch_sparse(sparse_mx):
    """
    Convert scipy sparse matrix to torch sparse tensor.
    
    Args:
        sparse_mx: Scipy sparse matrix
        
    Returns:
        Torch sparse tensor
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adjacency(adj):
    """
    Symmetrically normalize adjacency matrix for GCN.
    
    Args:
        adj: Adjacency matrix
        
    Returns:
        Normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def add_self_loops(adj):
    """
    Add self-loops to adjacency matrix.
    
    Args:
        adj: Adjacency matrix
        
    Returns:
        Adjacency matrix with self-loops
    """
    adj = adj.tolil()
    adj.setdiag(1)
    return adj.tocsr()

# Loss functions
def node_classification_loss(predictions, targets, mask):
    """
    Compute loss for node classification tasks.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        mask: Mask indicating which nodes to consider
        
    Returns:
        Loss value
    """
    # Apply mask to consider only active nodes
    masked_pred = predictions[mask]
    masked_targets = targets[mask]
    
    # Compute cross entropy loss
    loss = F.cross_entropy(masked_pred, masked_targets)
    
    return loss

def link_prediction_loss(edge_scores, true_edges, negative_samples):
    """
    Compute loss for link prediction tasks.
    
    Args:
        edge_scores: Predicted edge scores
        true_edges: True edges (positive samples)
        negative_samples: Negative edge samples
        
    Returns:
        Loss value
    """
    # Concatenate positive and negative samples
    all_edges = torch.cat([true_edges, negative_samples], dim=0)
    
    # Create labels (1 for positive, 0 for negative)
    labels = torch.zeros(all_edges.size(0))
    labels[:true_edges.size(0)] = 1.0
    
    # Compute binary cross entropy loss
    loss = F.binary_cross_entropy_with_logits(edge_scores, labels)
    
    return loss

# Snapshot utilities
def validate_snapshot(snapshot: Dict) -> Dict:
    """
    Validate and standardize a snapshot dictionary.
    
    Args:
        snapshot: A snapshot dictionary
        
    Returns:
        Validated and standardized snapshot dictionary
    """
    if not isinstance(snapshot, dict):
        raise ValueError(f"Snapshot must be a dictionary, got {type(snapshot)}")
    
    # Ensure required keys exist
    required_keys = ['active_nodes', 'edges', 'timestamp']
    for key in required_keys:
        if key not in snapshot:
            if key == 'active_nodes':
                snapshot['active_nodes'] = []
            elif key == 'edges':
                snapshot['edges'] = []
            elif key == 'timestamp':
                snapshot['timestamp'] = 0
    
    # Ensure window_start and window_end exist
    if 'window_start' not in snapshot:
        if 'timestamp' in snapshot:
            snapshot['window_start'] = snapshot['timestamp']
        else:
            snapshot['window_start'] = 0
            
    if 'window_end' not in snapshot:
        snapshot['window_end'] = snapshot['window_start'] + 1
    
    return snapshot

def create_empty_snapshot(timestamp: int) -> Dict:
    """
    Create an empty snapshot with the given timestamp.
    
    Args:
        timestamp: Timestamp for the snapshot
        
    Returns:
        Empty snapshot dictionary
    """
    return {
        'timestamp': timestamp,
        'active_nodes': [],
        'edges': [],
        'window_start': timestamp,
        'window_end': timestamp + 1
    }
```
