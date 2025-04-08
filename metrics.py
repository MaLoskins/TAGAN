import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
import time


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
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        micro_f1 = f1_score(all_labels, all_preds, average='micro')
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1
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
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        # Compute metrics
        auc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        return {
            'auc': auc,
            'ap': ap
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
                
                # Compute metrics
                metrics = temporal_node_classification_metrics(
                    future_preds, 
                    future_labels, 
                    future_masks
                )
                
            elif task == 'link_prediction':
                # Get predictions for future snapshots
                future_scores = []
                future_true_edges = []
                future_neg_edges = []
                
                for snapshot in future:
                    # Get true edges
                    true_edges = torch.tensor(snapshot['edges']).to(device)
                    
                    # Generate negative samples (placeholder - replace with actual negative sampling)
                    neg_edges = torch.zeros_like(true_edges).to(device)
                    
                    # Get predictions
                    with torch.no_grad():
                        scores = model.predict(temporal_graph, [snapshot], task='link_prediction')
                    
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