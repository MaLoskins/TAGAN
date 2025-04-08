import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple


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