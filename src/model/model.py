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