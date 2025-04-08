import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
import scipy.sparse as sp

from memory import MemoryBank, propagate_between_snapshots, handle_empty_snapshot, initialize_new_node
from utils import create_symmetric_mask, masked_attention, sparse_to_torch_sparse


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
        # Apply dropout to input features
        x = self.dropout(features)
        
        # Apply first layer with multiple attention heads and masking
        outputs = []
        for att in self.attentions:
            # Use masked attention for each head
            h = torch.mm(x, att.W)  # [N, hidden_dim]
            N = h.size(0)
            
            # Prepare for attention
            a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                                h.repeat(N, 1)], dim=1).view(N, N, 2 * att.out_features)
            
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
        
        # Create padded tensor for all embeddings
        padded_embeddings = torch.zeros(seq_len, max_nodes, feature_dim).to(device)
        masks = torch.zeros(seq_len, max_nodes, dtype=torch.bool).to(device)
        
        # Fill padded tensor with embeddings and create masks
        for t, embeddings in enumerate(sequence_embeddings):
            num_nodes = embeddings.size(0)
            padded_embeddings[t, :num_nodes] = embeddings
            masks[t, :num_nodes] = True
        
        # Apply multi-head attention
        outputs = []
        for head in range(self.num_heads):
            # Transform embeddings
            transformed = torch.matmul(padded_embeddings, self.W[head])  # [T, N, F]
            
            # Compute attention scores between all time steps
            attention_input = []
            for t1 in range(seq_len):
                for t2 in range(seq_len):
                    # Concatenate embeddings from different time steps
                    concat = torch.cat([
                        transformed[t1].unsqueeze(1).repeat(1, max_nodes, 1),
                        transformed[t2].unsqueeze(0).repeat(max_nodes, 1, 1)
                    ], dim=2)  # [N, N, 2F]
                    
                    attention_input.append(concat)
            
            # Stack attention inputs
            attention_input = torch.stack(attention_input)  # [T*T, N, N, 2F]
            
            # Compute attention coefficients
            e = self.leakyrelu(torch.matmul(attention_input, self.a[head]))  # [T*T, N, N, 1]
            e = e.squeeze(-1)  # [T*T, N, N]
            
            # Create attention mask based on node presence
            att_mask = torch.zeros(seq_len, seq_len, max_nodes, max_nodes, dtype=torch.bool).to(device)
            for t1 in range(seq_len):
                for t2 in range(seq_len):
                    # Nodes must be present in both time steps to attend
                    att_mask[t1, t2] = masks[t1].unsqueeze(1) & masks[t2].unsqueeze(0)
            
            # Reshape mask to match attention scores
            att_mask = att_mask.view(-1, max_nodes, max_nodes)  # [T*T, N, N]
            
            # Apply mask
            e = e.masked_fill(~att_mask, -9e15)
            
            # Apply softmax to get attention weights
            attention = F.softmax(e, dim=1)  # [T*T, N, N]
            attention = self.dropout_layer(attention)
            
            # Apply attention to transformed embeddings
            weighted = []
            idx = 0
            for t1 in range(seq_len):
                t1_weighted = []
                for t2 in range(seq_len):
                    # Apply attention weights from t1 to t2
                    attended = torch.matmul(attention[idx], transformed[t2])  # [N, F]
                    t1_weighted.append(attended)
                    idx += 1
                # Combine attended values from all time steps
                t1_weighted = torch.stack(t1_weighted).mean(dim=0)  # [N, F]
                weighted.append(t1_weighted)
            
            # Get final output for this head
            head_output = weighted[-1]  # Use the last time step's output
            outputs.append(head_output)
        
        # Combine outputs from all heads
        final_output = torch.mean(torch.stack(outputs), dim=0)  # [N, F]
        
        # Only keep embeddings for nodes that exist in the last time step
        last_mask = masks[-1]
        final_output = final_output[last_mask]
        
        return final_output


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
        
        # Process each snapshot in sequence
        prev_snapshot = None
        
        for i, snapshot in enumerate(snapshot_sequence):
            # Check if snapshot is a dictionary
            if not isinstance(snapshot, dict):
                print(f"Warning: Snapshot {i} is not a dictionary. Type: {type(snapshot)}")
                continue
                
            # Handle empty snapshots
            if 'active_nodes' not in snapshot or not snapshot['active_nodes']:
                if 'window_start' in snapshot:
                    empty_snapshot = handle_empty_snapshot(
                        snapshot['window_start'],
                        self.memory_bank
                    )
                else:
                    # Use a default timestamp if window_start is not available
                    empty_snapshot = handle_empty_snapshot(
                        i * 15,  # Default window size of 15 minutes
                        self.memory_bank
                    )
                snapshot = empty_snapshot
            
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
            for i, node_id in enumerate(active_nodes):
                self.memory_bank.store_node(node_id, embeddings[i], timestamp)
            
            # Save snapshot embeddings
            snapshot_result = {
                'timestamp': snapshot['timestamp'],
                'active_nodes': active_nodes,
                'embeddings': embeddings
            }
            all_embeddings.append(snapshot_result)
            
            # Update previous snapshot
            prev_snapshot = snapshot
        
        # Apply temporal attention if enabled and we have multiple snapshots
        if self.use_temporal_attention and len(all_embeddings) > 1:
            # Extract embeddings from all snapshots
            sequence_embeddings = [snapshot['embeddings'] for snapshot in all_embeddings]
            
            # Apply temporal attention
            temporal_embeddings = self.temporal_attention(sequence_embeddings)
            
            # Update the last snapshot's embeddings with temporally attended embeddings
            if temporal_embeddings is not None:
                all_embeddings[-1]['embeddings'] = temporal_embeddings
        
        # Return embeddings for the last snapshot
        if all_embeddings:
            last_embeddings = all_embeddings[-1]['embeddings']
            
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