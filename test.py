import unittest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data import TemporalGraph, create_temporal_batches
from memory import MemoryBank, propagate_between_snapshots, handle_empty_snapshot
from model import TempGAT, SnapshotGAT
from utils import create_symmetric_mask, masked_attention


class TestTemporalGraph(unittest.TestCase):
    """Test cases for the TemporalGraph class."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic interaction data
        self.interactions = []
        
        # Create 10 nodes with interactions over 5 time windows
        for t in range(5):
            timestamp = datetime(2023, 1, 1) + timedelta(minutes=t*15)
            
            # Each time window has different active nodes
            active_nodes = np.random.choice(10, size=5, replace=False)
            
            # Create interactions between active nodes
            for i in range(len(active_nodes)):
                source = active_nodes[i]
                for j in range(i+1, len(active_nodes)):
                    target = active_nodes[j]
                    
                    # Add interaction
                    self.interactions.append({
                        'timestamp': timestamp,
                        'source_id': int(source),
                        'target_id': int(target),
                        'feature1': np.random.randn(),
                        'feature2': np.random.randn()
                    })
        
        # Create DataFrame
        self.df = pd.DataFrame(self.interactions)
    
    def test_from_interactions(self):
        """Test creating a temporal graph from interactions."""
        # Create temporal graph
        temporal_graph = TemporalGraph.from_interactions(
            self.df,
            time_column='timestamp',
            source_column='source_id',
            target_column='target_id',
            features_columns=['feature1', 'feature2'],
            window_size=15
        )
        
        # Check basic properties
        self.assertIsNotNone(temporal_graph)
        self.assertEqual(temporal_graph.window_size, 15)
        self.assertGreater(len(temporal_graph.snapshots), 0)
        self.assertGreater(temporal_graph.num_nodes, 0)
        self.assertEqual(temporal_graph.feature_dim, 2)
    
    def test_get_snapshot_sequence(self):
        """Test getting a sequence of snapshots."""
        # Create temporal graph
        temporal_graph = TemporalGraph.from_interactions(
            self.df,
            time_column='timestamp',
            source_column='source_id',
            target_column='target_id',
            features_columns=['feature1', 'feature2'],
            window_size=15
        )
        
        # Get sequence
        start_time = 0  # minutes since start
        end_time = 60  # minutes since start
        sequence = temporal_graph.get_snapshot_sequence(start_time, end_time)
        
        # Check sequence
        self.assertIsNotNone(sequence)
        self.assertIsInstance(sequence, list)
    
    def test_create_adjacency_matrix(self):
        """Test creating an adjacency matrix for a snapshot."""
        # Create temporal graph
        temporal_graph = TemporalGraph.from_interactions(
            self.df,
            time_column='timestamp',
            source_column='source_id',
            target_column='target_id',
            features_columns=['feature1', 'feature2'],
            window_size=15
        )
        
        # Get first snapshot
        snapshot = temporal_graph.snapshots[0]
        
        # Create adjacency matrix
        adj = temporal_graph.create_adjacency_matrix(snapshot)
        
        # Check adjacency matrix
        self.assertIsNotNone(adj)
        self.assertEqual(adj.shape[0], len(snapshot['active_nodes']))
        self.assertEqual(adj.shape[1], len(snapshot['active_nodes']))


class TestMemoryBank(unittest.TestCase):
    """Test cases for the MemoryBank class."""
    
    def setUp(self):
        """Set up test data."""
        # Create memory bank
        self.memory_bank = MemoryBank(decay_factor=0.9, max_size=100)
        
        # Create test embeddings
        self.embeddings = torch.randn(10, 32)  # 10 nodes, 32-dim embeddings
    
    def test_store_retrieve_node(self):
        """Test storing and retrieving node embeddings."""
        # Store embeddings
        for i in range(10):
            self.memory_bank.store_node(i, self.embeddings[i], timestamp=0)
        
        # Check node count
        self.assertEqual(self.memory_bank.get_node_count(), 10)
        
        # Retrieve embeddings
        for i in range(10):
            embedding = self.memory_bank.retrieve_node(i, current_timestamp=0)
            self.assertIsNotNone(embedding)
            self.assertTrue(torch.allclose(embedding, self.embeddings[i]))
    
    def test_time_decay(self):
        """Test time decay of embeddings."""
        # Store embedding
        node_id = 0
        self.memory_bank.store_node(node_id, self.embeddings[node_id], timestamp=0)
        
        # Retrieve with time decay
        decayed_embedding = self.memory_bank.retrieve_node(node_id, current_timestamp=2)
        
        # Check decay
        expected_decay = self.embeddings[node_id] * (0.9 ** 2)
        self.assertTrue(torch.allclose(decayed_embedding, expected_decay))
    
    def test_prune_memory_bank(self):
        """Test pruning the memory bank."""
        # Store many embeddings
        for i in range(200):
            self.memory_bank.store_node(i, torch.randn(32), timestamp=0)
        
        # Check that pruning occurred
        self.assertLessEqual(self.memory_bank.get_node_count(), 100)
    
    def test_batch_operations(self):
        """Test batch store and retrieve operations."""
        # Batch store
        node_ids = list(range(10))
        self.memory_bank.batch_store_nodes(node_ids, self.embeddings, timestamp=0)
        
        # Batch retrieve
        retrieved_embeddings = self.memory_bank.batch_retrieve_nodes(node_ids, current_timestamp=0)
        
        # Check retrieved embeddings
        self.assertEqual(retrieved_embeddings.shape, self.embeddings.shape)
        self.assertTrue(torch.allclose(retrieved_embeddings, self.embeddings))


class TestModel(unittest.TestCase):
    """Test cases for the TempGAT model."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic interaction data
        self.interactions = []
        
        # Create 10 nodes with interactions over 5 time windows
        for t in range(5):
            timestamp = datetime(2023, 1, 1) + timedelta(minutes=t*15)
            
            # Each time window has different active nodes
            active_nodes = np.random.choice(10, size=5, replace=False)
            
            # Create interactions between active nodes
            for i in range(len(active_nodes)):
                source = active_nodes[i]
                for j in range(i+1, len(active_nodes)):
                    target = active_nodes[j]
                    
                    # Add interaction
                    self.interactions.append({
                        'timestamp': timestamp,
                        'source_id': int(source),
                        'target_id': int(target),
                        'feature1': np.random.randn(),
                        'feature2': np.random.randn()
                    })
        
        # Create DataFrame
        self.df = pd.DataFrame(self.interactions)
        
        # Create temporal graph
        self.temporal_graph = TemporalGraph.from_interactions(
            self.df,
            time_column='timestamp',
            source_column='source_id',
            target_column='target_id',
            features_columns=['feature1', 'feature2'],
            window_size=15
        )
        
        # Create model
        self.model = TempGAT(
            input_dim=self.temporal_graph.feature_dim,
            hidden_dim=16,
            output_dim=8,
            num_heads=2,
            memory_decay_factor=0.9,
            dropout=0.1
        )
    
    def test_snapshot_gat(self):
        """Test the SnapshotGAT model."""
        # Create SnapshotGAT
        snapshot_gat = SnapshotGAT(
            input_dim=self.temporal_graph.feature_dim,
            hidden_dim=16,
            output_dim=8,
            num_heads=2,
            dropout=0.1
        )
        
        # Get first snapshot
        snapshot = self.temporal_graph.snapshots[0]
        
        # Get features and adjacency matrix
        features = self.temporal_graph.get_node_features(snapshot['active_nodes'])
        adj = self.temporal_graph.create_adjacency_matrix(snapshot)
        mask = create_symmetric_mask(adj)
        
        # Convert to PyTorch tensors
        features = torch.FloatTensor(features)
        mask = torch.FloatTensor(mask)
        
        # Forward pass
        output = snapshot_gat.masked_forward(features, mask, mask)
        
        # Check output
        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], len(snapshot['active_nodes']))
        self.assertEqual(output.shape[1], 8)
    
    def test_tempgat_forward(self):
        """Test the TempGAT forward pass."""
        # Get sequence of snapshots
        sequence = self.temporal_graph.snapshots[:3]
        
        # Forward pass
        output = self.model(self.temporal_graph, sequence)
        
        # Check output
        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], len(sequence[-1]['active_nodes']))
        self.assertEqual(output.shape[1], 8)
    
    def test_tempgat_predict(self):
        """Test the TempGAT predict method."""
        # Get sequence of snapshots
        sequence = self.temporal_graph.snapshots[:3]
        
        # Node classification
        node_preds = self.model.predict(
            self.temporal_graph, 
            sequence, 
            task='node_classification'
        )
        
        # Check node predictions
        self.assertIsNotNone(node_preds)
        self.assertEqual(node_preds.shape[0], len(sequence[-1]['active_nodes']))
        
        # Link prediction
        link_preds = self.model.predict(
            self.temporal_graph, 
            sequence, 
            task='link_prediction'
        )
        
        # Check link predictions (may be empty if no edges)
        self.assertIsNotNone(link_preds)


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_symmetric_mask(self):
        """Test creating a symmetric mask."""
        # Create asymmetric adjacency matrix
        import scipy.sparse as sp
        adj = sp.csr_matrix((3, 3))
        adj[0, 1] = 1  # Edge from 0 to 1
        
        # Create symmetric mask
        mask = create_symmetric_mask(adj)
        
        # Check mask
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (3, 3))
        self.assertEqual(mask[0, 1].item(), 1)  # Edge from 0 to 1
        self.assertEqual(mask[1, 0].item(), 1)  # Edge from 1 to 0 (symmetrized)
    
    def test_masked_attention(self):
        """Test masked attention."""
        # Create query, key, value
        query = torch.randn(3, 4)
        key = torch.randn(3, 4)
        value = torch.randn(3, 4)
        
        # Create mask
        mask = torch.ones(3, 3)
        mask[0, 2] = 0  # Mask out attention from node 0 to node 2
        
        # Compute attention
        output, attention_weights = masked_attention(query, key, value, mask)
        
        # Check output
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (3, 4))
        
        # Check attention weights
        self.assertIsNotNone(attention_weights)
        self.assertEqual(attention_weights.shape, (3, 3))
        self.assertAlmostEqual(attention_weights[0, 2].item(), 0, places=5)  # Should be masked out


class TestIntegration(unittest.TestCase):
    """Integration tests for the TempGAT model."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic interaction data
        self.interactions = []
        
        # Create 20 nodes with interactions over 10 time windows
        for t in range(10):
            timestamp = datetime(2023, 1, 1) + timedelta(minutes=t*15)
            
            # Each time window has different active nodes
            active_nodes = np.random.choice(20, size=10, replace=False)
            
            # Create interactions between active nodes
            for i in range(len(active_nodes)):
                source = active_nodes[i]
                for j in range(i+1, len(active_nodes)):
                    target = active_nodes[j]
                    
                    # Add interaction
                    self.interactions.append({
                        'timestamp': timestamp,
                        'source_id': int(source),
                        'target_id': int(target),
                        'feature1': np.random.randn(),
                        'feature2': np.random.randn()
                    })
        
        # Create DataFrame
        self.df = pd.DataFrame(self.interactions)
        
        # Create temporal graph
        self.temporal_graph = TemporalGraph.from_interactions(
            self.df,
            time_column='timestamp',
            source_column='source_id',
            target_column='target_id',
            features_columns=['feature1', 'feature2'],
            window_size=15
        )
        
        # Create model
        self.model = TempGAT(
            input_dim=self.temporal_graph.feature_dim,
            hidden_dim=16,
            output_dim=8,
            num_heads=2,
            memory_decay_factor=0.9,
            dropout=0.1
        )
    
    def test_training_loop(self):
        """Test a complete training loop."""
        from trainer import TemporalTrainer
        
        # Create trainer
        trainer = TemporalTrainer(self.model, self.temporal_graph)
        
        # Train for a few epochs
        history = trainer.train(
            num_epochs=2,
            batch_size=2,
            sequence_length=3,
            learning_rate=0.01,
            task='node_classification',
            verbose=False
        )
        
        # Check history
        self.assertIsNotNone(history)
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_metrics', history)
    
    def test_memory_propagation(self):
        """Test memory propagation between snapshots."""
        # Get two consecutive snapshots
        if len(self.temporal_graph.snapshots) < 2:
            self.skipTest("Not enough snapshots for this test")
        
        prev_snapshot = self.temporal_graph.snapshots[0]
        curr_snapshot = self.temporal_graph.snapshots[1]
        
        # Create memory bank
        memory_bank = MemoryBank()
        
        # Store some embeddings in memory
        for node_id in prev_snapshot['active_nodes']:
            memory_bank.store_node(node_id, torch.randn(8), timestamp=0)
        
        # Propagate between snapshots
        memory_embeddings = propagate_between_snapshots(
            prev_snapshot, 
            curr_snapshot, 
            memory_bank
        )
        
        # Check memory embeddings
        self.assertIsNotNone(memory_embeddings)
    
    def test_empty_snapshot_handling(self):
        """Test handling of empty snapshots."""
        # Create memory bank
        memory_bank = MemoryBank()
        
        # Store some embeddings in memory
        for i in range(5):
            memory_bank.store_node(i, torch.randn(8), timestamp=0)
        
        # Handle empty snapshot
        snapshot = handle_empty_snapshot(10, memory_bank)
        
        # Check snapshot
        self.assertIsNotNone(snapshot)
        self.assertIn('active_nodes', snapshot)
        self.assertIn('edges', snapshot)
        self.assertEqual(len(snapshot['edges']), 0)  # No edges in empty snapshot


if __name__ == '__main__':
    unittest.main()