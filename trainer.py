import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from data import TemporalGraph, create_temporal_batches
from model import TempGAT
from utils import node_classification_loss, link_prediction_loss


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
                if batch_idx % 50 == 0 and hasattr(torch.cuda, 'memory_allocated'):
                    print(f"[Trainer] Epoch {epoch+1}, Batch {batch_idx}/{len(train_batches)}")
                
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
                
                # Force garbage collection and clear CUDA cache periodically
                if batch_idx % 20 == 0 and hasattr(torch.cuda, 'memory_allocated'):
                    import gc
                    gc.collect()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
            
            # Average loss for the epoch
            if len(train_batches) > 0:
                train_loss /= len(train_batches)
                self.history['train_loss'].append(train_loss)
            else:
                print("Warning: No training batches were created. Check your data and batch parameters.")
                self.history['train_loss'].append(0.0)
                
            # Force garbage collection at the end of each epoch
            import gc
            gc.collect()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                print(f"[Trainer] End of epoch {epoch+1}: GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            
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
            print("Warning: No evaluation batches were created. Check your data and batch parameters.")
            avg_loss = 0.0
        
        # Force garbage collection after evaluation
        import gc
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        
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