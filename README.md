# TempGAT: Temporal Graph Attention Network with Dynamic Memory

TempGAT is a PyTorch implementation of a novel geometric neural network architecture that applies graph attention mechanisms to temporal social media data. The architecture processes graph data as a sequence of temporal snapshots, maintaining node states in a memory bank between active periods to optimize computational efficiency while preserving long-term dependencies.

## Architecture Overview

TempGAT extends Graph Attention Networks (GAT) with temporal dynamics through a novel approach:

1. **Temporal Discretization**: Rather than processing the entire graph simultaneously, the model segments the social media activity timeline into fixed temporal windows (e.g., 15-minute intervals), creating a sequence of graph "snapshots."

2. **Dynamic Node Participation**: Each snapshot contains only actively participating nodes within that time window. This creates variable-sized adjacency matrices across snapshots, as the number of active nodes fluctuates over time.

3. **Symmetric Masking for Asymmetric Interactions**: To handle the mathematical challenges of variable-sized matrices in deep learning operations, the model applies symmetric masking to the fundamentally asymmetric social media interactions, enabling proper gradient propagation while preserving directional information.

4. **Localized GAT Operations**: The standard GAT attention mechanism operates efficiently on each snapshot's significantly reduced node set, computing embeddings based on local graph structure.

5. **Temporal State Propagation**: As the model advances to subsequent snapshots:
   - Nodes persisting across snapshots carry forward their updated embeddings and attention weights
   - Nodes becoming inactive are transferred to a memory bank
   - Previously inactive nodes re-entering the graph retrieve their stored embeddings from the memory bank

6. **Memory Persistence Mechanism**: The memory bank serves as a stateful repository, preserving node embeddings during periods of inactivity, allowing the model to maintain long-term dependencies without the computational cost of processing the entire graph.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tempgat.git
cd tempgat

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- PyTorch 1.10+
- torch-geometric
- numpy
- scipy
- pandas
- matplotlib
- networkx
- scikit-learn

## Project Structure

- `data.py`: Implementation of the `TemporalGraph` class and data handling utilities
- `memory.py`: Implementation of the `MemoryBank` class and memory-related functions
- `model.py`: Implementation of the `TempGAT` and `SnapshotGAT` models
- `utils.py`: Utility functions for masking, attention, and other operations
- `trainer.py`: Training infrastructure for TempGAT models
- `metrics.py`: Evaluation metrics for temporal graph tasks
- `main.py`: Example script demonstrating TempGAT usage
- `test.py`: Unit tests for the TempGAT implementation

## Usage

### Basic Usage

```python
import torch
import pandas as pd
from data import TemporalGraph
from model import TempGAT
from trainer import TemporalTrainer

# Load your temporal interaction data
interactions_df = pd.read_csv('your_data.csv')

# Create a temporal graph
temporal_graph = TemporalGraph.from_interactions(
    interactions_df,
    time_column='timestamp',
    source_column='source_id',
    target_column='target_id',
    features_columns=['feature1', 'feature2'],
    window_size=15  # minutes
)

# Create a TempGAT model
model = TempGAT(
    input_dim=temporal_graph.feature_dim,
    hidden_dim=64,
    output_dim=32,
    num_heads=8,
    memory_decay_factor=0.9,
    dropout=0.2
)

# Create a trainer
trainer = TemporalTrainer(model, temporal_graph)

# Train the model
trainer.train(
    num_epochs=200,
    batch_size=32,
    sequence_length=10,  # snapshots per sequence
    learning_rate=0.001,
    task='node_classification'  # or 'link_prediction'
)

# Make predictions
predictions = model.predict(
    temporal_graph,
    temporal_graph.get_snapshot_sequence(start_time, end_time),
    task='node_classification'  # or 'link_prediction'
)
```

### Running the Demo

```bash
# Run with synthetic data
python main.py --synthetic --num_nodes 100 --num_timesteps 50 --visualize

# Run with your own data
python main.py --data_path your_data.csv --visualize
```

### Running Tests

```bash
python test.py
```

## Key Components

### TemporalGraph

The `TemporalGraph` class handles the conversion from raw temporal interaction data to a sequence of snapshot graphs. It supports variable-sized adjacency matrices between snapshots and provides methods for creating snapshots, retrieving node features, and creating adjacency matrices.

### MemoryBank

The `MemoryBank` class efficiently stores and retrieves node embeddings by node ID. It implements a time-decay mechanism for long-stored embeddings and handles pruning of rarely-accessed nodes to prevent memory explosion.

### TempGAT

The `TempGAT` class is the main model architecture, composed of `SnapshotGAT`, `MemoryBank`, and temporal propagation logic. It processes each snapshot in sequence, propagating node states between snapshots and maintaining long-term dependencies through the memory bank.

### SnapshotGAT

The `SnapshotGAT` class is a modified GAT implementation for individual snapshots. It handles variable-sized inputs through masking and maintains the multi-head attention mechanism from the original GAT.

## Performance Characteristics

The TempGAT implementation demonstrates:

1. Significant computational savings compared to full-graph GAT
2. Preservation of predictive performance despite reduced computation
3. Ability to capture long-term dependencies through the memory mechanism
4. Scalability to large social media graphs (≥100K nodes)

## Citation

If you use this code in your research, please cite:

```
@article{tempgat2023,
  title={TempGAT: Temporal Graph Attention Networks with Dynamic Memory},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.