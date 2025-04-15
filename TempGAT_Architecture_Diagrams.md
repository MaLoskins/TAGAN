# TempGAT Architecture Diagrams

This document contains various diagrams illustrating the TempGAT architecture, component interactions, and data flow for Twitter rumor analysis.

## 1. System Overview

This diagram shows the high-level components of the TempGAT system:

```mermaid
graph TD
    A[Twitter Rumor Data] --> B[Data Processing Pipeline]
    B --> C[Temporal Graph Structure]
    C --> D[TempGAT Model]
    D --> E[Rumor Detection Results]
    
    B --> B1[download_twitter_rumor.py]
    B --> B2[preprocess_dataset.py]
    
    C --> C1[TemporalGraph Class]
    C --> C2[Snapshots]
    
    D --> D1[GraphAttentionLayer]
    D --> D2[SnapshotGAT]
    D --> D3[TemporalAttention]
    D --> D4[MemoryBank]
```

## 2. Component Interactions

This diagram shows how the different components of the TempGAT model interact:

```mermaid
graph TD
    A[TempGAT] --> B[SnapshotGAT]
    A --> C[MemoryBank]
    A --> D[TemporalAttention]
    
    B --> B1[GraphAttentionLayer]
    B --> B2[Multi-head Attention]
    
    C --> C1[Store Node]
    C --> C2[Retrieve Node]
    C --> C3[Prune Memory]
    
    D --> D1[Temporal Sequence Processing]
    D --> D2[Multi-head Attention]
    
    B --Node Embeddings--> C
    C --Retrieved Embeddings--> B
    B --Snapshot Embeddings--> D
    D --Temporal Embeddings--> A
```

## 3. Data Flow Diagram

This diagram illustrates how data flows through the TempGAT system:

```mermaid
flowchart TD
    A[Raw Twitter Data] --> B[Processed Data]
    B --> C[Temporal Graph Snapshots]
    C --> D[Feature Extraction]
    D --> E[Graph Attention Processing]
    E --> F[Memory Storage/Retrieval]
    F --> G[Temporal Attention Processing]
    G --> H[Final Node Embeddings]
    H --> I[Rumor Classification]
    
    subgraph "Data Processing"
    A --> B --> C
    end
    
    subgraph "TempGAT Model"
    D --> E --> F --> G --> H
    end
    
    subgraph "Output"
    H --> I
    end
```

## 4. Pipeline Execution

This diagram shows the execution flow of the Twitter rumor pipeline:

```mermaid
sequenceDiagram
    participant User
    participant Pipeline as run_twitter_rumor_pipeline.py
    participant Downloader as download_twitter_rumor.py
    participant Preprocessor as preprocess_dataset.py
    participant Runner as run_tempgat_on_social_data.py
    participant Model as TempGAT
    participant Trainer
    
    User->>Pipeline: Execute with arguments
    Pipeline->>Downloader: Download dataset (if not skipped)
    Downloader->>Pipeline: Return processed data
    Pipeline->>Preprocessor: Preprocess data
    Preprocessor->>Pipeline: Return temporal graph data
    Pipeline->>Runner: Run TempGAT on data
    Runner->>Model: Initialize model
    Runner->>Trainer: Create trainer
    Trainer->>Model: Train model
    Model->>Trainer: Return results
    Trainer->>Runner: Return metrics
    Runner->>Pipeline: Return final results
    Pipeline->>User: Display results
```

## 5. TempGAT Architecture

This diagram shows the detailed architecture of the TempGAT model:

```mermaid
graph TD
    A[Input: Temporal Graph Snapshots] --> B[SnapshotGAT]
    
    subgraph "SnapshotGAT"
    B --> C1[Multi-head Attention Layer 1]
    C1 --> C2[ELU Activation]
    C2 --> C3[Dropout]
    C3 --> C4[Multi-head Attention Layer 2]
    end
    
    B --> D[Node Embeddings]
    
    D --> E[MemoryBank]
    E --> F[Store Embeddings]
    E --> G[Retrieve Embeddings]
    G --> B
    
    D --> H[TemporalAttention]
    
    subgraph "TemporalAttention"
    H --> I1[Transform Embeddings]
    I1 --> I2[Compute Attention Scores]
    I2 --> I3[Apply Attention Weights]
    I3 --> I4[Combine Heads]
    end
    
    H --> J[Final Node Embeddings]
    J --> K[Task-specific Output]
    K --> L1[Node Classification]
    K --> L2[Link Prediction]
```

## 6. Memory Mechanism

This diagram illustrates how the memory mechanism works:

```mermaid
graph TD
    A[Active Nodes in Current Snapshot] --> B[Process with SnapshotGAT]
    B --> C[Node Embeddings]
    C --> D[Store in MemoryBank]
    
    E[Inactive Nodes] --> F[Retrieve from MemoryBank]
    F --> G[Apply Time Decay]
    G --> H[Use in Future Snapshots]
    
    I[Memory Size Check] --> J{Size > Max?}
    J -->|Yes| K[Calculate Node Importance]
    K --> L[Prune Least Important Nodes]
    J -->|No| M[Continue]
    
    D --> I
    L --> M
```

## 7. Temporal Attention Mechanism

This diagram shows how the temporal attention mechanism works:

```mermaid
graph TD
    A[Sequence of Node Embeddings] --> B[Pad Sequences]
    B --> C[Create Masks]
    
    subgraph "For Each Attention Head"
    C --> D[Transform Embeddings]
    D --> E[Compute Attention Scores]
    E --> F[Apply Masks]
    F --> G[Apply Softmax]
    G --> H[Apply Dropout]
    H --> I[Apply Attention to Values]
    end
    
    I --> J[Combine Heads]
    J --> K[Final Temporal Embeddings]
```

## 8. Data Processing Pipeline

This diagram illustrates the data processing pipeline:

```mermaid
flowchart TD
    A[Raw Twitter Rumor Data] --> B[Extract Tweets and Interactions]
    B --> C[Create User Features]
    C --> D[Create Temporal Windows]
    D --> E[Create Graph Snapshots]
    E --> F[Map Node IDs]
    F --> G[Create Adjacency Matrices]
    G --> H[Processed Temporal Graph Data]
    
    subgraph "Preprocessing"
    B --> C --> D
    end
    
    subgraph "Graph Creation"
    E --> F --> G
    end
```

## 9. Training and Evaluation Workflow

This diagram shows the training and evaluation workflow:

```mermaid
graph TD
    A[Temporal Graph Data] --> B[Create Batches]
    B --> C[Split Train/Val/Test]
    
    C --> D[Train Model]
    D --> E[Evaluate on Validation]
    E --> F{Early Stopping?}
    F -->|No| D
    F -->|Yes| G[Evaluate on Test]
    
    G --> H[Compute Metrics]
    H --> I[Report Results]
    
    subgraph "Training Loop"
    D --> D1[Process Sequences]
    D1 --> D2[Compute Loss]
    D2 --> D3[Backpropagation]
    D3 --> D4[Update Parameters]
    D4 --> D
    end
```

## 10. Command Execution Flow

This diagram shows the execution flow of the command `python run_twitter_rumor_pipeline.py --skip-download --dataset pheme`:

```mermaid
flowchart TD
    A[Command Execution] --> B[Parse Arguments]
    B --> C{Skip Download?}
    C -->|Yes| D[Skip Dataset Download]
    C -->|No| E[Download Dataset]
    
    D --> F{Skip Preprocessing?}
    E --> F
    F -->|Yes| G[Skip Preprocessing]
    F -->|No| H[Preprocess Dataset]
    
    G --> I[Run TempGAT]
    H --> I
    
    I --> J[Initialize Model]
    J --> K[Train Model]
    K --> L[Evaluate Model]
    L --> M[Report Results]
    
    subgraph "Model Parameters"
    J --> J1[hidden_dim: 64]
    J --> J2[output_dim: 32]
    J --> J3[num_heads: 8]
    J --> J4[memory_decay: 0.9]
    J --> J5[dropout: 0.2]
    end
```

These diagrams provide a visual representation of the TempGAT architecture and its components, helping to understand how the system works for Twitter rumor analysis.
