# TempGAT Architecture Diagrams

This document provides detailed Mermaid diagrams of the Temporal Graph Attention Network (TempGAT) architecture, showing both high-level overview and component-specific details.

## 1. High-Level Architecture Overview

```mermaid
flowchart TB
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef memory fill:#e6d0ff,stroke:#8a2be2,stroke-width:2px,color:black
    classDef prediction fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    classDef edgeCase fill:#fff2cc,stroke:#d6b656,stroke-width:2px,color:black
    
    %% Main data flow
    InputData([Raw Temporal Interaction Data]):::data --> TempDiscretization{Temporal Discretization}:::process
    TempDiscretization --> Snapshots([Sequence of Snapshot Graphs]):::data
    Snapshots --> TempGAT
    TempGAT --> Predictions([Node States/Predictions]):::prediction
    
    %% TempGAT architecture
    subgraph TempGAT[TempGAT Model Architecture]
        direction TB
        
        subgraph ProcessingLoop["Temporal Processing Loop"]
            direction LR
            CurrentSnapshot([Current Snapshot t]):::data --> | Identify active nodes | ActiveNodes[[Active Nodes Set]]:::process
            MemoryBank -->| Retrieve embeddings for reactivated nodes | ActiveNodes
            ActiveNodes --> | Process with GAT | SnapshotGAT
            SnapshotGAT --> | Update embeddings | UpdatedNodes[[Updated Node Embeddings]]:::process
            UpdatedNodes --> | Identify inactive nodes | InactiveNodes[[Newly Inactive Nodes]]:::process
            UpdatedNodes --> | Pass active nodes | NextSnapshot([Next Snapshot t+1]):::data
        end
        
        InactiveNodes -->|Store embeddings| MemoryBank
        
        subgraph SnapshotGAT[SnapshotGAT Module]
            direction TB
            AdjMatrix[(Adjacency Matrix)]:::data --> SymMask{Symmetric Masking}:::process
            SymMask --> MaskAttn{Masked Attention Mechanism}:::process
            NodeFeatures[(Node Features)]:::data --> MaskAttn
            MaskAttn --> |Multi-head attention| GAT_Output[[Node Embeddings]]:::process
        end
        
        subgraph MemoryBank[Memory Bank]
            direction TB
            StoreNode[(Store Node Embeddings)]:::memory
            RetrieveNode[(Retrieve Node Embeddings)]:::memory
            TimeDecay{Apply Time Decay}:::process
            PruneMemory{Prune Memory}:::process
        end

        subgraph TemporalAttention[Temporal Attention]
            direction TB
            SeqEmbed[(Sequence Embeddings)]:::data --> TempAttn{Multi-head Temporal Attention}:::process
            TempAttn --> TempEmbed[[Temporally Attended Embeddings]]:::process
        end

        ProcessingLoop --> TemporalAttention
    end
    
    %% Downstream tasks
    Predictions --> NodeClass>Node Classification]:::prediction
    Predictions --> LinkPred>Link Prediction]:::prediction
    
    %% Edge cases
    subgraph EdgeCases[Edge Case Handling]
        direction TB
        EmptySnap{{Empty Snapshots}}:::edgeCase
        NewNodes{{New Node Initialization}}:::edgeCase
        MemoryOverflow{{Memory Overflow Protection}}:::edgeCase
    end
    
    TempGAT --- EdgeCases
    
    %% Style subgraphs
    style TempGAT fill:#f5f5f5,stroke:#333,stroke-width:2px
    style ProcessingLoop fill:#e6f3e6,stroke:#6aa84f,stroke-width:1px
    style SnapshotGAT fill:#e6f0ff,stroke:#4285f4,stroke-width:1px
    style MemoryBank fill:#f3e6ff,stroke:#9a67ea,stroke-width:1px
    style TemporalAttention fill:#fff0e6,stroke:#ff6d33,stroke-width:1px
    style EdgeCases fill:#fcf8e3,stroke:#c09853,stroke-width:1px
```

The high-level architecture shows how TempGAT processes temporal graph data through a sequence of snapshots, using memory to maintain node states between active periods and applying temporal attention to capture long-term dependencies.

## 2. Data Processing Component (TemporalGraph)

```mermaid
classDiagram
    class TemporalGraph {
        +int window_size
        +List snapshots
        +Dict node_id_map
        +Dict reverse_node_id_map
        +int feature_dim
        +int num_nodes
        +Dict node_features
        +__init__(window_size)
        +from_interactions(interactions_df, time_column, source_column, target_column, features_columns, window_size)
        +get_snapshot_sequence(start_time, end_time)
        +get_node_features(node_ids)
        +create_adjacency_matrix(snapshot)
        +create_symmetric_mask(adjacency_matrix)
    }

    class create_temporal_batches {
        +create_temporal_batches(temporal_graph, batch_size, sequence_length)
    }

    TemporalGraph -- create_temporal_batches : uses >
```

```mermaid
flowchart TB
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef output fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    
    %% Data processing flow
    RawData([Raw Interaction Data]):::data --> SortData{Sort by Timestamp}:::process
    SortData --> ConvertTime{Convert to Minutes}:::process
    ConvertTime --> CreateNodeMap{Create Node ID Mapping}:::process
    
    CreateNodeMap --> ExtractFeatures{Extract Node Features}:::process
    ExtractFeatures --> OneHotEncoding{One-Hot Encoding}:::process
    ExtractFeatures --> FeatureExtraction{Feature Extraction}:::process
    
    OneHotEncoding --> CreateSnapshots{Create Snapshots}:::process
    FeatureExtraction --> CreateSnapshots
    
    CreateSnapshots --> WindowInteractions{Get Window Interactions}:::process
    WindowInteractions --> ActiveNodes{Identify Active Nodes}:::process
    ActiveNodes --> CreateAdjMatrix{Create Adjacency Matrix}:::process
    CreateAdjMatrix --> SnapshotCreation{Create Snapshot}:::process
    
    SnapshotCreation --> Snapshots([Sequence of Snapshots]):::output
    
    %% Batch creation
    Snapshots --> CreateSequences{Create Sequences}:::process
    CreateSequences --> CreateBatches{Create Batches}:::process
    CreateBatches --> TemporalBatches([Temporal Batches]):::output
```

The TemporalGraph component handles the conversion of raw temporal interaction data into a sequence of snapshot graphs. It segments the timeline into fixed windows, identifies active nodes in each window, and creates adjacency matrices for each snapshot.

## 3. Memory Management Component (MemoryBank)

```mermaid
classDiagram
    class MemoryBank {
        +Dict node_embeddings
        +float decay_factor
        +int max_size
        +int pruning_threshold
        +int current_timestamp
        +float pruning_percentage
        +float buffer_percentage
        +__init__(decay_factor, max_size, pruning_threshold)
        +store_node(node_id, embedding, timestamp)
        +retrieve_node(node_id, current_timestamp)
        +prune_memory_bank(current_timestamp, max_size)
        +get_all_nodes()
        +get_node_count()
        +clear()
        +batch_retrieve_nodes(node_ids, current_timestamp)
        +batch_store_nodes(node_ids, embeddings, timestamp)
    }

    class propagate_between_snapshots {
        +propagate_between_snapshots(previous_snapshot, current_snapshot, memory_bank)
    }

    class handle_empty_snapshot {
        +handle_empty_snapshot(current_timestamp, memory_bank, sample_size)
    }

    class initialize_new_node {
        +initialize_new_node(node_features)
    }

    MemoryBank -- propagate_between_snapshots : used by >
    MemoryBank -- handle_empty_snapshot : used by >
    MemoryBank -- initialize_new_node : used with >
```

```mermaid
flowchart TB
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef memory fill:#e6d0ff,stroke:#8a2be2,stroke-width:2px,color:black
    classDef output fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    
    %% Memory operations
    subgraph MemoryOperations[Memory Bank Operations]
        direction TB
        
        %% Store operation
        NodeEmbed([Node Embedding]):::data --> StoreNode{Store Node}:::process
        StoreNode --> UpdateTimestamp{Update Timestamp}:::process
        UpdateTimestamp --> CheckExisting{Node Exists?}:::process
        
        CheckExisting -->|Yes| UpdateExisting{Update Existing}:::process
        CheckExisting -->|No| CreateNew{Create New Entry}:::process
        
        UpdateExisting --> CheckSize{Check Memory Size}:::process
        CreateNew --> CheckSize
        
        CheckSize -->|Size > Threshold| PruneMemory{Prune Memory}:::process
        CheckSize -->|Size <= Threshold| MemoryState([Updated Memory State]):::memory
        PruneMemory --> MemoryState
        
        %% Retrieve operation
        NodeID([Node ID]):::data --> RetrieveNode{Retrieve Node}:::process
        RetrieveNode --> CheckNode{Node Exists?}:::process
        
        CheckNode -->|No| ReturnNull([Return None]):::output
        CheckNode -->|Yes| GetEmbedding{Get Embedding}:::process
        
        GetEmbedding --> ApplyDecay{Apply Time Decay}:::process
        ApplyDecay --> UpdateAccess{Update Access Count}:::process
        UpdateAccess --> ReturnEmbedding([Return Embedding]):::output
        
        %% Prune operation
        PruneMemory --> CalculateScores{Calculate Importance Scores}:::process
        CalculateScores --> SortNodes{Sort by Importance}:::process
        SortNodes --> KeepTopNodes{Keep Top Nodes}:::process
        KeepTopNodes --> RemoveRest{Remove Rest}:::process
        RemoveRest --> GarbageCollection{Garbage Collection}:::process
        GarbageCollection --> PrunedMemory([Pruned Memory]):::memory
    end
    
    %% Propagation between snapshots
    subgraph PropagationProcess[Propagation Between Snapshots]
        direction TB
        
        PrevSnapshot([Previous Snapshot]):::data --> GetPrevActive{Get Active Nodes}:::process
        CurrSnapshot([Current Snapshot]):::data --> GetCurrActive{Get Active Nodes}:::process
        
        GetPrevActive --> FindInactive{Find Newly Inactive}:::process
        GetCurrActive --> FindInactive
        
        GetPrevActive --> FindActive{Find Newly Active}:::process
        GetCurrActive --> FindActive
        
        FindInactive --> StoreInactive{Store Inactive in Memory}:::process
        FindActive --> RetrieveActive{Retrieve from Memory}:::process
        
        StoreInactive --> UpdatedMemory([Updated Memory]):::memory
        RetrieveActive --> PropagatedState([Propagated State]):::output
    end
    
    %% Empty snapshot handling
    subgraph EmptySnapshotHandling[Empty Snapshot Handling]
        direction TB
        
        EmptyWindow([Empty Time Window]):::data --> GetAllNodes{Get All Nodes from Memory}:::process
        GetAllNodes --> CheckMemoryEmpty{Memory Empty?}:::process
        
        CheckMemoryEmpty -->|Yes| ReturnEmptySnapshot([Return Empty Snapshot]):::output
        CheckMemoryEmpty -->|No| SampleNodes{Sample Nodes}:::process
        
        SampleNodes --> CreateSyntheticSnapshot{Create Synthetic Snapshot}:::process
        CreateSyntheticSnapshot --> SyntheticSnapshot([Synthetic Snapshot]):::output
    end
```

The MemoryBank component efficiently stores and retrieves node embeddings, implementing time-decay for long-stored embeddings and pruning mechanisms to prevent memory overflow. It's a critical component for maintaining node states between active periods.

## 4. Graph Attention Network Component (SnapshotGAT)

```mermaid
classDiagram
    class GraphAttentionLayer {
        +int in_features
        +int out_features
        +float dropout
        +float alpha
        +bool concat
        +Parameter W
        +Parameter a
        +LeakyReLU leakyrelu
        +__init__(in_features, out_features, dropout, alpha, concat)
        +forward(input, adj)
    }

    class SnapshotGAT {
        +int input_dim
        +int hidden_dim
        +int output_dim
        +int num_heads
        +float dropout
        +ModuleList attentions
        +GraphAttentionLayer out_att
        +Dropout dropout
        +__init__(input_dim, hidden_dim, output_dim, num_heads, dropout, alpha)
        +forward(features, adj)
        +masked_forward(features, adj, mask)
    }

    GraphAttentionLayer <-- SnapshotGAT : contains
```

```mermaid
flowchart TB
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef attention fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    classDef output fill:#e6d0ff,stroke:#8a2be2,stroke-width:2px,color:black
    
    %% GAT Layer
    subgraph GraphAttentionLayer[Graph Attention Layer]
        direction TB
        
        InputFeatures([Input Features]):::data --> LinearTransform{Linear Transformation}:::process
        AdjMatrix([Adjacency Matrix]):::data --> AttentionMask{Create Attention Mask}:::process
        
        LinearTransform --> PrepareAttention{Prepare for Attention}:::process
        PrepareAttention --> ComputeCoefficients{Compute Attention Coefficients}:::process
        
        ComputeCoefficients --> ApplyMask{Apply Mask}:::process
        AttentionMask --> ApplyMask
        
        ApplyMask --> Softmax{Apply Softmax}:::process
        Softmax --> ApplyDropout{Apply Dropout}:::process
        
        ApplyDropout --> WeightedAggregation{Weighted Aggregation}:::process
        LinearTransform --> WeightedAggregation
        
        WeightedAggregation --> ApplyActivation{Apply Activation}:::process
        ApplyActivation --> OutputFeatures([Output Features]):::output
    end
    
    %% SnapshotGAT
    subgraph SnapshotGAT[Snapshot GAT]
        direction TB
        
        Features([Node Features]):::data --> ApplyInputDropout{Apply Dropout}:::process
        SymmetricMask([Symmetric Mask]):::data --> MultiHeadAttention{Multi-Head Attention}:::attention
        
        ApplyInputDropout --> MultiHeadAttention
        
        MultiHeadAttention --> ConcatenateHeads{Concatenate Heads}:::process
        ConcatenateHeads --> ApplyHiddenDropout{Apply Dropout}:::process
        
        ApplyHiddenDropout --> OutputAttention{Output Attention Layer}:::attention
        SymmetricMask --> OutputAttention
        
        OutputAttention --> NodeEmbeddings([Node Embeddings]):::output
    end
    
    %% Masked Forward
    subgraph MaskedForward[Masked Forward Process]
        direction TB
        
        InputFeatures2([Input Features]):::data --> ApplyDropout2{Apply Dropout}:::process
        Mask([Attention Mask]):::data --> ProcessHeads{Process Each Head}:::process
        
        ApplyDropout2 --> ProcessHeads
        
        ProcessHeads --> HeadOutputs{Head Outputs}:::process
        HeadOutputs --> ConcatenateOutputs{Concatenate Outputs}:::process
        ConcatenateOutputs --> FinalDropout{Apply Dropout}:::process
        
        FinalDropout --> OutputLayer{Output Layer}:::process
        Mask --> OutputLayer
        
        OutputLayer --> FinalEmbeddings([Final Embeddings]):::output
    end
```

The SnapshotGAT component implements a modified Graph Attention Network that handles variable-sized inputs through masking. It maintains the multi-head attention mechanism from the original GAT while adapting it for temporal graph processing.

## 5. Temporal Attention Component

```mermaid
classDiagram
    class TemporalAttention {
        +int hidden_dim
        +int num_heads
        +float dropout
        +float alpha
        +Parameter W
        +Parameter a
        +LeakyReLU leakyrelu
        +Dropout dropout_layer
        +__init__(hidden_dim, num_heads, dropout, alpha)
        +forward(sequence_embeddings)
    }
```

```mermaid
flowchart TB
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef attention fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    classDef output fill:#e6d0ff,stroke:#8a2be2,stroke-width:2px,color:black
    
    %% Temporal Attention
    subgraph TemporalAttention[Temporal Attention Process]
        direction TB
        
        SequenceEmbeddings([Sequence Embeddings]):::data --> CheckEmpty{Check if Empty}:::process
        
        CheckEmpty -->|Empty| ReturnNone([Return None]):::output
        CheckEmpty -->|Not Empty| CheckLength{Check Sequence Length}:::process
        
        CheckLength -->|Length <= 1| ReturnLast([Return Last Embedding]):::output
        CheckLength -->|Length > 1| PadEmbeddings{Create Padded Tensor}:::process
        
        PadEmbeddings --> CreateMasks{Create Masks}:::process
        CreateMasks --> ProcessHeads{Process Each Head}:::process
        
        ProcessHeads --> TransformEmbeddings{Transform Embeddings}:::process
        TransformEmbeddings --> ProcessChunks{Process in Chunks}:::process
        
        ProcessChunks --> ComputeAttention{Compute Attention}:::attention
        ComputeAttention --> ApplyAttention{Apply Attention Weights}:::process
        
        ApplyAttention --> CombineHeads{Combine Head Outputs}:::process
        CombineHeads --> FilterActive{Filter Active Nodes}:::process
        
        FilterActive --> FinalEmbeddings([Final Embeddings]):::output
    end
    
    %% Memory-optimized implementation
    subgraph ChunkProcessing[Chunk Processing]
        direction TB
        
        NodeChunk([Node Chunk]):::data --> ComputeChunkAttention{Compute Chunk Attention}:::process
        
        ComputeChunkAttention --> ProcessTimeSteps{Process Time Steps}:::process
        ProcessTimeSteps --> ComputeWeights{Compute Weights}:::attention
        
        ComputeWeights --> ApplyMasking{Apply Masking}:::process
        ApplyMasking --> ApplySoftmax{Apply Softmax}:::process
        
        ApplySoftmax --> ApplyWeights{Apply Weights to Values}:::process
        ApplyWeights --> AverageTimeSteps{Average Time Steps}:::process
        
        AverageTimeSteps --> StoreChunkResult{Store Chunk Result}:::process
        StoreChunkResult --> GarbageCollection{Garbage Collection}:::process
    end
```

The TemporalAttention component applies attention mechanisms across time to capture long-term dependencies in node embeddings. It uses a memory-optimized implementation to handle large graphs efficiently.

## 6. TempGAT Main Model

```mermaid
classDiagram
    class TempGAT {
        +int input_dim
        +int hidden_dim
        +int output_dim
        +int num_heads
        +bool use_temporal_attention
        +SnapshotGAT snapshot_gat
        +TemporalAttention temporal_attention
        +MemoryBank memory_bank
        +Linear node_init
        +Dropout dropout
        +__init__(input_dim, hidden_dim, output_dim, num_heads, memory_decay_factor, dropout, alpha, max_memory_size, temporal_attention_heads, use_temporal_attention)
        +forward(temporal_graph, snapshot_sequence, return_embeddings)
        +predict(temporal_graph, snapshot_sequence, task)
    }

    TempGAT --> SnapshotGAT : contains
    TempGAT --> TemporalAttention : contains
    TempGAT --> MemoryBank : contains
```

```mermaid
flowchart TB
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef memory fill:#e6d0ff,stroke:#8a2be2,stroke-width:2px,color:black
    classDef attention fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    classDef output fill:#ffcccc,stroke:#ff6666,stroke-width:2px,color:black
    
    %% TempGAT Forward Pass
    subgraph TempGATForward[TempGAT Forward Pass]
        direction TB
        
        SnapshotSequence([Snapshot Sequence]):::data --> ProcessSnapshots{Process Each Snapshot}:::process
        
        ProcessSnapshots --> CheckEmpty{Check if Empty}:::process
        CheckEmpty -->|Empty| HandleEmpty{Handle Empty Snapshot}:::process
        CheckEmpty -->|Not Empty| GetActiveNodes{Get Active Nodes}:::process
        
        HandleEmpty --> GetActiveNodes
        
        GetActiveNodes --> GetFeatures{Get Node Features}:::process
        GetFeatures --> CreateAdjMatrix{Create Adjacency Matrix}:::process
        CreateAdjMatrix --> CreateMask{Create Symmetric Mask}:::process
        
        CreateMask --> InitializeEmbeddings{Initialize Embeddings}:::process
        
        InitializeEmbeddings --> CheckPrevious{Previous Snapshot?}:::process
        CheckPrevious -->|Yes| PropagateEmbeddings{Propagate Embeddings}:::process
        CheckPrevious -->|No| InitializeNew{Initialize New Nodes}:::process
        
        PropagateEmbeddings --> ProcessGAT{Process with GAT}:::attention
        InitializeNew --> ProcessGAT
        
        ProcessGAT --> StoreInMemory{Store in Memory}:::memory
        StoreInMemory --> SaveSnapshot{Save Snapshot Result}:::process
        SaveSnapshot --> UpdatePrevious{Update Previous Snapshot}:::process
        
        UpdatePrevious --> GarbageCollection{Garbage Collection}:::process
        GarbageCollection --> NextSnapshot{Next Snapshot}:::process
        
        NextSnapshot --> CheckTemporalAttention{Use Temporal Attention?}:::process
        CheckTemporalAttention -->|Yes| ApplyTemporalAttention{Apply Temporal Attention}:::attention
        CheckTemporalAttention -->|No| ReturnLastEmbeddings{Return Last Embeddings}:::output
        
        ApplyTemporalAttention --> UpdateLastEmbeddings{Update Last Embeddings}:::process
        UpdateLastEmbeddings --> ReturnLastEmbeddings
    end
    
    %% TempGAT Predict
    subgraph TempGATPredict[TempGAT Prediction]
        direction TB
        
        InputSequence([Input Sequence]):::data --> ForwardPass{Forward Pass}:::process
        ForwardPass --> GetEmbeddings{Get Embeddings}:::process
        
        GetEmbeddings --> CheckTask{Check Task}:::process
        
        CheckTask -->|Node Classification| ApplySoftmax{Apply Softmax}:::process
        ApplySoftmax --> NodePredictions([Node Predictions]):::output
        
        CheckTask -->|Link Prediction| CreateNodePairs{Create Node Pairs}:::process
        CreateNodePairs --> ComputeScores{Compute Edge Scores}:::process
        ComputeScores --> EdgePredictions([Edge Predictions]):::output
    end
```

## 7. Training System

```mermaid
classDiagram
    class TemporalTrainer {
        +TempGAT model
        +TemporalGraph temporal_graph
        +str device
        +Dict history
        +__init__(model, temporal_graph, device)
        +train(num_epochs, batch_size, sequence_length, learning_rate, weight_decay, val_ratio, task, patience, verbose, scheduler_type, scheduler_params)
        +evaluate(batches, task)
        +plot_training_history()
        +save_model(path)
        +load_model(path)
    }
```

```mermaid
flowchart TB
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef training fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    classDef output fill:#ffcccc,stroke:#ff6666,stroke-width:2px,color:black
    
    %% Training Process
    subgraph TrainingProcess[Training Process]
        direction TB
        
        InitializeTrainer([Initialize Trainer]):::process --> CreateOptimizer{Create Optimizer}:::process
        CreateOptimizer --> CreateScheduler{Create Scheduler}:::process
        CreateScheduler --> CreateBatches{Create Temporal Batches}:::process
        
        CreateBatches --> SplitData{Split Train/Val}:::process
        SplitData --> TrainingLoop{Training Loop}:::training
        
        TrainingLoop --> ProcessBatch{Process Each Batch}:::process
        ProcessBatch --> ProcessSequence{Process Each Sequence}:::process
        
        ProcessSequence --> ForwardPass{Forward Pass}:::process
        ForwardPass --> ComputeLoss{Compute Loss}:::process
        ComputeLoss --> Backpropagation{Backpropagation}:::process
        Backpropagation --> UpdateWeights{Update Weights}:::process
        
        UpdateWeights --> AccumulateLoss{Accumulate Loss}:::process
        AccumulateLoss --> GarbageCollection{Garbage Collection}:::process
        
        GarbageCollection --> ValidationPhase{Validation Phase}:::process
        ValidationPhase --> UpdateScheduler{Update Scheduler}:::process
        
        UpdateScheduler --> EarlyStoppingCheck{Early Stopping Check}:::process
        EarlyStoppingCheck -->|Continue| NextEpoch{Next Epoch}:::process
        EarlyStoppingCheck -->|Stop| ReturnHistory{Return History}:::output
        
        NextEpoch --> TrainingLoop
    end
    
    %% Evaluation Process
    subgraph EvaluationProcess[Evaluation Process]
        direction TB
        
        EvalBatches([Evaluation Batches]):::data --> ModelEval{Set Model to Eval}:::process
        ModelEval --> ProcessEvalBatch{Process Each Batch}:::process
        
        ProcessEvalBatch --> EvalForwardPass{Forward Pass}:::process
        EvalForwardPass --> ComputeEvalLoss{Compute Loss}:::process
        
        ComputeEvalLoss --> CollectPredictions{Collect Predictions}:::process
        CollectPredictions --> ComputeMetrics{Compute Metrics}:::process
        
        ComputeMetrics --> ReturnResults{Return Results}:::output
    end
```

The training system handles the training, validation, and evaluation of TempGAT models on temporal graph data. It supports different learning rate schedulers, early stopping, and various evaluation metrics.

## 8. Data Flow Diagram

```mermaid
flowchart LR
    %% Define node styles
    classDef data fill:#c2e0ff,stroke:#1a83ff,stroke-width:2px,color:black
    classDef process fill:#d4f4d4,stroke:#38761d,stroke-width:2px,color:black
    classDef memory fill:#e6d0ff,stroke:#8a2be2,stroke-width:2px,color:black
    classDef output fill:#ffe6cc,stroke:#ff9900,stroke-width:2px,color:black
    
    %% Data sources
    RawData([Raw Temporal Data]):::data --> TemporalGraph{TemporalGraph}:::process
    
    %% Data processing
    TemporalGraph --> Snapshots([Snapshot Sequence]):::data
    Snapshots --> BatchCreation{Create Batches}:::process
    BatchCreation --> TemporalBatches([Temporal Batches]):::data
    
    %% Model processing
    TemporalBatches --> TempGATModel{TempGAT Model}:::process
    
    %% Memory operations
    MemoryBank{Memory Bank}:::memory <--> TempGATModel
    
    %% Model components
    TempGATModel --> SnapshotGAT{Snapshot GAT}:::process
    TempGATModel --> TemporalAttention{Temporal Attention}:::process
    
    %% Outputs
    TempGATModel --> NodeEmbeddings([Node Embeddings]):::output
    NodeEmbeddings --> NodeClassification{Node Classification}:::process
    NodeEmbeddings --> LinkPrediction{Link Prediction}:::process
    
    %% Final outputs
    NodeClassification --> ClassificationResults([Classification Results]):::output
    LinkPrediction --> PredictionResults([Prediction Results]):::output
    
    %% Training
    TemporalTrainer{Temporal Trainer}:::process --> TempGATModel
    TemporalTrainer --> TrainingHistory([Training History]):::output
```

This data flow diagram shows how information moves through the entire TempGAT system, from raw temporal data to final predictions, highlighting the key processing steps and components involved.
