#!/bin/bash

# Script to download, process, and run TempGAT on Twitter rumor datasets

# Default parameters
DATASET="pheme"  # Options: pheme, twitter15, rumoureval
WINDOW_SIZE=15
NUM_EPOCHS=20
BATCH_SIZE=8
SEQUENCE_LENGTH=5
HIDDEN_DIM=64
OUTPUT_DIM=32
NUM_HEADS=8
MEMORY_DECAY=0.9
DROPOUT=0.2
LEARNING_RATE=0.001
VISUALIZE="--visualize"
SKIP_DOWNLOAD=false
SKIP_PROCESSING=false
SKIP_PREPROCESSING=false
SKIP_TRAINING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --window_size)
      WINDOW_SIZE="$2"
      shift 2
      ;;
    --num_epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --sequence_length)
      SEQUENCE_LENGTH="$2"
      shift 2
      ;;
    --hidden_dim)
      HIDDEN_DIM="$2"
      shift 2
      ;;
    --output_dim)
      OUTPUT_DIM="$2"
      shift 2
      ;;
    --num_heads)
      NUM_HEADS="$2"
      shift 2
      ;;
    --memory_decay)
      MEMORY_DECAY="$2"
      shift 2
      ;;
    --dropout)
      DROPOUT="$2"
      shift 2
      ;;
    --learning_rate)
      LEARNING_RATE="$2"
      shift 2
      ;;
    --no-visualize)
      VISUALIZE=""
      shift
      ;;
    --skip-download)
      SKIP_DOWNLOAD=true
      shift
      ;;
    --skip-processing)
      SKIP_PROCESSING=true
      shift
      ;;
    --skip-preprocessing)
      SKIP_PREPROCESSING=true
      shift
      ;;
    --skip-training)
      SKIP_TRAINING=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset pheme|twitter15|rumoureval] [--window_size N] [--num_epochs N] [--batch_size N] [--sequence_length N] [--no-visualize] [--skip-download] [--skip-processing] [--skip-preprocessing] [--skip-training]"
      exit 1
      ;;
  esac
done

echo "=== TempGAT Twitter Rumor Pipeline ==="
echo "Dataset: $DATASET"
echo "Window size: $WINDOW_SIZE minutes"
echo "Number of epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQUENCE_LENGTH"

# Create required directories
mkdir -p data/twitter_rumor/raw
mkdir -p data/twitter_rumor/processed
mkdir -p results/twitter_rumor

# Step 1: Download and process the dataset
if [ "$SKIP_DOWNLOAD" = false ] || [ "$SKIP_PROCESSING" = false ]; then
  echo -e "\n=== Step 1: Downloading and processing dataset ==="
  
  DOWNLOAD_ARG=""
  if [ "$SKIP_DOWNLOAD" = true ]; then
    DOWNLOAD_ARG="--skip_download"
  fi
  
  PROCESSING_ARG=""
  if [ "$SKIP_PROCESSING" = true ]; then
    PROCESSING_ARG="--skip_processing"
  fi
  
  python download_twitter_rumor.py --dataset $DATASET $DOWNLOAD_ARG $PROCESSING_ARG
else
  echo -e "\n=== Step 1: Skipping download and processing ==="
fi

# Step 2: Preprocess the dataset
if [ "$SKIP_PREPROCESSING" = false ]; then
  echo -e "\n=== Step 2: Preprocessing dataset ==="
  python preprocess_dataset.py \
    --raw_data_dir data/twitter_rumor/processed \
    --processed_data_dir data/twitter_rumor/processed \
    --window_size $WINDOW_SIZE
else
  echo -e "\n=== Step 2: Skipping preprocessing ==="
fi

# Step 3: Run TempGAT on the processed dataset
if [ "$SKIP_TRAINING" = false ]; then
  echo -e "\n=== Step 3: Running TempGAT on processed dataset ==="
  python run_tempgat_on_social_data.py \
    --data_path data/twitter_rumor/processed/temporal_graph_data_${WINDOW_SIZE}min.pkl \
    --output_dir results/twitter_rumor \
    --hidden_dim $HIDDEN_DIM \
    --output_dim $OUTPUT_DIM \
    --num_heads $NUM_HEADS \
    --memory_decay $MEMORY_DECAY \
    --dropout $DROPOUT \
    --num_epochs $NUM_EPOCHS \
    --batch_size $BATCH_SIZE \
    --sequence_length $SEQUENCE_LENGTH \
    --learning_rate $LEARNING_RATE \
    --scheduler_type plateau \
    --scheduler_factor 0.5 \
    --scheduler_patience 3 \
    --scheduler_min_lr 0.0001 \
    --task node_classification \
    $VISUALIZE \
    --seed 42
else
  echo -e "\n=== Step 3: Skipping training ==="
fi

echo -e "\n=== Pipeline completed ==="
echo "Results saved to results/twitter_rumor"