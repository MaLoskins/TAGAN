#!/bin/bash

# Run the entire TempGAT pipeline: generate data, preprocess, and run model

# Create directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results

# Default parameters
NUM_USERS=500
NUM_DAYS=7
AVG_DAILY_INTERACTIONS=2000
NUM_COMMUNITIES=5
WINDOW_SIZE=15
NUM_EPOCHS=20
BATCH_SIZE=64
SEQUENCE_LENGTH=5
HIDDEN_DIM=64
OUTPUT_DIM=32
NUM_HEADS=8
MEMORY_DECAY=0.9
DROPOUT=0.2
LEARNING_RATE=0.0001
TASK="node_classification"
VISUALIZE=""
SEED=42
SCHEDULER_TYPE="plateau"
SCHEDULER_FACTOR=0.5
SCHEDULER_PATIENCE=3
SCHEDULER_MIN_LR=0.000001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num_users)
      NUM_USERS="$2"
      shift 2
      ;;
    --num_days)
      NUM_DAYS="$2"
      shift 2
      ;;
    --avg_daily_interactions)
      AVG_DAILY_INTERACTIONS="$2"
      shift 2
      ;;
    --num_communities)
      NUM_COMMUNITIES="$2"
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
    --task)
      TASK="$2"
      shift 2
      ;;
    --visualize)
      VISUALIZE="--visualize"
      shift
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --scheduler_type)
      SCHEDULER_TYPE="$2"
      shift 2
      ;;
    --scheduler_factor)
      SCHEDULER_FACTOR="$2"
      shift 2
      ;;
    --scheduler_patience)
      SCHEDULER_PATIENCE="$2"
      shift 2
      ;;
    --scheduler_min_lr)
      SCHEDULER_MIN_LR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "=== TempGAT Pipeline ==="
echo "Parameters:"
echo "  Number of users: $NUM_USERS"
echo "  Number of days: $NUM_DAYS"
echo "  Average daily interactions: $AVG_DAILY_INTERACTIONS"
echo "  Number of communities: $NUM_COMMUNITIES"
echo "  Window size (minutes): $WINDOW_SIZE"
echo "  Number of epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Sequence length: $SEQUENCE_LENGTH"
echo "  Task: $TASK"
echo "  Seed: $SEED"
echo "  Scheduler type: $SCHEDULER_TYPE"
echo "  Scheduler factor: $SCHEDULER_FACTOR"
echo "  Scheduler patience: $SCHEDULER_PATIENCE"
echo "  Scheduler min LR: $SCHEDULER_MIN_LR"
echo ""

# Step 1: Generate dataset
echo "=== Step 1: Generating dataset ==="
python generate_dataset.py \
  --num_users $NUM_USERS \
  --num_days $NUM_DAYS \
  --avg_daily_interactions $AVG_DAILY_INTERACTIONS \
  --num_communities $NUM_COMMUNITIES \
  --output_dir data/raw \
  --seed $SEED

# Check if generation was successful
if [ $? -ne 0 ]; then
  echo "Error generating dataset. Exiting."
  exit 1
fi

echo ""

# Step 2: Preprocess dataset
echo "=== Step 2: Preprocessing dataset ==="
python preprocess_dataset.py \
  --raw_data_dir data/raw \
  --processed_data_dir data/processed \
  --window_size $WINDOW_SIZE

# Check if preprocessing was successful
if [ $? -ne 0 ]; then
  echo "Error preprocessing dataset. Exiting."
  exit 1
fi

echo ""

# Step 3: Run TempGAT model
echo "=== Step 3: Running TempGAT model ==="
python run_tempgat_on_social_data.py \
  --data_path data/processed/temporal_graph_data_${WINDOW_SIZE}min.pkl \
  --output_dir results \
  --hidden_dim $HIDDEN_DIM \
  --output_dim $OUTPUT_DIM \
  --num_heads $NUM_HEADS \
  --memory_decay $MEMORY_DECAY \
  --dropout $DROPOUT \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --sequence_length $SEQUENCE_LENGTH \
  --learning_rate $LEARNING_RATE \
  --task $TASK \
  --scheduler_type $SCHEDULER_TYPE \
  --scheduler_factor $SCHEDULER_FACTOR \
  --scheduler_patience $SCHEDULER_PATIENCE \
  --scheduler_min_lr $SCHEDULER_MIN_LR \
  $VISUALIZE \
  --seed $SEED

# Check if model run was successful
if [ $? -ne 0 ]; then
  echo "Error running TempGAT model. Exiting."
  exit 1
fi

echo ""
echo "=== Pipeline completed successfully ==="
echo "Results saved to the 'results' directory"