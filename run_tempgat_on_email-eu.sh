#!/bin/bash

# Script to run TempGAT on the email-eu dataset

# Create directories
mkdir -p results/email-eu

# Preprocess the dataset
echo "=== Preprocessing email-eu dataset ==="
python preprocess_dataset.py \
  --raw_data_dir data/real_world/processed \
  --processed_data_dir data/real_world/processed/email-eu \
  --window_size 4630

# Run TempGAT
echo "=== Running TempGAT on email-eu dataset ==="
python run_tempgat_on_social_data.py \
  --data_path data/real_world/processed/email-eu/temporal_graph_data_4630min.pkl \
  --output_dir results/email-eu \
  --hidden_dim 64 \
  --output_dim 32 \
  --num_heads 8 \
  --memory_decay 0.9 \
  --dropout 0.2 \
  --num_epochs 5 \
  --batch_size 16 \
  --sequence_length 10 \
  --learning_rate 0.001 \
  --scheduler_type plateau \
  --scheduler_factor 0.5 \
  --scheduler_patience 3 \
  --scheduler_min_lr 0.0001 \
  --task node_classification \
  --visualize \
  --seed 42

echo "=== TempGAT run completed ==="
echo "Results saved to results/email-eu"
