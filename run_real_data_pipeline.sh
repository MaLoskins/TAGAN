#!/bin/bash

# Combined script to download, process, and run TempGAT on real-world datasets

# Default parameters
DATASET="email-eu"  # Options: email-eu, reddit, bitcoin
SAMPLE_SIZE=""      # Empty means use full dataset
VISUALIZE="--visualize"
SKIP_DOWNLOAD=false
SKIP_PROCESSING=false
SKIP_TRAINING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --sample_size)
      SAMPLE_SIZE="--sample_size $2"
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
    --skip-training)
      SKIP_TRAINING=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dataset email-eu|reddit|bitcoin] [--sample_size N] [--no-visualize] [--skip-download] [--skip-processing] [--skip-training]"
      exit 1
      ;;
  esac
done

echo "=== TempGAT Real-World Data Pipeline ==="
echo "Dataset: $DATASET"
if [ -n "$SAMPLE_SIZE" ]; then
  echo "Sample size: $SAMPLE_SIZE"
else
  echo "Using full dataset"
fi

# Create required directories
mkdir -p data/real_world/raw
mkdir -p data/real_world/processed
mkdir -p results/$DATASET

# Step 1: Download the dataset
if [ "$SKIP_DOWNLOAD" = false ]; then
  echo -e "\n=== Step 1: Downloading dataset ==="
  
  # Try shell script first
  bash download_real_data.sh $DATASET
  
  # Check if download was successful
  DOWNLOAD_SUCCESS=false
  if [ "$DATASET" = "email-eu" ] && [ -f "data/real_world/raw/email-Eu-core-temporal.txt" ]; then
    DOWNLOAD_SUCCESS=true
  elif [ "$DATASET" = "reddit" ] && [ -f "data/real_world/raw/soc-redditHyperlinks-title.tsv" ]; then
    DOWNLOAD_SUCCESS=true
  elif [ "$DATASET" = "bitcoin" ] && [ -f "data/real_world/raw/soc-sign-bitcoinotc.csv" ]; then
    DOWNLOAD_SUCCESS=true
  fi
  
  # If shell script failed, try Python script
  if [ "$DOWNLOAD_SUCCESS" = false ]; then
    echo "Shell script download failed. Trying Python download script..."
    python download_datasets.py --dataset $DATASET
  fi
else
  echo -e "\n=== Step 1: Skipping download ==="
fi

# Check if download was successful
if [ ! -f "data/real_world/raw/email-Eu-core-temporal.txt" ] && [ "$DATASET" = "email-eu" ] && [ "$SKIP_DOWNLOAD" = false ]; then
  echo "Warning: Dataset download or extraction failed."
  echo "Would you like to continue anyway? You may need to manually download and extract the dataset."
  echo "See README_REAL_DATA.md for manual download links."
  echo "Continue? (y/n)"
  read -r response
  if [[ "$response" != "y" ]]; then
    echo "Exiting. Please try again after manually downloading the dataset."
    exit 1
  fi
fi

if [ ! -f "data/real_world/raw/soc-redditHyperlinks-title.tsv" ] && [ "$DATASET" = "reddit" ] && [ "$SKIP_DOWNLOAD" = false ]; then
  echo "Warning: Dataset download or extraction failed."
  echo "Would you like to continue anyway? You may need to manually download and extract the dataset."
  echo "See README_REAL_DATA.md for manual download links."
  echo "Continue? (y/n)"
  read -r response
  if [[ "$response" != "y" ]]; then
    echo "Exiting. Please try again after manually downloading the dataset."
    exit 1
  fi
fi

if [ ! -f "data/real_world/raw/soc-sign-bitcoinotc.csv" ] && [ "$DATASET" = "bitcoin" ] && [ "$SKIP_DOWNLOAD" = false ]; then
  echo "Warning: Dataset download or extraction failed."
  echo "Would you like to continue anyway? You may need to manually download and extract the dataset."
  echo "See README_REAL_DATA.md for manual download links."
  echo "Continue? (y/n)"
  read -r response
  if [[ "$response" != "y" ]]; then
    echo "Exiting. Please try again after manually downloading the dataset."
    exit 1
  fi
fi

# Step 2: Process the dataset
if [ "$SKIP_PROCESSING" = false ]; then
  echo -e "\n=== Step 2: Processing dataset ==="
  python process_real_data.py --dataset $DATASET $SAMPLE_SIZE
else
  echo -e "\n=== Step 2: Skipping processing ==="
fi

# Check if processing was successful
if [ ! -f "run_tempgat_on_${DATASET}.sh" ] && [ "$SKIP_PROCESSING" = false ]; then
  echo "Error: Dataset processing failed. Please check the error messages above."
  exit 1
fi

# Step 3: Run TempGAT on the processed dataset
if [ "$SKIP_TRAINING" = false ]; then
  echo -e "\n=== Step 3: Running TempGAT on processed dataset ==="
  bash run_tempgat_on_${DATASET}.sh
else
  echo -e "\n=== Step 3: Skipping training ==="
fi

echo -e "\n=== Pipeline completed ==="
echo "Results saved to results/$DATASET"
echo ""
echo "To run just the TempGAT model again (without downloading/processing):"
echo "  bash run_tempgat_on_${DATASET}.sh"
echo ""
echo "To run the pipeline with different options:"
echo "  bash run_real_data_pipeline.sh --dataset [email-eu|reddit|bitcoin] [--sample_size N] [--no-visualize]"