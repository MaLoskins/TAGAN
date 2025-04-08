import os
import argparse
import subprocess
import sys
import time

def run_command(command, description=None):
    """Run a command and print its output in real-time."""
    if description:
        print(f"\n=== {description} ===")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    return process.returncode

def main():
    parser = argparse.ArgumentParser(description='TempGAT Twitter Rumor Pipeline (Python version)')
    parser.add_argument('--dataset', type=str, default='pheme',
                        choices=['pheme', 'twitter15', 'rumoureval'],
                        help='Dataset to use')
    parser.add_argument('--window_size', type=int, default=15,
                        help='Window size in minutes')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='Sequence length')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=32,
                        help='Output dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--memory_decay', type=float, default=0.9,
                        help='Memory decay factor')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download')
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip dataset processing')
    parser.add_argument('--skip-preprocessing', action='store_true',
                        help='Skip dataset preprocessing')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training')
    args = parser.parse_args()
    
    # Create required directories
    os.makedirs('data/twitter_rumor/raw', exist_ok=True)
    os.makedirs('data/twitter_rumor/processed', exist_ok=True)
    os.makedirs('results/twitter_rumor', exist_ok=True)
    
    print("=== TempGAT Twitter Rumor Pipeline (Python version) ===")
    print(f"Dataset: {args.dataset}")
    print(f"Window size: {args.window_size} minutes")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.sequence_length}")
    
    # Step 1: Download and process the dataset
    if not args.skip_download or not args.skip_processing:
        download_args = []
        if args.skip_download:
            download_args.append("--skip_download")
        if args.skip_processing:
            download_args.append("--skip_processing")
        
        download_cmd = f"python download_twitter_rumor.py --dataset {args.dataset} {' '.join(download_args)}"
        exit_code = run_command(download_cmd, "Step 1: Downloading and processing dataset")
    else:
        print("\n=== Step 1: Skipping download and processing ===")
    
    # Step 2: Preprocess the dataset
    if not args.skip_preprocessing:
        preprocess_cmd = (
            f"python preprocess_dataset.py "
            f"--raw_data_dir data/twitter_rumor/processed "
            f"--processed_data_dir data/twitter_rumor/processed "
            f"--window_size {args.window_size}"
        )
        exit_code = run_command(preprocess_cmd, "Step 2: Preprocessing dataset")
    else:
        print("\n=== Step 2: Skipping preprocessing ===")
    
    # Step 3: Run TempGAT on the processed dataset
    if not args.skip_training:
        visualize = "" if args.no_visualize else "--visualize"
        
        train_cmd = (
            f"python run_tempgat_on_social_data.py "
            f"--data_path data/twitter_rumor/processed/temporal_graph_data_{args.window_size}min.pkl "
            f"--output_dir results/twitter_rumor "
            f"--hidden_dim {args.hidden_dim} "
            f"--output_dim {args.output_dim} "
            f"--num_heads {args.num_heads} "
            f"--memory_decay {args.memory_decay} "
            f"--dropout {args.dropout} "
            f"--num_epochs {args.num_epochs} "
            f"--batch_size {args.batch_size} "
            f"--sequence_length {args.sequence_length} "
            f"--learning_rate {args.learning_rate} "
            f"--scheduler_type plateau "
            f"--scheduler_factor 0.5 "
            f"--scheduler_patience 3 "
            f"--scheduler_min_lr 0.0001 "
            f"--task node_classification "
            f"{visualize} "
            f"--seed 42"
        )
        exit_code = run_command(train_cmd, "Step 3: Running TempGAT on processed dataset")
    else:
        print("\n=== Step 3: Skipping training ===")
    
    print("\n=== Pipeline completed ===")
    print("Results saved to results/twitter_rumor")

if __name__ == '__main__':
    main()