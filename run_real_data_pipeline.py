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
    parser = argparse.ArgumentParser(description='TempGAT Real-World Data Pipeline (Python version)')
    parser.add_argument('--dataset', type=str, default='email-eu',
                        choices=['email-eu', 'reddit', 'bitcoin'],
                        help='Dataset to use')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of interactions to sample (for large datasets)')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download')
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip dataset processing')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training')
    args = parser.parse_args()
    
    # Create required directories
    os.makedirs('data/real_world/raw', exist_ok=True)
    os.makedirs('data/real_world/processed', exist_ok=True)
    os.makedirs(f'results/{args.dataset}', exist_ok=True)
    
    print("=== TempGAT Real-World Data Pipeline (Python version) ===")
    print(f"Dataset: {args.dataset}")
    if args.sample_size:
        print(f"Sample size: {args.sample_size}")
    else:
        print("Using full dataset")
    
    # Step 1: Download the dataset
    if not args.skip_download:
        # Use the Python download script
        sample_arg = f"--sample_size {args.sample_size}" if args.sample_size else ""
        download_cmd = f"python download_datasets.py --dataset {args.dataset} {sample_arg}"
        
        exit_code = run_command(download_cmd, "Step 1: Downloading dataset")
        
        # Check if download was successful
        success = False
        if args.dataset == 'email-eu' and os.path.exists('data/real_world/raw/email-Eu-core-temporal.txt'):
            success = True
        elif args.dataset == 'reddit' and os.path.exists('data/real_world/raw/soc-redditHyperlinks-title.tsv'):
            success = True
        elif args.dataset == 'bitcoin' and os.path.exists('data/real_world/raw/soc-sign-bitcoinotc.csv'):
            success = True
        
        if not success:
            print("Error: Dataset download failed.")
            response = input("Would you like to continue anyway? You may need to manually download and extract the dataset. (y/n): ")
            if response.lower() != 'y':
                print("Exiting. Please try again after manually downloading the dataset.")
                sys.exit(1)
    else:
        print("\n=== Step 1: Skipping download ===")
    
    # Step 2: Process the dataset
    if not args.skip_processing:
        sample_arg = f"--sample_size {args.sample_size}" if args.sample_size else ""
        process_cmd = f"python process_real_data.py --dataset {args.dataset} {sample_arg}"
        
        exit_code = run_command(process_cmd, "Step 2: Processing dataset")
        
        # Check if processing was successful
        if not os.path.exists(f"run_tempgat_on_{args.dataset}.sh"):
            print("Error: Dataset processing failed.")
            sys.exit(1)
    else:
        print("\n=== Step 2: Skipping processing ===")
    
    # Step 3: Run TempGAT on the processed dataset
    if not args.skip_training:
        # Extract parameters from the generated shell script
        params = {}
        with open(f"run_tempgat_on_{args.dataset}.sh", 'r') as f:
            for line in f:
                if 'python run_tempgat_on_social_data.py' in line:
                    # Parse the command line
                    parts = line.strip().split('\\')
                    for part in parts:
                        part = part.strip()
                        if part.startswith('--'):
                            key_value = part.split(' ', 1)
                            if len(key_value) == 2:
                                key, value = key_value
                                params[key] = value
        
        # Build the command
        train_cmd = "python run_tempgat_on_social_data.py"
        for key, value in params.items():
            if key == '--visualize' and args.no_visualize:
                continue
            train_cmd += f" {key} {value}"
        
        exit_code = run_command(train_cmd, "Step 3: Running TempGAT on processed dataset")
    else:
        print("\n=== Step 3: Skipping training ===")
    
    print("\n=== Pipeline completed ===")
    print(f"Results saved to results/{args.dataset}")
    print("")
    print("To run just the TempGAT model again (without downloading/processing):")
    print(f"  python run_tempgat_on_social_data.py --data_path data/real_world/processed/{args.dataset}/temporal_graph_data_*.pkl --output_dir results/{args.dataset}")

if __name__ == '__main__':
    main()