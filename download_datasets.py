import os
import argparse
import requests
import gzip
import shutil
from tqdm import tqdm

def download_file(url, local_path):
    """Download a file from a URL with progress bar."""
    print(f"Downloading {url} to {local_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Download completed: {local_path}")
    return local_path

def extract_gzip(gzip_path, output_path):
    """Extract a gzip file."""
    print(f"Extracting {gzip_path} to {output_path}...")
    
    try:
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extraction completed: {output_path}")
        return True
    except Exception as e:
        print(f"Error extracting {gzip_path}: {e}")
        return False

def download_email_eu_dataset(output_dir):
    """Download and extract the EU Email Communication Network dataset."""
    url = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
    gzip_path = os.path.join(output_dir, "email-Eu-core-temporal.txt.gz")
    output_path = os.path.join(output_dir, "email-Eu-core-temporal.txt")
    
    download_file(url, gzip_path)
    success = extract_gzip(gzip_path, output_path)
    
    if success:
        print("\nDataset info:")
        print("- Temporal network of email communications in a European research institution")
        print("- 986 nodes (users) and ~332K temporal edges (emails)")
        print("- Each line represents an email: sender receiver timestamp")
        print("- Timestamps are in seconds since epoch")
    
    return success

def download_reddit_dataset(output_dir):
    """Download and extract the Reddit Hyperlink Network dataset."""
    url = "https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv.gz"
    gzip_path = os.path.join(output_dir, "soc-redditHyperlinks-title.tsv.gz")
    output_path = os.path.join(output_dir, "soc-redditHyperlinks-title.tsv")
    
    download_file(url, gzip_path)
    success = extract_gzip(gzip_path, output_path)
    
    if success:
        print("\nDataset info:")
        print("- Temporal network of hyperlinks between subreddits")
        print("- ~35K nodes (subreddits) and ~860K temporal edges (hyperlinks)")
        print("- Each line represents a hyperlink with timestamp and additional features")
    
    return success

def download_bitcoin_dataset(output_dir):
    """Download and extract the Bitcoin OTC Trust Network dataset."""
    url = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
    gzip_path = os.path.join(output_dir, "soc-sign-bitcoinotc.csv.gz")
    output_path = os.path.join(output_dir, "soc-sign-bitcoinotc.csv")
    
    download_file(url, gzip_path)
    success = extract_gzip(gzip_path, output_path)
    
    if success:
        print("\nDataset info:")
        print("- Temporal network of Bitcoin users' trust ratings")
        print("- 5,881 nodes (users) and 35,592 temporal edges (ratings)")
        print("- Each line represents a rating: source target rating timestamp")
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Download datasets for TempGAT')
    parser.add_argument('--dataset', type=str, default='email-eu',
                        choices=['email-eu', 'reddit', 'bitcoin', 'all'],
                        help='Dataset to download')
    parser.add_argument('--output_dir', type=str, default='data/real_world/raw',
                        help='Directory to save downloaded data')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == 'all':
        print("=== Downloading all datasets ===")
        success_email = download_email_eu_dataset(args.output_dir)
        success_reddit = download_reddit_dataset(args.output_dir)
        success_bitcoin = download_bitcoin_dataset(args.output_dir)
        
        if success_email and success_reddit and success_bitcoin:
            print("\nAll datasets downloaded and extracted successfully!")
        else:
            print("\nSome datasets failed to download or extract. Please check the error messages above.")
    
    elif args.dataset == 'email-eu':
        print("=== Downloading EU Email Communication Network dataset ===")
        success = download_email_eu_dataset(args.output_dir)
        
        if success:
            print("\nDataset downloaded and extracted successfully!")
        else:
            print("\nFailed to download or extract the dataset. Please check the error messages above.")
    
    elif args.dataset == 'reddit':
        print("=== Downloading Reddit Hyperlink Network dataset ===")
        success = download_reddit_dataset(args.output_dir)
        
        if success:
            print("\nDataset downloaded and extracted successfully!")
        else:
            print("\nFailed to download or extract the dataset. Please check the error messages above.")
    
    elif args.dataset == 'bitcoin':
        print("=== Downloading Bitcoin OTC Trust Network dataset ===")
        success = download_bitcoin_dataset(args.output_dir)
        
        if success:
            print("\nDataset downloaded and extracted successfully!")
        else:
            print("\nFailed to download or extract the dataset. Please check the error messages above.")
    
    print("\nNext step: python process_real_data.py --dataset", args.dataset)

if __name__ == '__main__':
    main()