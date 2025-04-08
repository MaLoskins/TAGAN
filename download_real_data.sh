#!/bin/bash

# Script to download real-world temporal graph datasets for TempGAT testing
# This script downloads the EU Email Communication Network dataset from SNAP

# Create directories
mkdir -p data/real_world/raw
mkdir -p data/real_world/processed

# Set dataset choice
DATASET=${1:-"email-eu"}  # Default to email-eu if no argument provided

echo "=== Downloading $DATASET dataset ==="

case $DATASET in
  "email-eu")
    echo "Downloading EU Email Communication Network dataset..."
    
    # Try different download methods (for Windows/Linux compatibility)
    if command -v wget &> /dev/null; then
        wget -O data/real_world/raw/email-Eu-core-temporal.txt.gz https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
    elif command -v curl &> /dev/null; then
        curl -L -o data/real_world/raw/email-Eu-core-temporal.txt.gz https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
    else
        # PowerShell method for Windows
        powershell -Command "Invoke-WebRequest -Uri 'https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz' -OutFile 'data/real_world/raw/email-Eu-core-temporal.txt.gz'"
    fi
    
    echo "Extracting dataset..."
    # Try different extraction methods
    if command -v gunzip &> /dev/null; then
        gunzip -f data/real_world/raw/email-Eu-core-temporal.txt.gz
    else
        # PowerShell method for Windows
        powershell -Command "Expand-Archive -Path 'data/real_world/raw/email-Eu-core-temporal.txt.gz' -DestinationPath 'data/real_world/raw/' -Force"
    fi
    
    echo "Dataset info:"
    echo "- Temporal network of email communications in a European research institution"
    echo "- 986 nodes (users) and ~332K temporal edges (emails)"
    echo "- Each line represents an email: sender receiver timestamp"
    echo "- Timestamps are in seconds since epoch"
    ;;
    
  "reddit")
    echo "Downloading Reddit Hyperlink Network dataset..."
    
    # Try different download methods (for Windows/Linux compatibility)
    if command -v wget &> /dev/null; then
        wget -O data/real_world/raw/soc-redditHyperlinks-title.tsv.gz https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv.gz
    elif command -v curl &> /dev/null; then
        curl -L -o data/real_world/raw/soc-redditHyperlinks-title.tsv.gz https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv.gz
    else
        # PowerShell method for Windows
        powershell -Command "Invoke-WebRequest -Uri 'https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv.gz' -OutFile 'data/real_world/raw/soc-redditHyperlinks-title.tsv.gz'"
    fi
    
    echo "Extracting dataset..."
    # Try different extraction methods
    if command -v gunzip &> /dev/null; then
        gunzip -f data/real_world/raw/soc-redditHyperlinks-title.tsv.gz
    else
        # PowerShell method for Windows
        powershell -Command "Expand-Archive -Path 'data/real_world/raw/soc-redditHyperlinks-title.tsv.gz' -DestinationPath 'data/real_world/raw/' -Force"
    fi
    
    echo "Dataset info:"
    echo "- Temporal network of hyperlinks between subreddits"
    echo "- ~35K nodes (subreddits) and ~860K temporal edges (hyperlinks)"
    echo "- Each line represents a hyperlink with timestamp and additional features"
    ;;
    
  "bitcoin")
    echo "Downloading Bitcoin OTC Trust Network dataset..."
    
    # Try different download methods (for Windows/Linux compatibility)
    if command -v wget &> /dev/null; then
        wget -O data/real_world/raw/soc-sign-bitcoinotc.csv.gz https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
    elif command -v curl &> /dev/null; then
        curl -L -o data/real_world/raw/soc-sign-bitcoinotc.csv.gz https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
    else
        # PowerShell method for Windows
        powershell -Command "Invoke-WebRequest -Uri 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz' -OutFile 'data/real_world/raw/soc-sign-bitcoinotc.csv.gz'"
    fi
    
    echo "Extracting dataset..."
    # Try different extraction methods
    if command -v gunzip &> /dev/null; then
        gunzip -f data/real_world/raw/soc-sign-bitcoinotc.csv.gz
    else
        # PowerShell method for Windows
        powershell -Command "Expand-Archive -Path 'data/real_world/raw/soc-sign-bitcoinotc.csv.gz' -DestinationPath 'data/real_world/raw/' -Force"
    fi
    
    echo "Dataset info:"
    echo "- Temporal network of Bitcoin users' trust ratings"
    echo "- 5,881 nodes (users) and 35,592 temporal edges (ratings)"
    echo "- Each line represents a rating: source target rating timestamp"
    ;;
    
  *)
    echo "Unknown dataset: $DATASET"
    echo "Available datasets: email-eu, reddit, bitcoin"
    exit 1
    ;;
esac

echo "Download completed. Data saved to data/real_world/raw/"
echo "Next, run: python process_real_data.py --dataset $DATASET"