# Testing TempGAT with Real-World Datasets

This guide explains how to download, process, and run the TempGAT model on real-world temporal graph datasets.

## Available Datasets

1. **EU Email Communication Network** (`email-eu`)
   - Temporal network of email communications in a European research institution
   - 986 nodes (users) and ~332K temporal edges (emails)
   - Source: [SNAP](https://snap.stanford.edu/data/email-Eu-core-temporal.html)

2. **Reddit Hyperlink Network** (`reddit`)
   - Temporal network of hyperlinks between subreddits
   - ~35K nodes (subreddits) and ~860K temporal edges (hyperlinks)
   - Source: [SNAP](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

3. **Bitcoin OTC Trust Network** (`bitcoin`)
   - Temporal network of Bitcoin users' trust ratings
   - 5,881 nodes (users) and 35,592 temporal edges (ratings)
   - Source: [SNAP](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)

## Prerequisites

1. Install the required packages:

```bash
pip install -r requirements_real_data.txt
```

2. **Windows Users**: The scripts use PowerShell commands as fallbacks when Unix tools aren't available. No additional setup is needed.

3. **Linux/Mac Users**: The scripts use standard Unix tools like wget and gunzip. If these aren't installed, you can install them with your package manager:

```bash
# Ubuntu/Debian
sudo apt-get install wget gzip

# macOS with Homebrew
brew install wget
```

## Quick Start

To download, process, and run TempGAT on a real-world dataset in one go:

### Bash Script (Linux/Mac/WSL):

```bash
bash run_real_data_pipeline.sh --dataset email-eu
```

### Python Script (Windows/Linux/Mac):

```bash
python run_real_data_pipeline.py --dataset email-eu
```

This will:
1. Download the EU Email Communication Network dataset
2. Process it into the format required by TempGAT
3. Run TempGAT on the processed dataset

The Python script is recommended for Windows users.

## Step-by-Step Usage

If you prefer to run each step separately:

### 1. Download a dataset

You can download datasets using either the shell script or the Python script:

#### Shell Script (Linux/Mac/WSL):

```bash
bash download_real_data.sh email-eu
```

#### Python Script (Windows/Linux/Mac):

```bash
python download_datasets.py --dataset email-eu
```

Replace `email-eu` with `reddit` or `bitcoin` for other datasets. The Python script is recommended for Windows users.

### 2. Process the dataset

```bash
python process_real_data.py --dataset email-eu
```

This will:
- Convert the raw data to the format required by TempGAT
- Detect communities using the Louvain method
- Generate synthetic features for nodes
- Create a run script for the specific dataset

### 3. Run TempGAT on the processed dataset

```bash
bash run_tempgat_on_email-eu.sh
```

## Options

### Sample Size

For large datasets, you can process a sample:

```bash
bash run_real_data_pipeline.sh --dataset reddit --sample_size 10000
```

### Skip Steps

You can skip specific steps if you've already completed them:

```bash
bash run_real_data_pipeline.sh --dataset email-eu --skip-download --skip-processing
```

### Visualization

To disable visualization:

```bash
bash run_real_data_pipeline.sh --dataset email-eu --no-visualize
```

## Handling Large Datasets

The Reddit dataset is quite large. For better performance:

1. Use sampling:
   ```bash
   bash run_real_data_pipeline.sh --dataset reddit --sample_size 50000
   ```

2. Adjust window size in the generated run script:
   ```bash
   # Edit run_tempgat_on_reddit.sh
   # Change --window_size to a larger value (e.g., 60 minutes)
   ```

3. Reduce batch size and sequence length:
   ```bash
   # Edit run_tempgat_on_reddit.sh
   # Change --batch_size to 4
   # Change --sequence_length to 3
   ```

## Troubleshooting

1. **Memory Issues**: If you encounter memory errors, try:
   - Reducing the sample size
   - Increasing the window size
   - Reducing batch size and sequence length

2. **Runtime Issues**: For faster processing:
   - Reduce the number of epochs
   - Use a smaller dataset (email-eu is the smallest)
   - Skip visualization with `--no-visualize`

3. **Download Issues**: If downloads fail, manually download the datasets from the SNAP links and place them in `data/real_world/raw/`.

4. **Windows-Specific Issues**:
   - If PowerShell extraction fails with `.gz` files, manually download and extract the files using 7-Zip or a similar tool
   - If you see `Expand-Archive : The path specified for the extract operation is not a valid file path`, this is because PowerShell's `Expand-Archive` doesn't natively support `.gz` files. You can:
     1. Install 7-Zip
     2. Manually download the files from the SNAP links
     3. Extract them using 7-Zip
     4. Place the extracted files in `data/real_world/raw/`
   - Alternatively, install the Windows Subsystem for Linux (WSL) for a more Linux-like environment

5. **Manual Download Links**:
   - EU Email: https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz
   - Reddit: https://snap.stanford.edu/data/soc-redditHyperlinks-title.tsv.gz
   - Bitcoin: https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz