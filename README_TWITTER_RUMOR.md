# Twitter Rumor Analysis with TempGAT

This guide explains how to use the TempGAT model to analyze Twitter rumor datasets, which can be particularly useful for rumor detection and propagation analysis.

## Available Datasets

1. **PHEME Dataset** (`pheme`)
   - Collection of rumors and non-rumors from Twitter during five breaking news events
   - Events: Charlie Hebdo, Ferguson, Germanwings crash, Ottawa shooting, Sydney siege
   - ~6,000 tweets in ~300 conversation threads
   - Source: [PHEME](https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619)

2. **Twitter15/16 Dataset** (`twitter15`)
   - Tweets collected during 2015 and 2016 with rumor annotations
   - ~2,000 rumor events with ~1,500 source tweets
   - Source: [Twitter15/16](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1)

3. **RumourEval Dataset** (`rumoureval`)
   - Dataset from SemEval-2019 Task 7 for rumor verification
   - Includes stance and veracity annotations
   - Source: [RumourEval](https://figshare.com/articles/dataset/RumourEval_2019_data/8845580)

## Prerequisites

Install the required packages:

```bash
pip install -r requirements_real_data.txt
```

## Quick Start

To download, process, and run TempGAT on a Twitter rumor dataset in one go:

### Bash Script (Linux/Mac/WSL):

```bash
bash run_twitter_rumor_pipeline.sh --dataset pheme
```

### Python Script (Windows/Linux/Mac):

```bash
python run_twitter_rumor_pipeline.py --dataset pheme
```

This will:
1. Download the PHEME rumor dataset
2. Process it into the format required by TempGAT
3. Run TempGAT on the processed dataset

The Python script is recommended for Windows users.

## Step-by-Step Usage

If you prefer to run each step separately:

### 1. Download and Process a Dataset

```bash
python download_twitter_rumor.py --dataset pheme
```

This will:
- Download the specified dataset
- Extract the files
- Convert the data to the format required by TempGAT (users.csv and interactions.csv)

### 2. Preprocess the Dataset

```bash
python preprocess_dataset.py --raw_data_dir data/twitter_rumor/processed --processed_data_dir data/twitter_rumor/processed --window_size 15
```

This will:
- Create temporal snapshots from the interaction data
- Save the processed data as a pickle file

### 3. Run TempGAT on the Processed Dataset

```bash
python run_tempgat_on_social_data.py --data_path data/twitter_rumor/processed/temporal_graph_data_15min.pkl --output_dir results/twitter_rumor
```

## Options

### Dataset Selection

Choose which Twitter rumor dataset to use:

```bash
python run_twitter_rumor_pipeline.py --dataset pheme
```

Available datasets: `pheme`, `twitter15`, `rumoureval`

### Window Size

Adjust the temporal window size (in minutes):

```bash
python run_twitter_rumor_pipeline.py --dataset pheme --window_size 30
```

### Training Parameters

Customize the training process:

```bash
python run_twitter_rumor_pipeline.py --dataset pheme --num_epochs 50 --batch_size 16 --sequence_length 10
```

### Skip Steps

Skip specific steps if you've already completed them:

```bash
python run_twitter_rumor_pipeline.py --dataset pheme --skip-download --skip-processing
```

## Rumor Analysis

The TempGAT model can be used for several rumor analysis tasks:

1. **Rumor Detection**: Classify nodes (users) as rumor spreaders or non-rumor spreaders
2. **Rumor Propagation**: Analyze how rumors spread through the network over time
3. **User Influence**: Identify influential users in rumor propagation
4. **Temporal Patterns**: Discover temporal patterns in rumor spreading

## Dataset Structure

The processed Twitter rumor datasets are structured as follows:

1. **users.csv**:
   - `user_id`: Twitter user ID
   - `community_id`: 1 for rumor spreaders, 0 for non-rumor spreaders
   - `event`: The news event (e.g., charliehebdo, ferguson)
   - `feature_*`: User features derived from Twitter metadata

2. **interactions.csv**:
   - `source_id`: Source user ID
   - `target_id`: Target user ID
   - `timestamp`: Interaction timestamp
   - `thread_id`: Conversation thread ID
   - `event`: The news event
   - `label`: 'rumour' or 'non-rumour'

## Customizing for Your Research

If you're working on rumor analysis for your thesis, you can customize the pipeline:

1. **Feature Engineering**: Modify the `generate_user_features` function in `download_twitter_rumor.py` to create more sophisticated features

2. **Community Detection**: Adjust the community labeling in the processing script to focus on specific aspects of rumor spreading

3. **Temporal Analysis**: Experiment with different window sizes to capture different temporal dynamics

4. **Model Parameters**: Tune the TempGAT parameters (e.g., number of heads, hidden dimensions) for optimal performance

## Troubleshooting

1. **Download Issues**: If downloads fail, manually download the datasets from the provided links and place them in `data/twitter_rumor/raw/`

2. **Memory Issues**: For large datasets, try:
   - Reducing the batch size (`--batch_size 4`)
   - Increasing the window size (`--window_size 60`)
   - Processing only a subset of the data

3. **Windows-Specific Issues**: Use the Python script version (`run_twitter_rumor_pipeline.py`) which is more compatible with Windows environments

## Citation

If you use these datasets in your research, please cite the original papers:

- PHEME: Zubiaga, A., Liakata, M., Procter, R., Wong Sak Hoi, G., & Tolmie, P. (2016). Analysing how people orient to and spread rumours in social media by looking at conversational threads.

- Twitter15/16: Ma, J., Gao, W., Mitra, P., Kwon, S., Jansen, B. J., Wong, K. F., & Cha, M. (2016). Detecting rumors from microblogs with recurrent neural networks.

- RumourEval: Gorrell, G., Bontcheva, K., Derczynski, L., Kochkina, E., Liakata, M., & Zubiaga, A. (2019). SemEval-2019 Task 7: RumourEval, determining rumour veracity and support for rumours.