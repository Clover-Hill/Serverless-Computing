# Serverless-Computing

Serverless Computing Lab for Cloud Computing Course

- By Maximus Cao

## Feature

- [x] MapReduce w/o Shuffle Implementation
- [x] MapReduce w/ Shuffle Implementation
- [x] Five different split strategies
- [x] Comprehensive Performance Analysis
- [x] Wikitext Dataset pre-processing

## Project Structure

- `dataset/`: contains the wikitext dataset
- `output/`: contains the metrics of the experiments
- `scripts/`: contains the scripts for running the experiments
- `serverless_fn/`: contains the serverless functions used on Alibaba Cloud Function Compute
- `run_sort.py`: the main entry point for running the MapReduce w/ Shuffle Implementation experiment
- `run_no_sort.py`: the main entry point for running the MapReduce w/o Shuffle Implementation experiment
- `split_strategy/`: contains the split strategies