import argparse
import requests
import time
import json
from loguru import logger
from dotenv import load_dotenv
from collections import defaultdict
import oss2
import os
from multiprocessing import Pool

def load_text(filename):
    # Read words from file
    with open(filename, "r", encoding="utf-8") as f:
        words = f.read().split("\n")
    return words

def split_text(words, num_splits, strategy='naive'):
    if strategy == 'naive':
        split_size = len(words) // num_splits
        splits = [words[i*split_size : (i+1)*split_size] for i in range(num_splits)]
        # Handle any remaining words
        if len(words) % num_splits != 0:
            splits[-1].extend(words[num_splits*split_size:])
        return splits
    elif strategy == 'advanced':
        # Example advanced strategy: split by paragraphs or other criteria
        # For simplicity, implement naive here
        return split_text(words, num_splits, strategy='naive')
    else:
        raise ValueError("Unknown split strategy")

def map_worker(args):
    index, split = args
    start_total = time.time()
    # We need to access oss_details and map_url from the outer scope
    payload = {
        'text': split,
        'index': index,
        'oss_bucket_name': map_worker.oss_details['name'],
        'oss_bucket_key': map_worker.oss_details['key'],
        'oss_bucket_secret': map_worker.oss_details['secret']
    }
    try:
        response = requests.post(map_worker.map_url, json=payload)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Map Worker {index} completed.")
        return {
            'filename': data['filename'],
            'execution_time': data['execution_time'],
            'oss_time': data['oss_time'],
            'memory': data['memory'],
            'total_time': time.time() - start_total
        }
    except requests.RequestException as e:
        logger.info(f"Error dispatching map task {index}: {e}")
        return None

def reduce_worker(args):
    index, assigned_files = args
    start_total = time.time()
    payload = {
        'index': index,
        'assigned_files': assigned_files,
        'oss_bucket_name': reduce_worker.oss_details['name'],
        'oss_bucket_key': reduce_worker.oss_details['key'],
        'oss_bucket_secret': reduce_worker.oss_details['secret']
    }
    try:
        response = requests.post(reduce_worker.reduce_url, json=payload)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Reduce Worker {index} completed.")
        return {
            'result': data['result'],
            'execution_time': data['execution_time'],
            'oss_time': data['oss_time'],
            'memory': data['memory'],
            'total_time': time.time() - start_total
        }
    except requests.RequestException as e:
        logger.error(f"Error dispatching reduce task {index}: {e}")
        return None

def dispatch_map_tasks(map_url, splits, oss_details):
    map_results = []
    metrics = {'execution_time': [], 'oss_time': [], 'memory': [], 'total_time': []}

    # Set the required attributes on the worker function
    map_worker.map_url = map_url
    map_worker.oss_details = oss_details

    # Create process pool and dispatch tasks
    with Pool() as pool:
        results = pool.map(map_worker, enumerate(splits))

    # Collect results
    for result in results:
        if result:
            map_results.append(result['filename'])
            metrics['execution_time'].append(result['execution_time'])
            metrics['oss_time'].append(result['oss_time']) 
            metrics['memory'].append(result['memory'])
            metrics['total_time'].append(result['total_time'])

    return map_results, metrics

def dispatch_reduce_tasks(reduce_url, map_output_files, num_reduces, oss_details):
    reduce_results = []
    metrics = {'execution_time': [], 'oss_time': [], 'memory': [], 'total_time': []}
    
    # Split map_results evenly among reduce workers
    files_per_reducer = len(map_output_files) // num_reduces
    reduce_assignments = [map_output_files[i:i + files_per_reducer] for i in range(0, len(map_output_files), files_per_reducer)]
    
    # Handle any remaining files
    if len(reduce_assignments) > num_reduces:
        reduce_assignments[num_reduces-1].extend([f for sublist in reduce_assignments[num_reduces:] for f in sublist])
        reduce_assignments = reduce_assignments[:num_reduces]

    # Set the required attributes on the worker function
    reduce_worker.reduce_url = reduce_url
    reduce_worker.oss_details = oss_details

    # Create process pool and dispatch tasks
    with Pool() as pool:
        results = pool.map(reduce_worker, enumerate(reduce_assignments))

    # Collect results
    for result in results:
        if result:
            reduce_results.append(result['result'])
            metrics['execution_time'].append(result['execution_time'])
            metrics['oss_time'].append(result['oss_time'])
            metrics['memory'].append(result['memory'])
            metrics['total_time'].append(result['total_time'])

    return reduce_results, metrics

def aggregate_results(reduce_results):
    final_freq = defaultdict(int)
    for partial in reduce_results:
        for word, count in partial.items():
            final_freq[word] += count
    return final_freq

def local_word_count(words):
    freq = defaultdict(int)
    for word in words:
        freq[word] += 1
    return freq

def verify_results(local_freq, final_freq):
    assert local_freq == final_freq, "Mismatch between local and reduce results"
    logger.info("Verification successful: Final word frequency matches local computation.")

def report_metrics(map_metrics, reduce_metrics, output_file, main_execution_time, map_execution_time, reduce_execution_time):
    metrics_data = {}
    
    # Map metrics
    map_total_exec = sum(map_metrics['execution_time'])
    map_total_oss = sum(map_metrics['oss_time']) 
    map_total_time = sum(map_metrics['total_time'])
    map_num_workers = len(map_metrics['execution_time'])
    map_total_memory = sum(map_metrics['memory'])

    metrics_data['map_metrics'] = {
        'total_execution_time (s)': round(map_total_exec, 6),
        'total_oss_time (s)': round(map_total_oss, 6),
        'total_time (s)': round(map_total_time, 6),
        'avg_execution_time_per_worker (s)': round(map_total_exec/map_num_workers, 6),
        'avg_oss_time_per_worker (s)': round(map_total_oss/map_num_workers, 6),
        'avg_total_time_per_worker (s)': round(map_total_time/map_num_workers, 6),
        'execution_time_ratio (%)': round(map_total_exec/map_total_time, 6),
        'oss_time_ratio (%)': round(map_total_oss/map_total_time, 6),
        'total_memory (MB)': round(map_total_memory, 6),
        'avg_memory_per_worker (MB)': round(map_total_memory/map_num_workers, 6)
    }

    # Reduce metrics
    reduce_total_exec = sum(reduce_metrics['execution_time'])
    reduce_total_oss = sum(reduce_metrics['oss_time'])
    reduce_total_time = sum(reduce_metrics['total_time'])
    reduce_num_workers = len(reduce_metrics['execution_time'])
    reduce_total_memory = sum(reduce_metrics['memory'])

    metrics_data['reduce_metrics'] = {
        'total_execution_time (s)': round(reduce_total_exec, 6),
        'total_oss_time (s)': round(reduce_total_oss, 6),
        'total_time (s)': round(reduce_total_time, 6),
        'avg_execution_time_per_worker (s)': round(reduce_total_exec/reduce_num_workers, 6),
        'avg_oss_time_per_worker (s)': round(reduce_total_oss/reduce_num_workers, 6),
        'avg_total_time_per_worker (s)': round(reduce_total_time/reduce_num_workers, 6),
        'execution_time_ratio (%)': round(reduce_total_exec/reduce_total_time, 6),
        'oss_time_ratio (%)': round(reduce_total_oss/reduce_total_time, 6),
        'total_memory (MB)': round(reduce_total_memory, 6),
        'avg_memory_per_worker (MB)': round(reduce_total_memory/reduce_num_workers, 6)
    }

    # Combined metrics
    total_exec = map_total_exec + reduce_total_exec
    total_oss = map_total_oss + reduce_total_oss
    total_time = map_total_time + reduce_total_time
    total_memory = map_total_memory + reduce_total_memory
    total_workers = map_num_workers + reduce_num_workers

    metrics_data['combined_metrics'] = {
        'execution_time_ratio (%)': round(total_exec/total_time, 6),
        'oss_time_ratio (%)': round(total_oss/total_time, 6),
        'total_memory (MB)': round(total_memory, 6),
        'avg_memory_per_worker (MB)': round(total_memory/total_workers, 6)
    }

    metrics_data['global_metrics'] = {
        'main_execution_time (s)': round(main_execution_time, 6),
        'map_execution_time (s)': round(map_execution_time, 6),
        'reduce_execution_time (s)': round(reduce_execution_time, 6)
    }

    # Save metrics to JSON file
    with open(output_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    logger.info(f"Metrics have been saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Serverless Map-Reduce Framework')
    parser.add_argument('--text_file', type=str, help='Path to the text file', default="dataset/wikitext-train-100-words.txt")
    parser.add_argument('--map_workers', type=int, help='Number of map workers', default=10)
    parser.add_argument('--reduce_workers', type=int, help='Number of reduce workers', default=10)
    parser.add_argument('--map_function_url', type=str, help='HTTP URL of the map function', default="https://map-test-qlanljpbup.cn-hangzhou.fcapp.run")
    parser.add_argument('--reduce_function_url', type=str, help='HTTP URL of the reduce function', default="https://reduce-test-mervgqspff.cn-hangzhou.fcapp.run")
    parser.add_argument('--split_strategy', type=str, default='naive', help='Strategy to split the text file')
    parser.add_argument('--output_file', type=str, default='metrics_report.json', help='Path to the output file')
    args = parser.parse_args()
    
    # Load and split text
    logger.info("Loading text file...")
    words = load_text(args.text_file)
    logger.info(f"Total words loaded: {len(words)}")
    
    logger.info("Splitting text...")
    splits = split_text(words, args.map_workers, args.split_strategy)
    for i, split in enumerate(splits):
        logger.info(f"Map Worker {i}: {len(split)} words")
    
    # OSS Details (Assuming pre-created or passed as environment variables)
    load_dotenv()
    oss_details = {
        'name': os.getenv('OSS_BUCKET'),
        'key': os.getenv('OSS_ACCESS_KEY_ID'),
        'secret': os.getenv('OSS_ACCESS_KEY_SECRET')
    }
    
    if not oss_details['name'] or not oss_details['key'] or not oss_details['secret']:
        logger.info("OSS details are not fully provided in environment variables.")
        return
    
    # Create OSS bucket if it doesn't exist
    auth = oss2.Auth(oss_details['key'], oss_details['secret'])
    endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
    region = "cn-hangzhou"
    bucket = oss2.Bucket(auth, endpoint, oss_details['name'], region=region)

    try:
        bucket.get_bucket_info()  # Check if bucket exists
    except oss2.exceptions.NoSuchBucket:
        logger.warning("Bucket does not exist, creating...")
        try:
            bucket.create_bucket(oss2.models.BUCKET_ACL_PRIVATE)
            logger.info("Bucket created successfully")
        except oss2.exceptions.OssError as e:
            logger.error(f"Failed to create bucket: {e}")
            return
    
    # List bucket information
    logger.info(f"--------------------------------")
    logger.info("Retrieving bucket information...")
    try:
        bucket_info = bucket.get_bucket_info()
        logger.info(f"Bucket Name: {bucket_info.name}")
        logger.info(f"Creation Time: {bucket_info.creation_date}")
        logger.info(f"Storage Class: {bucket_info.storage_class}")
        logger.info(f"Location: {bucket_info.location}")
        logger.info(f"Extranet Endpoint: {bucket_info.extranet_endpoint}")
        logger.info(f"--------------------------------")
    except oss2.exceptions.OssError as e:
        logger.error(f"Failed to get bucket info: {e}")
        return

    main_start_time = time.time()
    start_time = time.time()

    logger.info("Dispatching map tasks...")
    map_results, map_metrics = dispatch_map_tasks(args.map_function_url, splits, oss_details)
    
    map_execution_time = time.time() - start_time
    start_time = time.time()
    
    logger.info("Dispatching reduce tasks...")
    reduce_results, reduce_metrics = dispatch_reduce_tasks(args.reduce_function_url, map_results, args.reduce_workers, oss_details)
    
    reduce_execution_time = time.time() - start_time
    
    logger.info("Aggregating results...")
    final_freq = aggregate_results(reduce_results)
    
    logger.info("Verifying results...")
    local_freq = local_word_count(words)
    
    try:
        verify_results(local_freq, final_freq)
    except AssertionError as e:
        logger.error(str(e))
        return
    logger.info(f"Verification successful !!!")
    
    logger.info("Reporting metrics...")

    main_end_time = time.time()
    logger.info(f"Main execution time: {main_end_time - main_start_time:.2f} seconds")
    report_metrics(map_metrics, reduce_metrics, args.output_file, main_execution_time=main_end_time - main_start_time, map_execution_time=map_execution_time, reduce_execution_time=reduce_execution_time)
    
if __name__ == '__main__':
    main()