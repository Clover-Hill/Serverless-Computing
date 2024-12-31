from flask import Flask, request, jsonify
import time
import os
import json
from collections import defaultdict
import oss2
import tracemalloc

app = Flask(__name__)

@app.route('/', methods=['POST'])
def reduce_task():
    # Collect usage
    tracemalloc.start()
    data = request.get_json()
    
    index = data.get('index', 0)
    assigned_files = data.get('assigned_files', [])
    oss_bucket = data.get('oss_bucket_name')
    oss_key = data.get('oss_bucket_key')
    oss_secret = data.get('oss_bucket_secret')
    
    if not all([oss_bucket, oss_key, oss_secret]):
        return jsonify({'error': 'OSS credentials are missing.'}), 400
    
    # Communication: Download files from OSS
    start_oss = time.time()
    contents = []
    for filename in assigned_files:
        try:
            # Create auth and bucket objects
            auth = oss2.Auth(oss_key, oss_secret)
            endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
            region = "cn-hangzhou"
            bucket = oss2.Bucket(auth, endpoint, oss_bucket, region=region)
            
            # Download and decode content
            file_obj = bucket.get_object(filename)
            content = file_obj.read().decode('utf-8')
            if content:
                contents.append(content)
        except Exception as e:
            return jsonify({'error': f'Failed to download {filename}: {e}'}), 500
    end_oss = time.time()
    oss_time = end_oss - start_oss

    # Execution: Aggregate word frequencies
    start_exec = time.time()
    aggregated_freq = defaultdict(int)
    for content in contents:
        word_freq = json.loads(content)
        for word, count in word_freq.items():
            aggregated_freq[word] += count
    end_exec = time.time()
    execution_time = end_exec - start_exec
    
    # Collect memory and CPU usage
    current, peak = tracemalloc.get_traced_memory()
    memory = peak / (1024 * 1024)  # Convert to MB
    tracemalloc.stop()
    
    response = {
        'execution_time': execution_time,
        'oss_time': oss_time,
        'memory': memory,
        'result': aggregated_freq
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
