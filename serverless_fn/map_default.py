from flask import Flask, request, jsonify
import time
import oss2
import os
import json
from collections import defaultdict
from datetime import datetime
import tracemalloc

app = Flask(__name__)

@app.route('/', methods=['POST'])
def map_task():
    # Collect memory usage
    tracemalloc.start()
    start_total = time.time()
    data = request.get_json()
    
    text = data.get('text', '')
    index = data.get('index', 0)
    oss_bucket = data.get('oss_bucket_name')
    oss_key = data.get('oss_bucket_key')
    oss_secret = data.get('oss_bucket_secret')
    
    if not all([oss_bucket, oss_key, oss_secret]):
        return jsonify({'error': 'OSS credentials are missing.'}), 400
    
    # Execution: Compute word frequencies
    start_exec = time.time()
    word_freq = defaultdict(int)
    for word in text:
        word_freq[word] += 1
    end_exec = time.time()
    execution_time = end_exec - start_exec
    
    # Serialize word_freq
    word_freq_serialized = json.dumps(word_freq)
    
    # Communication: Upload to OSS
    start_oss = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    upload_filename = f"map_output_{index}_{timestamp}.json"
    try:
        auth = oss2.Auth(oss_key, oss_secret)
        endpoint = "https://oss-cn-hangzhou.aliyuncs.com"
        region = "cn-hangzhou"
        bucket = oss2.Bucket(auth, endpoint, oss_bucket, region=region)
        
        result = bucket.put_object(upload_filename, word_freq_serialized)
        if result.status != 200:
            raise Exception(f"Upload failed with status {result.status}")
    except Exception as e:
        return jsonify({'error': f'Failed to upload to OSS: {e}'}), 500
    end_oss = time.time()
    oss_time = end_oss - start_oss
    
    # Collect usage
    current, peak = tracemalloc.get_traced_memory()
    memory = peak / (1024 * 1024)  # Convert to MB
    tracemalloc.stop()
    
    response = {
        'execution_time': execution_time,
        'oss_time': oss_time,
        'memory': memory,
        'filename': upload_filename
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)