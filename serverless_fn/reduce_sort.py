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
    assigned_contents = data.get('assigned_contents', [])

    # Execution: Aggregate word frequencies
    start_exec = time.time()
    aggregated_freq = defaultdict(int)
    for word, count in assigned_contents:
        aggregated_freq[word] += count
    end_exec = time.time()
    execution_time = end_exec - start_exec
    
    # Collect memory and CPU usage
    current, peak = tracemalloc.get_traced_memory()
    memory = peak / (1024 * 1024)  # Convert to MB
    tracemalloc.stop()
    
    response = {
        'execution_time': execution_time,
        'memory': memory,
        'result': aggregated_freq
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
