# Cloud Serverless computing project requirement

1. Implement a map-reduce framework to handle word frequency task.
2. How many map and reduce functions do you set? How do they affect the completion time?
3. Can you make a summary of the function execution time and communication time ratio?
4. What is the memory consumption of each function?
5. How the memory/cpu parameter influences the execution time?
6. Design an automatic algorithm to find the minimal cost: 贝叶斯优化、机器学习、启发式算法

# Project Structure

You need to implement at least three files:

1. main.py: the main function to run the map-reduce framework.
2. map_function.py: the map function to handle the word frequency task.
3. reduce_function.py: the reduce function to handle the word frequency task.

In main.py, you need to implement the following:

Input args: 
text file: the text file to be word counted
map worker number: the number of map workers
reduce worker number: the number of reduce workers
map function http: the http url of the map function
reduce function http: the http url of the reduce function
(Optional) split strategy: the strategy to split the text file, default is naive split. Based on requirement 6, also try to implement other advanced strategies.\

the text file can be loaded to list in the following way:
def load_words(filename):
    # Read words from file
    with open(filename, "r", encoding="utf-8") as f:
        words = f.read().split("\n")
    return words

main.py pipeline:
1. Read the text file and split it into map worker number parts based on the given split strategy.
2. Call map function for each part.
3. Call reduce function for each map function result.
4. Aggregate the reduce function result to get the final word frequency.
5. Check the result with the local result.
6. Output the execution time and communication time ratio, memory consumption, and cpu usage for each function. Note that these metrics need to be calculated in map_function.py and reduce_function.py, and aggregated in main.py. You need to output both total metric and averaged per-worker metric. In specific, the communication time is the time to write / read from oss bucket. The execution time is the time to process the word frequency task in map and reduce function.

In map_function.py and reduce_function.py, these two functions will be run in the cloud serverless environment, an example is provided in example/serverless_fn_example.py.

You need to implement the following:
In map_function.py, you need to implement the map function to handle the word frequency task:
1. Convert the given text to word-frequency pairs.
2. Save the word-frequency pairs to the oss bucket. The example to use oss sdk is provided in example/oss_example.py. Oss key and secret is in .env file.
3. Input args: 
    - text: the text to be word counted
    - index: the index of the map worker
    - oss_bucket name: the name of the oss bucket
    - oss_bucket_key: the key of the oss bucket
    - oss_bucket_secret: the secret of the oss bucket
4. Output args:
    - memory: the memory consumption of the map function
    - cpu: the cpu usage of the map function
    - execution time: the execution time of the map function
    - communication time: the communication time of the map function

In reduce_function.py, you need to implement the reduce function to handle the word frequency task:
1. Read the word-frequency pairs from the oss bucket.
2. Aggregate the word-frequency pairs to get the final word frequency.
3. Return the final word frequency for this reduce worker.
4. Input args:
    - index: the index of the reduce worker
    - oss_bucket name: the name of the oss bucket
    - oss_bucket_key: the key of the oss bucket
    - oss_bucket_secret: the secret of the oss bucket
5. Output args:
    - memory: the memory consumption of the reduce function
    - cpu: the cpu usage of the reduce function
    - execution time: the execution time of the reduce function
    - communication time: the communication time of the reduce function
    - result: the final word frequency for this reduce worker

Reduce function http: https://reduce-test-mervgqspff.cn-hangzhou.fcapp.run
Map function http: https://map-test-qlanljpbup.cn-hangzhou.fcapp.run

# Additional requirements

1. Design different map and reduce strategies, and compare their performance.

A naive map strategy is to split the text file into equal parts, and each map function process one part.

A naive reduce strategy is each reduce function process one part of the word-frequency pairs.

A differnet reduce strategy can be each reduce function process words with the same inital letter.

Try to find the minimal cost strategy based on the execution time, memory consumption, and cpu usage.