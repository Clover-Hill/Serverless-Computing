python -m run_sort \
    --text_file dataset/wikitext-train-1M-words.txt \
    --map_workers 10 \
    --reduce_workers 10 \
    --map_function_url https://map-test-qlanljpbup.cn-hangzhou.fcapp.run \
    --reduce_function_url https://reduce-sort-mzbugqspff.cn-hangzhou.fcapp.run \
    --split_strategy bayesian \
    --output_file output/sort_word_1M_10_10.json