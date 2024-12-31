python -m run_no_sort \
    --text_file dataset/wikitext-train-1M-words.txt \
    --map_workers 100 \
    --reduce_workers 100 \
    --map_function_url https://map-test-qlanljpbup.cn-hangzhou.fcapp.run \
    --reduce_function_url https://reduce-test-mervgqspff.cn-hangzhou.fcapp.run \
    --split_strategy naive \
    --output_file output/no_sort_word_1M_100_100.json