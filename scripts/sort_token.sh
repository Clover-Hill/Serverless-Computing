python -m run_sort \
    --text_file dataset/wikitext-train-100-tokens.txt \
    --map_workers 2 \
    --reduce_workers 2 \
    --map_function_url https://map-test-qlanljpbup.cn-hangzhou.fcapp.run \
    --reduce_function_url https://reduce-sort-mzbugqspff.cn-hangzhou.fcapp.run \
    --split_strategy bayesian \
    --output_file output/sort_word_1M_100_100.json