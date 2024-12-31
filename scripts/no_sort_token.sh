python -m run_no_sort \
    --text_file dataset/wikitext-train-1M-tokens.txt \
    --map_workers 100 \
    --reduce_workers 100 \
    --map_function_url https://map-test-qlanljpbup.cn-hangzhou.fcapp.run \
    --reduce_function_url https://reduce-test-mervgqspff.cn-hangzhou.fcapp.run \
    --output_file output/no_sort_token_1M_100_100.json