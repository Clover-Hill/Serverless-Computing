python -m run_sort \
    --text_file dataset/wikitext-train-100K-tokens.txt \
    --map_workers 40 \
    --reduce_workers 40 \
    --map_function_url https://map-test-qlanljpbup.cn-hangzhou.fcapp.run \
    --reduce_function_url https://reduce-sort-mzbugqspff.cn-hangzhou.fcapp.run \
    --split_strategy naive \
    --output_file output/sort_token_100K_40_40.json