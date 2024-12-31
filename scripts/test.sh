python -m main \
    --text_file dataset/wikitext-train-1M-words.txt \
    --map_workers 100 \
    --reduce_workers 10 \
    --map_function_url https://map-test-qlanljpbup.cn-hangzhou.fcapp.run \
    --reduce_function_url https://reduce-test-mervgqspff.cn-hangzhou.fcapp.run \
    --split_strategy naive \
    --output_file output/1M_100_10.json