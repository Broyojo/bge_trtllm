python3 build.py \
    --dtype float16 \
    --max_batch_size 512 \
    --max_input_len 512 \
    --gpus_per_node 1 \
    --output_dir trtllm_bge \
    --model BAAI/bge-base-en-v1.5 \
    # --use_bert_attention_plugin float16 \
    # --use_gemm_plugin float16 \
    # --enable_context_fmha \
