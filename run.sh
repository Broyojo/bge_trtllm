python3 run.py \
    --engine_dir trtllm_bge/ \
    --batch_size 1 \
    --dataset lighteval/med_paragraph_simplification \
    --remove_columns answer \
    --target_column query \
    --run_hf \
    --run_trtllm \