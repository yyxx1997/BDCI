export CUDA_VISIBLE_DEVICES=2,3
python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 29501 baseline_bert.py \
    --config configs/baseline-bert.yaml \
    --output_dir output/baseline_bert_debug \
    --gradient_accumulation_steps 1 \
    --logging_step  100
    # --eval_before_train