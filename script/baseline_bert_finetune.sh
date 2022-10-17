export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port 29502 baseline_bert.py \
    --config configs/baseline-bert.yaml \
    --output_dir output/baseline_bert_debug \
    --gradient_accumulation_steps 1 \
    --logging_step  100
    # --eval_before_train