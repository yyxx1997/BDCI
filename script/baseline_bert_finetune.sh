export CUDA_VISIBLE_DEVICES=6,7
python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 29502 baseline_bert.py \
    --config configs/baseline-bert.yaml \
    --output_dir output/baseline_bert_wo_assignee_rdrop0_fl \
    --gradient_accumulation_steps 4 \
    --logging_step  400
    # --eval_before_train