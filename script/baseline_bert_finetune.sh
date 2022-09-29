export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=4 --use_env baseline_bert.py \
    --config configs/baseline-bert.yaml \
    --output_dir output/baseline_bert_wo_assignee_rdrop3 \
    --gradient_accumulation_steps 4 \
    --logging_step  1000 \
    # --eval_before_train \