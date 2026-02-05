# 22GB
model=Qwen2-7B-Instruct

CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model ./$model \
    --tuner_backend peft \
    --dataset 'Dataset2.0/en/train.jsonl' \
    --val_dataset 'Dataset2.0/en/valid.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output/$model/ \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot