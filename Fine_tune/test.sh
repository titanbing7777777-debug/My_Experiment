CUDA_VISIBLE_DEVICES=0,1
model=Qwen2-7B-Instruct
adapters=Qwen2-7B-Instruct/v0-20260205-105228/checkpoint-72
swift infer \
    --model $model \
    --adapters output/$adapters \
    --stream false \
    --infer_backend pt \
    --val_dataset 'Dataset(reply_chain)/DiaASQ/test.jsonl' \
    --temperature 0 \
    --max_new_tokens 2048 \
    --result_path results/$adapters/test_result.jsonl \
    --max_batch_size 2