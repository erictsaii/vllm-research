#!/bin/bash 

model="meta-llama/Llama-3.1-8B-Instruct"
num_prompts=50
qps=4
prefix_len=0
input_len=300
output_len=100
max_concurrency=10

python3 benchmark_serving.py \
    --backend vllm \
    --model $model \
    --dataset-name random \
    --random-input-len $input_len \
    --random-output-len $output_len \
    --num-prompts $num_prompts \
    --max-concurrency $max_concurrency \
    --port 8000 \
    --host vllm-proxy \
    --metric-percentiles "95" \
    --request-rate "$qps"
