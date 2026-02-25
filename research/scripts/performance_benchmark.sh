#!/bin/bash 

# model="meta-llama/Llama-3.2-1B-Instruct"
model="meta-llama/Llama-2-7b-hf"
num_prompts=10
qps=1.0
max_concurrency=1

python benchmark_serving.py \
    --backend vllm \
    --model $model \
    --dataset-name sharegpt \
    --dataset-path /home/erictsai/vllm-research/research/sharegpt.json \
    --num-prompts $num_prompts \
    --request-rate "$qps" \
    --max-concurrency $max_concurrency \
    --metric-percentiles "95" \
    --port 8000


# results_folder="/home/erictsai/vllm-test/benchmark_result"
# model="meta-llama/Llama-3.2-1B-Instruct"
# dataset_name="sonnet"
# dataset_path="a.txt"
# num_prompts=1
# qps=1.0
# prefix_len=0
# input_len=2500
# output_len=200
# max_concurrency=1

# python3 benchmark_serving.py \
#         --backend vllm \
#         --model $model \
#         --dataset-name $dataset_name \
#         --dataset-path $dataset_path \
#         --sonnet-input-len $input_len \
#         --sonnet-output-len $output_len \
#         --sonnet-prefix-len $prefix_len \
#         --num-prompts $num_prompts \
#         --port 8000 \
#         --max-concurrency $max_concurrency \
#         --metric-percentiles "1,90,95" \
#         --request-rate "$qps"