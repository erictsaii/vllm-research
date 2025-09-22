#!/bin/bash 

results_folder="/home/erictsai/vllm-test/benchmark_result"
model="meta-llama/Llama-3.2-1B-Instruct"
dataset_name="sonnet"
dataset_path="sonnet.txt"
num_prompts=20
qps=3
prefix_len=50
input_len=2500
output_len=50
max_concurrency=1

python3 benchmark_serving.py \
        --backend vllm \
        --model $model \
        --dataset-name $dataset_name \
        --dataset-path $dataset_path \
        --sonnet-input-len $input_len \
        --sonnet-output-len $output_len \
        --sonnet-prefix-len $prefix_len \
        --num-prompts $num_prompts \
        --port 8000 \
        --save-result \
        --result-dir $results_folder \
        --result-filename "qps-"$qps".json" \
        --request-rate $qps \
        --max-concurrency $max_concurrency