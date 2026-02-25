#!/bin/bash 

# model="meta-llama/Llama-3.2-1B-Instruct"
model="meta-llama/Llama-2-7b-hf"
ratio=0.66

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_NET_GDR_LEVEL=0 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=enp1s0f0 \
python3 -m vllm.entrypoints.openai.api_server \
  --model ${model} \
  --port 8100 \
  --max-model-len 2000 \
  --gpu-memory-utilization 0.9 \
  --dtype "half" \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":1e9,"kv_ip":"10.121.187.102","kv_cache_send_ratio":'"${ratio}"',"kv_isl_threshold":2000}'