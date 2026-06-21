#!/bin/bash 

model="meta-llama/Llama-3.2-1B-Instruct"
ratio=0.66
kv_ip=${VLLM_HOST_IP:-$(hostname -I | awk '{print $1}')}
nccl_ifname=${NCCL_SOCKET_IFNAME:-eth0}

CUDA_VISIBLE_DEVICES=0 \
NCCL_P2P_DISABLE=1 \
NCCL_NET_GDR_LEVEL=0 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME=${nccl_ifname} \
python3 -m vllm.entrypoints.openai.api_server \
  --model ${model} \
  --host 0.0.0.0 \
  --port 8100 \
  --max-model-len 2000 \
  --gpu-memory-utilization 0.9 \
  --dtype "half" \
  --kv-transfer-config \
  '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":1e9,"kv_ip":"'"${kv_ip}"'","kv_cache_send_ratio":'"${ratio}"',"kv_isl_threshold":2000}'