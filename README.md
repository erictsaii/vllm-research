# NIKA on vLLM Research Fork

This repository is a vLLM-based research fork for the paper:

**NIKA: Optimal KV Cache Transfer for Minimizing the Latency of Disaggregated LLM Inference**

This README focuses on one goal: how to start the disaggregated inference system (prefill + decode + proxy) in Docker.


## System Topology

- **Prefill server**: GPU 0, port 8100
- **Decode server**: GPU 1, port 8200
- **Proxy server**: CPU container, port 8000
- **Benchmark/client container**: optional traffic generation

Proxy route:
1. Send a short prefill request (`max_tokens=1`) to prefill server.
2. Send the original request to decode server.
3. Stream decode response back to client.

## Prerequisites

- Linux host with NVIDIA drivers and Docker GPU runtime.
- At least 2 GPUs (recommended for prefill/decode split).
- Hugging Face access for `meta-llama/Llama-3.2-1B-Instruct`.
- Local clone of this repository.

## Step-by-Step Startup (Docker, 4 Containers)

### 1. Pull image and clone repo

```bash
docker pull erictsai90/myvllm:5090
git clone https://github.com/erictsaii/vllm-research.git ~/vllm-research
cd ~/vllm-research
```

### 2. Create Docker network

```bash
docker network create vllm-pd-net
```

### 3. Start proxy container

```bash
docker run --rm -it \
  --name vllm-proxy \
  --network vllm-pd-net \
  -v ~/vllm-research:/workspace/vllm \
  -e OPENAI_API_KEY=EMPTY \
  erictsai90/myvllm:5090 \
  bash
```

### 4. Start decode container (GPU 1)

```bash
docker run --rm -it \
  --name vllm-decode \
  --network vllm-pd-net \
  --gpus '"device=1"' \
  --ipc=host \
  --shm-size=16g \
  -p 8200:8200 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-research:/workspace/vllm \
  -e NCCL_SOCKET_IFNAME=eth0 \
  erictsai90/myvllm:5090 \
  bash
```

### 5. Start prefill container (GPU 0)

```bash
docker run --rm -it \
  --name vllm-prefill \
  --network vllm-pd-net \
  --gpus '"device=0"' \
  --ipc=host \
  --shm-size=16g \
  --cap-add NET_ADMIN \
  -p 8100:8100 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-research:/workspace/vllm \
  -e NCCL_SOCKET_IFNAME=eth0 \
  erictsai90/myvllm:5090 \
  bash
```

### 6. Start benchmark/client container (optional)

```bash
docker run --rm -it \
  --name vllm-benchmark \
  --network vllm-pd-net \
  -v ~/vllm-research:/workspace/vllm \
  erictsai90/myvllm:5090 \
  bash
```

## Start Services Inside Containers

### A. In `vllm-prefill`

```bash
cd /workspace/vllm/research/scripts
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
bash run_prefill.sh
```

### B. In `vllm-decode`

Use the same KV producer IP as prefill.

```bash
cd /workspace/vllm/research/scripts
export VLLM_HOST_IP=vllm-prefill
bash run_decode.sh
```

### C. In `vllm-proxy`

```bash
cd /workspace/vllm
python research/disagg_prefill_proxy_server.py
```

## Send Requests to Proxy

From host machine (or benchmark container), send traffic to port 8000:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Explain NIKA disaggregated inference in one paragraph.",
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": false
  }'
```

## Optional Benchmark Entry

In `vllm-benchmark` container:

```bash
cd /workspace/vllm/benchmarks/disagg_benchmarks
bash disagg_performance_benchmark.sh
```

