# ---- 0. 基底：PyTorch 2.6.0 + CUDA 12.4 + CUDNN 9 ----
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# ---- 1. 系統套件 ----
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git build-essential ninja-build cmake vim && \  
    rm -rf /var/lib/apt/lists/*

# ---- 2. Python 依賴（鎖版）----
# 建議用 constraints 檔，避免覆蓋掉 torch/cuda 內建依賴
COPY research/docker-experiment/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir --constraint /tmp/constraints.txt \
      --upgrade pip wheel setuptools \
      ninja

# ---- 3. 複製 repo 並刪舊編譯產物 ----
WORKDIR /workspace
COPY . /workspace/vllm-research
RUN find /workspace/vllm-research/vllm -name '_C*.so' -delete && \
    rm -rf /workspace/vllm-research/build /workspace/vllm-research/*.egg-info

# ---- 4. 先安裝 build‑time 依賴 ----
RUN pip install --no-cache-dir setuptools_scm==8.0.4 tomli ninja

# ---- 5. 編譯並安裝 vLLM（可維持 --no-build-isolation）----
RUN pip install --no-build-isolation --no-cache-dir -e /workspace/vllm-research

CMD ["tail", "-f", "/dev/null"]
