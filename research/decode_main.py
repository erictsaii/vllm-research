#!/usr/bin/env python3
import os
import re
import sys
import subprocess
from dynamic_adjustment import *

DECODE_LOG_FILE_NAME = "decode.log"

def run_profile(log_filename):
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_NET_GDR_LEVEL"] = "0"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = "enp1s0f0"

    cmd = [
        "/home/erictsai/miniconda3/envs/vllm-research/bin/python", "two_nodes.py",
        "--mode", "decode",
        "--ip", "10.121.187.102",
        "--kv-cache-send-ratio", "1.0",
    ]

    with open(log_filename, "w") as f:
        completed = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode

def extract_recv_time(log_path):
    pat = re.compile(r"kv cache recv time:\s*([0-9]+(?:\.[0-9]+)?)")
    last = None
    with open(log_path, "r") as f:
        for line in f:
            m = pat.search(line)
            if m:
                last = m.group(1)
    return float(last)


def run_inference(log_filename, kv_cache_send_ratio):
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_NET_GDR_LEVEL"] = "0"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = "enp1s0f0"

    cmd = [
        "/home/erictsai/miniconda3/envs/vllm-research/bin/python", "two_nodes.py",
        "--mode", "decode",
        "--ip", "10.121.187.102",
        "--kv-cache-send-ratio", str(kv_cache_send_ratio),
    ]

    with open(log_filename, "w") as f:
        completed = subprocess.run(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


PROFILE_TOKEN_NUM = 8500
INPUT_TOKEN_NUM = 8500
SLO = 10.0

def main():
    rc = run_profile(DECODE_LOG_FILE_NAME)

    recv_time = extract_recv_time(DECODE_LOG_FILE_NAME)
    if recv_time is None:
        print("ERROR: No 'kv cache recv time' found", file=sys.stderr)
        return
    else:
        print(f"Profile recv time: {recv_time}") 

    profile = Profile(seconds_for_profile_tokens=recv_time, profile_token_num=PROFILE_TOKEN_NUM)
    dims = ModelDims(layer_num=16, num_heads=8, head_size=64, hidden_size=2048)

    result = solve_layers_for_slo(token_num=INPUT_TOKEN_NUM, slo_seconds=SLO, profile=profile, dims=dims)
    print("SLO Plan:", result)

    rc = run_inference(DECODE_LOG_FILE_NAME, result['send_kv_cache_ratio'])

    recv_time = extract_recv_time(DECODE_LOG_FILE_NAME)
    print(f"SLO: {SLO}")
    print(f"Real recv time: {recv_time}")


if __name__ == "__main__":
    main()