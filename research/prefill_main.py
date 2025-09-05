#!/usr/bin/env python3
import os
import re
import sys
import subprocess
from dynamic_adjustment import *
from utils import *

PREFILL_LOG_FILE_NAME = "prefill.log"
DECODE_LOG_FILE_NAME = "decode.log"
PROFILE_TOKEN_NUM = 8500
INPUT_TOKEN_NUM = 8500
SLO = 8.5

def main():
    # Profile run
    cmd = [
        "/home/erictsai/miniconda3/envs/vllm-research/bin/python", "two_nodes.py",
        "--mode", "prefill",
        "--ip", "10.121.187.102",
        "--kv-cache-send-ratio", "1.0",
        "--token-num", str(PROFILE_TOKEN_NUM),
    ]
    rc = run_cmd(cmd, PREFILL_LOG_FILE_NAME)

    # Extract profile recv time
    recv_time = extract_recv_time(DECODE_LOG_FILE_NAME)
    if recv_time is None:
        print("ERROR: No 'kv cache recv time' found", file=sys.stderr)
        return
    else:
        print(f"Profile recv time: {recv_time}") 
    
    # Solve for SLO
    profile = Profile(seconds_for_profile_tokens=recv_time, profile_token_num=PROFILE_TOKEN_NUM)
    dims = ModelDims(layer_num=16, num_heads=8, head_size=64, hidden_size=2048)

    result = solve_layers_for_slo(token_num=INPUT_TOKEN_NUM, slo_seconds=SLO, profile=profile, dims=dims)
    print("SLO Plan:", result)

    # Run inference with the adjusted kv_cache_send_ratio
    cmd = [
        "/home/erictsai/miniconda3/envs/vllm-research/bin/python", "two_nodes.py",
        "--mode", "prefill",
        "--ip", "10.121.187.102",
        "--kv-cache-send-ratio", str(result['send_kv_cache_ratio']),
        "--token-num", str(INPUT_TOKEN_NUM),
    ]
    rc = run_cmd(cmd, PREFILL_LOG_FILE_NAME)


if __name__ == "__main__":
    main()