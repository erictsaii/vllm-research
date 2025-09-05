import os
import re
import subprocess
from dynamic_adjustment import *

def run_cmd(cmd, log_filename):
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_NET_GDR_LEVEL"] = "0"
    env["NCCL_IB_DISABLE"] = "1"
    env["NCCL_SOCKET_IFNAME"] = "enp1s0f0"

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