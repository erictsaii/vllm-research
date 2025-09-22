#!/usr/bin/env python3
import sys
import argparse
import logging

from dynamic_adjustment import *
from utils import *

# ===== Default values (same as your original) =====
PYTHON_BIN = "/home/erictsai/miniconda3/envs/vllm-research/bin/python"
SCRIPT_NAME = "two_nodes.py"

PREFILL_LOG_FILE_NAME = "prefill.log"
DECODE_LOG_FILE_NAME = "decode.log"

PROFILE_TOKEN_NUM = 8500
INPUT_TOKEN_NUM = 8500
SLO = 2

MODE = "prefill"
IP = "10.121.187.102"
KV_RATIO_FOR_PROFILE = 1.0

# Model dimensions (same as original)
DIMS = ModelDims(layer_num=16, num_heads=8, head_size=64, hidden_size=2048)


def run_two_nodes(python_bin, script_name, mode, ip, kv_ratio, token_num, log_file):
    """Wrapper to run two_nodes.py with given parameters."""
    cmd = [
        python_bin, script_name,
        "--mode", mode,
        "--ip", ip,
        "--kv-cache-send-ratio", str(kv_ratio),
        "--token-num", str(token_num),
    ]
    logging.debug("Command: %s", " ".join(cmd))
    return run_cmd(cmd, log_file)


def parse_args():
    p = argparse.ArgumentParser(description="Profile -> SLO planning -> Final run")
    p.add_argument("--ip", default=IP)
    p.add_argument("--mode", default=MODE)
    p.add_argument("--profile-token-num", type=int, default=PROFILE_TOKEN_NUM)
    p.add_argument("--input-token-num", type=int, default=INPUT_TOKEN_NUM)
    p.add_argument("--slo", type=float, default=SLO)
    p.add_argument("--kv-ratio-for-profile", type=float, default=KV_RATIO_FOR_PROFILE)
    p.add_argument("--python-bin", default=PYTHON_BIN)
    p.add_argument("--script", default=SCRIPT_NAME)
    p.add_argument("--prefill-log", default=PREFILL_LOG_FILE_NAME)
    p.add_argument("--decode-log", default=DECODE_LOG_FILE_NAME)
    g = p.add_mutually_exclusive_group()
    g.add_argument("-v", "--verbose", action="store_true", help="Show more detailed logs")
    g.add_argument("-q", "--quiet", action="store_true", help="Show less logs")
    return p.parse_args()


def setup_logging(verbose: bool, quiet: bool):
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(message)s",
    )


def main():
    args = parse_args()
    setup_logging(args.verbose, args.quiet)

    # ===== Stage 1: Profiling =====
    logging.info("Stage 1/4: Running profiling (tokens=%d, kv_ratio=%.2f) ...",
                 args.profile_token_num, args.kv_ratio_for_profile)
    rc = run_two_nodes(
        python_bin=args.python_bin,
        script_name=args.script,
        mode=args.mode,
        ip=args.ip,
        kv_ratio=args.kv_ratio_for_profile,
        token_num=args.profile_token_num,
        log_file=args.prefill_log,
    )
    if rc != 0:
        logging.error("Profiling failed (rc=%d). See log: %s", rc, args.prefill_log)
        sys.exit(rc)
    logging.info("Profiling completed.")

    # ===== Stage 2: Extract recv time =====
    logging.info("Stage 2/4: Extracting recv time from %s ...", args.decode_log)
    recv_time = extract_recv_time(args.decode_log)
    if recv_time is None:
        logging.error("No 'kv cache recv time' found. Cannot continue.")
        sys.exit(1)
    logging.info("Profile recv time = %.6f s", recv_time)

    # recv_time = 13.3

    # ===== Stage 3: Solve SLO plan =====
    logging.info("Stage 3/4: Solving SLO plan (SLO=%.3fs, tokens=%d) ...",
                 args.slo, args.input_token_num)
    profile = Profile(seconds_for_profile_tokens=recv_time, profile_token_num=args.profile_token_num)
    result = solve_layers_for_slo(
        token_num=args.input_token_num,
        slo_seconds=args.slo,
        profile=profile,
        dims=DIMS,
    )
    if not isinstance(result, dict) or "send_kv_cache_ratio" not in result:
        logging.error("Unexpected SLO result: %r", result)
        sys.exit(1)
    kv_ratio = float(result["send_kv_cache_ratio"])
    logging.info("SLO plan: send_kv_cache_ratio=%.6f", kv_ratio)

    # ===== Stage 4: Final run with adjusted kv_ratio =====
    logging.info("Stage 4/4 : Running final job with adjusted kv_ratio (tokens=%d) ...",
                 args.input_token_num)
    rc = run_two_nodes(
        python_bin=args.python_bin,
        script_name=args.script,
        mode=args.mode,
        ip=args.ip,
        kv_ratio=kv_ratio,
        token_num=args.input_token_num,
        log_file=args.prefill_log,
    )
    if rc != 0:
        logging.error("Final run failed (rc=%d). See log: %s", rc, args.prefill_log)
        sys.exit(rc)

    logging.info("All stages completed successfully.")


if __name__ == "__main__":
    main()
