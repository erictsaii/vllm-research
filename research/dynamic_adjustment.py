#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ModelDims:
    # Fixed model dimensions
    layer_num: int = 16
    num_heads: int = 8
    head_size: int = 64
    hidden_size: int = 2048

@dataclass
class Profile:
    # Baseline: measured seconds to transfer K/V cache + hidden state for profile_token_num tokens
    seconds_for_profile_tokens: float
    profile_token_num: int = 9000
    bytes_per_elem: int = 2

def elements_per_token(layers: int, dims: ModelDims) -> int:
    """
    Compute number of scalar elements to transfer per token:
    - Key cache:     (layers, token, heads, head_size)
    - Value cache:   (layers, token, heads, head_size)
    - Hidden state:  (token, hidden_size)
    Total per token = 2 * layers * heads * head_size + hidden_size
    """
    return 2 * layers * dims.num_heads * dims.head_size + dims.hidden_size

def predict_time_seconds(token_num: int,
                         layers: int,
                         profile: Profile,
                         dims: ModelDims = ModelDims()) -> float:
    """
    Predict transfer time (seconds) for a given token_num and layers using
    proportional scaling from the baseline profile.
    Time ∝ total elements to transfer.
    """
    base_layers = dims.layer_num
    base_elems_per_token = elements_per_token(base_layers, dims)
    target_elems_per_token = elements_per_token(layers, dims)

    # Ratio by total element count
    ratio = (token_num * target_elems_per_token) / (profile.profile_token_num * base_elems_per_token)
    return profile.seconds_for_profile_tokens * ratio

def solve_layers_for_slo(token_num: int,
                         slo_seconds: float,
                         profile: Profile,
                         dims: ModelDims = ModelDims()) -> dict:
    """
    Given an SLO (seconds), compute the maximum integer layer_num' (1..base_layers)
    such that predicted time <= SLO.
    Also return the remaining transfer ratio vs. the original design (base_layers).
    If even 1 layer exceeds the SLO, return layer_num'=1 and mark 'feasible': False.
    """
    base_layers = dims.layer_num
    H, S = dims.num_heads, dims.head_size
    hidden = dims.hidden_size

    # Constants to simplify the closed-form
    base_per_tok = 2 * base_layers * H * S + hidden
    denom_per_layer = 2 * H * S  # increase per token per layer for K/V

    # From inequality:
    # slo / T0 * N0 * (denom_per_layer * L0 + hidden) / N = denom_per_layer * L' + hidden
    T0 = profile.seconds_for_profile_tokens
    N0 = profile.profile_token_num
    N = token_num

    bound = (slo_seconds / T0) * N0 * base_per_tok / N
    max_L_cont = (bound - hidden) / denom_per_layer
    max_L_int = math.floor(max_L_cont)

    # Clamp to [1, base_layers]
    # max_L_int -= 1 # empirically
    L_prime = max(1, min(base_layers, max_L_int))

    # Check feasibility at L_prime
    pred_time = predict_time_seconds(token_num, L_prime, profile, dims)
    feasible = pred_time <= slo_seconds

    # send kv cache ratio
    remaining_ratio = L_prime / base_layers

    return {
        "layer_num_prime": L_prime,
        "feasible": feasible,
        "pred_time_seconds_at_L_prime": pred_time,
        "send_kv_cache_ratio": remaining_ratio,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-result", type=float, required=True)
    parser.add_argument("--profile-token-num", type=int, default=8500)

    args = parser.parse_args()

    profile = Profile(seconds_for_profile_tokens=args.profile_result,
                      profile_token_num=args.profile_token_num)

    dims = ModelDims(layer_num=16, num_heads=8, head_size=64, hidden_size=2048)

    # --- predict time for arbitrary tokens with current layers ---
    token_num = 8500
    # t_pred = predict_time_seconds(token_num, layers=dims.layer_num, profile=profile, dims=dims)
    # print(f"[Predict] {token_num=}, layers={dims.layer_num} -> time ≈ {t_pred:.4f}s")

    # --- given an SLO, compute required layers and remaining ratio ---
    slo = 10 # seconds
    result = solve_layers_for_slo(token_num=token_num, slo_seconds=slo, profile=profile, dims=dims)
    print("[SLO Plan]", result)
