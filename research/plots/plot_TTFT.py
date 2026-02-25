#!/usr/bin/env python3
# Overlay empirical TTFT with theoretical sequential model
# Model (sequential / 先算後傳):
#   TTFT(n) = max(1, X*(1-n)) + alpha*n
# 理想曲線以 n=1.0 的實測值做對齊（縮放）。

import numpy as np
import matplotlib.pyplot as plt

alpha = 3.5
X = 4

# 實測資料（n -> ms）
emp_points = {
    1.0: 1062,
    0.75: 896,
    0.6: 926,
    0.4: 956,
    0.2: 981,
    0.0: 1050,
}


ns_emp = np.array(sorted(emp_points.keys()))
ttft_emp = np.array([emp_points[n] for n in ns_emp])

n_curve = np.linspace(0, 1, 1001)
ttft_th = np.maximum(1.0, X*(1.0 - n_curve)) + alpha*n_curve

ttft_th_n1 = 1.0 + alpha  # max(1, X*(1-1)) + alpha*1
scale_ms_per_unit = emp_points[1.0] / ttft_th_n1
ttft_th_scaled = ttft_th * scale_ms_per_unit

n_star = 1.0 - 1.0/X if alpha <= X else 0.0

ttft_star_ms = (np.maximum(1.0, X*(1.0 - n_star)) + alpha*n_star) * scale_ms_per_unit

print(f"Scale factor (ms per unit): {scale_ms_per_unit:.3f}")
print(f"Optimal n* (sequential model): {n_star:.3f}")
print(f"Theoretical TTFT at n*: {ttft_star_ms:.1f} ms")

plt.figure(figsize=(8, 5))
plt.plot(n_curve, ttft_th_scaled, label="ideal TTFT", linewidth=2, color="blue")
plt.plot(ns_emp, ttft_emp, marker="o", label="real TTFT", linewidth=2, color="red")

plt.ylim(bottom=0, top=1200)   


plt.axvline(n_star, linestyle="--", label=f"ideally best n* = {n_star:.2f}", color="blue")
plt.scatter([n_star], [ttft_star_ms], s=60)

plt.title(f"TTFT vs n (alpha={alpha:.1f}, X={X:.0f})")
plt.xlabel("n")
plt.ylabel("TTFT (ms)")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plot.png", dpi=200)
plt.show()
