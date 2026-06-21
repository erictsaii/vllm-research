import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14}) 

n_values = np.array([100, 300, 500, 700, 900, 1100, 1300, 1500])
T_values = np.array([0.022, 0.032, 0.046, 0.058, 0.074, 0.086, 0.102, 0.114])
F_values = np.array([0.010, 0.011, 0.011, 0.013, 0.016, 0.016, 0.018, 0.019])

X_scenarios = [3, 4, 5]

def find_intersection_n(n_arr, y1_arr, y2_arr):
    diff = y2_arr - y1_arr
    
    for i in range(len(n_arr) - 1):
        if (diff[i] >= 0 and diff[i+1] < 0) or (diff[i] < 0 and diff[i+1] >= 0):
            n_cross = n_arr[i] + (0 - diff[i]) * (n_arr[i+1] - n_arr[i]) / (diff[i+1] - diff[i])
            return n_cross
    return None

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# fig.suptitle('Analysis of T <= XF Condition', fontsize=16)

for i, X in enumerate(X_scenarios):
    ax = axes[i]
    
    XF_values = F_values * X
    
    ax.plot(n_values, T_values, 'b-o', label='T', linewidth=2)
    ax.plot(n_values, XF_values, 'r--s', label=f'{X}F', linewidth=2)
    
    cross_n = find_intersection_n(n_values, T_values, XF_values)
    
    ax.fill_between(n_values, T_values, XF_values, 
                    where=(T_values <= XF_values), 
                    interpolate=True, color='green', alpha=0.2, label=f'T <= {X}F')
    
    if cross_n:
        cross_y = np.interp(cross_n, n_values, T_values)
        
        ax.axvline(x=cross_n, color='k', linestyle=':', alpha=0.7)
        ax.annotate(
            f'length ≈ {cross_n:.0f}',
            xy=(cross_n, cross_y),
            xytext=(15, -50),              # 相對偏移：右 20 pt、下 15 pt
            textcoords='offset points',
            arrowprops=dict(
                arrowstyle='simple',
                fc='black',
                ec='black',
                mutation_scale=10
            )
        )

    ax.set_xlabel('Input Length')
    ax.set_ylabel('Time (s)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('isl_comparison_plot.png', dpi=300, bbox_inches='tight')