import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15}) 


# qps_points = [4, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
qps_points = [2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]


throughput_n1 = {
    2: 2.02,
    2.5: 2.45,
    3.0: 2.57,
    3.5: 2.64,
    4.0: 2.66,
    4.5: 2.71,
    5.0: 2.75,
    5.5: 2.73
}

throughput_n08 = {
    2: 2.02,
    2.5: 2.45,
    3.0: 2.66,
    3.5: 2.7,
    4.0: 2.75,
    4.5: 2.78,
    5.0: 2.81,
    5.5: 2.81
}



y1 = [throughput_n1[q] for q in qps_points]
y2 = [throughput_n08[q] for q in qps_points]

plt.figure(figsize=(7,5))
plt.plot(qps_points, y1, marker='o', label='vllm', color="#265ADE")
plt.plot(qps_points, y2, marker='s', label='NIKA', color='#F35959')

plt.ylim(bottom=0, top=5.0)   


plt.xlabel('Request rate (qps)')
plt.ylabel('Request throughput (req/s)')
plt.xticks(qps_points)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig('request_rate_vs_throughput.png', dpi=200)
plt.show()
