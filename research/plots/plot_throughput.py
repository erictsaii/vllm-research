import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15}) 


# qps_points = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
qps_points = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]



# 實測資料
# throughput_n1 = {
#     5.0: 4.51,
#     6.0: 5.20,
#     7.0: 5.47,
#     8.0: 5.75,
#     9.0: 5.87,
#     10.0: 5.85
# }
# throughput_n08 = {
#     5.0: 4.54,
#     6.0: 5.32,
#     7.0: 5.83,
#     8.0: 6.15,
#     9.0: 6.18,
#     10.0:6.22
# }

throughput_n1 = {
    2.0: 2.09,
    2.5: 2.56,
    3.0: 3.0,
    3.5: 3.1,
    4.0: 3.18,
    4.5: 3.22
}
throughput_n08 = {
    2.0: 2.09,
    2.5: 2.56,
    3.0: 3.0,
    3.5: 3.34,
    4.0: 3.37,
    4.5: 3.41
}

y1 = [throughput_n1[q] for q in qps_points]
y2 = [throughput_n08[q] for q in qps_points]

plt.figure(figsize=(7,5))
plt.plot(qps_points, y1, marker='o', label='vllm', color="#265ADE")
plt.plot(qps_points, y2, marker='s', label='NIKA', color='#F35959')

plt.ylim(bottom=0, top=7.0)   


plt.xlabel('Request rate (qps)')
plt.ylabel('Request throughput (req/s)')
plt.xticks(qps_points)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('request_rate_vs_throughput.png', dpi=200)
plt.show()
