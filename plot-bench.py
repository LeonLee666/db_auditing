import matplotlib.pyplot as plt
import numpy as np

# 数据集名称
datasets = ['TPC-C', 'YCSB', 'Voter', 'Twitter', 'Wikipedia']

# Throughput数据
throughput_log_enabled = [34.85, 165.399, 84.164, 737.438, 124.618]    # Performance with logging enabled
throughput_log_disabled = [35.7383, 176.218, 92.5, 746.243, 128.2315]  # Performance with logging disabled

# Latency数据 (示例数据，请替换为实际的延迟数据)
latency_log_enabled = [2285004, 481664, 947001, 108445, 637285]    # Latency with logging enabled
latency_log_disabled = [2234222, 450431, 861746, 107170, 622565]   # Latency with logging disabled

# 对数据进行归一化处理
normalized_throughput_enabled = [e/d for e, d in zip(throughput_log_enabled, throughput_log_disabled)]
normalized_throughput_disabled = [1.0] * len(datasets)  # 基准值都为1.0

normalized_latency_enabled = [e/d for e, d in zip(latency_log_enabled, latency_log_disabled)]
normalized_latency_disabled = [1.0] * len(datasets)  # 基准值都为1.0

# 设置柱状图的位置
x = np.arange(len(datasets))
width = 0.35

# 设置全局字体大小
plt.rcParams.update({'font.size': 14})

# 绘制并保存归一化后的Throughput图
fig1, ax1 = plt.subplots(figsize=(6,4))
rects1 = ax1.bar(x - width/2, normalized_throughput_disabled, width, label='Logging Disabled')
rects2 = ax1.bar(x + width/2, normalized_throughput_enabled, width, label='Logging Enabled')

ax1.set_ylabel('Normalized Throughput', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=14, rotation=15)
ax1.tick_params(axis='y', labelsize=14)
ax1.legend(loc='lower right', fontsize=14)

plt.tight_layout()
plt.savefig('fig/bench-throughput.pdf')
plt.close()

# 绘制并保存归一化后的Latency图
fig2, ax2 = plt.subplots(figsize=(6,4))
rects3 = ax2.bar(x - width/2, normalized_latency_disabled, width, label='Logging Disabled')
rects4 = ax2.bar(x + width/2, normalized_latency_enabled, width, label='Logging Enabled')

ax2.set_ylabel('Normalized Latency', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=14, rotation=15)
ax2.tick_params(axis='y', labelsize=14)
ax2.legend(loc='lower right', fontsize=14)

plt.tight_layout()
plt.savefig('fig/bench-latency.pdf')
plt.close()
