import numpy as np
import matplotlib.pyplot as plt

# 数据集
labels = ['TPC-C', 'Voter', 'YCSB', 'Sysbench']
algorithms = ['No AuditLog', 'Recording Single SQL', 'Recording ALL SQL']

# 吞吐量数据
throughput_data = np.array([
    [100, 95, 88],
    [120, 96, 86],
    [78, 70, 50],
    [800, 780, 700]
])

# 延迟数据 (这里需要替换成实际的延迟数据)
latency_data = np.array([
    [10, 12, 15],
    [8, 10, 12],
    [15, 18, 25],
    [5, 6, 8]
])

# 设置柱状图的宽度
bar_width = 0.2

# 设置颜色方案（不使用透明度）
colors = ['#2878B5', '#9AC9DB', '#C82423']

# 为每个数据集创建吞吐量和延迟图表
for i, label in enumerate(labels):
    # 绘制吞吐量图表
    plt.figure(figsize=(3, 4))
    index = np.arange(3)
    
    for j in range(len(algorithms)):
        plt.bar(index[j], throughput_data[i, j], bar_width, 
                label=algorithms[j], color=colors[j])
    
    plt.xlabel('Algorithms', fontsize=16)
    plt.ylabel('TPS', fontsize=16)
    plt.xticks(index, ['', '', ''], fontsize=14)
    plt.legend(fontsize=14)
    plt.tick_params(axis='y', labelsize=14)  # 设置y轴刻度字体大小
    plt.tight_layout()
    plt.savefig(f'fig/throughput_{label.lower()}.pdf', format='pdf')
    plt.close()

    # 绘制延迟图表
    plt.figure(figsize=(3, 4))
    index = np.arange(3)
    
    for j in range(len(algorithms)):
        plt.bar(index[j], latency_data[i, j], bar_width, 
                label=algorithms[j], color=colors[j])
    
    plt.xlabel('Algorithms', fontsize=16)
    plt.ylabel('Latency (ms)', fontsize=16)
    plt.xticks(index, ['', '', ''], fontsize=14)
    plt.legend(fontsize=14)
    plt.tick_params(axis='y', labelsize=14)  # 设置y轴刻度字体大小
    plt.tight_layout()
    plt.savefig(f'fig/latency_{label.lower()}.pdf', format='pdf', transparent=False)
    plt.close()
