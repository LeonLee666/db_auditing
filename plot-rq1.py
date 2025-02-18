import numpy as np
import matplotlib.pyplot as plt

# 数据集和算法
labels = ['TPC-C', 'Voter', 'YCSB', 'Sysbench']
algorithms = ['LSTM-Euler', 'LSTM-HASH-Grid', 'LSTM-Centroid']

# 示例数据，行表示不同的指标（accuracy, recall, F1），列表示不同的数据集
# 注意：这里的数据是示例数据，您可以根据实际情况进行替换
data = {
    'Precision': np.array([[98.96, 97.13, 83.6], [98.96, 97.13, 83.6], [98.96, 97.13, 83.6], [98.96, 97.13, 83.6]]),
    'Recall': np.array([[98.86, 97.13, 81.54], [98.86, 97.13, 81.54], [98.86, 97.13, 81.54], [98.86, 97.13, 81.54]]),
    'F1-score': np.array([[98.91, 97.13, 82.57], [98.91, 97.13, 82.57], [98.91, 97.13, 82.57], [98.91, 97.13, 82.57]]),
    'FE Time': np.array([[2802, 1.98, 13.39], [2802, 1.98, 13.39], [2802, 1.98, 13.39], [2802, 1.98, 13.39]])
}

# 设置柱状图的宽度
bar_width = 0.2
index = np.arange(len(labels))

# 绘制三个独立的图
metrics = list(data.keys())
for i, metric in enumerate(metrics):
    # 创建新的图形
    plt.figure(figsize=(6, 4))
    
    # 绘制柱状图，设置透明度为1（完全不透明）
    for j in range(len(algorithms)):
        plt.bar(index + j * bar_width, data[metric][:, j], bar_width, 
                label=algorithms[j], alpha=1.0)
    
    # 为FE Time设置对数坐标轴
    if metric == 'FE Time':
        plt.yscale('log')
    
    # 添加标签
    plt.xlabel('Dataset', fontsize=16)
    if metric == 'FE Time':
        plt.ylabel(f'{metric} (log scale, s)', fontsize=16)
        # FE Time 图例放在中间
        plt.legend(fontsize=14, loc='center')
    else:
        plt.ylabel(metric, fontsize=16)
        # 其他图表的图例保持在右下角
        plt.legend(fontsize=14, loc='lower right')
    plt.xticks(index + bar_width / 2, labels, fontsize=14, rotation=15)
    plt.tick_params(axis='y', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    # 保存为pdf格式
    plt.savefig(f'fig/RQ1-{metric}.pdf', format='pdf', dpi=300, transparent=False)
    plt.close()
