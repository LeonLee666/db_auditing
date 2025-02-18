import numpy as np
import matplotlib.pyplot as plt

# 数据密度和算法
densities = ['1/10', '1/50', '1/100', '1/500', '1/1000']
algorithms = ['LSTM-Euler', 'LSTM-HASH-Grid', 'LSTM-Centroid']

# 示例数据，每个指标对应一个二维数组：行表示不同的密度，列表示不同的算法
data = {
    'Precision': np.array([[100, 100, 95.34],  # 这些是示例数据
                          [98.85, 100, 86.7],     # 请替换为实际数据
                          [98.96, 97.13, 83.6],
                          [63.66, 49.88, 79.4],
                          [49.93, 50, 70.23]]),
    'Recall': np.array([[100, 100, 94.6426],
                       [97.9541, 100, 86.7],
                       [98.8601, 97.13, 81.4286],
                       [70.5605, 73.16, 80.6091],
                       [69.86, 18, 58.349]]),
    'F1-score': np.array([[100, 100, 94.99],
                         [98.4, 100, 86.7],
                         [98.91, 97.13, 82.5],
                         [66.75, 69.91, 80],
                         [58.23, 26.47, 63.74]]),
    'FE Time': np.array([[2814, 2.08, 13],
                        [2798, 2, 13.53],
                        [2812, 2.04, 13.38],
                        [2777, 2.07, 13.7],
                        [2829, 2.05, 13.71]])
}

# 设置柱状图的宽度
bar_width = 0.2
index = np.arange(len(densities))

# 绘制四个独立的图
metrics = list(data.keys())
for i, metric in enumerate(metrics):
    plt.figure(figsize=(6, 4))
    
    if metric != 'FE Time':  # 对于Precision, Recall, F1-score使用折线图
        # 绘制折线图
        for j in range(len(algorithms)):
            plt.plot(index, data[metric][:, j], marker='o', linewidth=2, 
                    markersize=8, label=algorithms[j])
        # 设置y轴范围为[0,100]
        plt.ylim(0, 100)
        # 添加50%的水平虚线
        plt.axhline(y=50, color='gray', linestyle='--', linewidth=2)
    else:  # FE Time保持使用柱状图
        # 绘制柱状图
        for j in range(len(algorithms)):
            plt.bar(index + j * bar_width, data[metric][:, j], bar_width, 
                    label=algorithms[j], alpha=1.0)
        plt.yscale('log')
        plt.xticks(index + bar_width, densities, fontsize=14, rotation=15)
    
    if metric != 'FE Time':
        plt.xticks(index, densities, fontsize=14, rotation=15)
    
    # 添加标签
    plt.xlabel('Data Density', fontsize=16)
    if metric == 'FE Time':
        plt.ylabel(f'{metric} (log scale, s)', fontsize=16)
    else:
        plt.ylabel(metric, fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    # 保存为pdf格式
    plt.savefig(f'fig/RQ2-TPCC-{metric}.pdf', format='pdf', dpi=300, transparent=False)
    plt.close()
