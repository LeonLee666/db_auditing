import numpy as np
import matplotlib.pyplot as plt

# 数据密度和算法
windows = ['128', '256', '512', '1024', '2048', '4096', '8192']  # x轴值：window size
algorithms = ['LSTM-Euler', 'LSTM-HASH-Grid', 'LSTM-Centroid']

# 示例数据，每个指标对应一个二维数组：行表示不同的window size，列表示不同的算法
data = {
    'Precision': np.array([[63.09, 71.15, 91.6],    
                          [93.53, 94.77, 76.0],
                          [95.98, 99.19, 94.4],
                          [97.31, 97.43, 84.8],
                          [98.01, 97.22, 74.8],
                          [98.96, 97.13, 83.6],
                          [100.0, 94.3, 83.87]]),
    'Recall': np.array([[45.5291, 80.5528, 91.8002],    
                       [93.3302, 94.0924, 80.1081],
                       [95.7603, 99.15, 94.0008],
                       [97.0703, 97.1105, 85.8059],
                       [97.7703, 97.1201, 74.8],
                       [98.86005, 97.13, 81.54556],
                       [100, 93.16685, 84.89624]]),
    'F1-score': np.array([[52.89, 75.56, 91.7],     
                         [93.43, 94.43, 78.0],
                         [95.87, 99.17, 94.2],
                         [97.19, 97.27, 85.3],
                         [97.82, 97.17, 74.8],
                         [98.91, 97.13, 82.56],
                         [100.0, 93.73, 84.38]]),
    'FE Time': np.array([[84.01, 1.94, 3.85],
                        [165.98, 2.09, 4.24],
                        [332.79, 2.12, 5.12],
                        [667.58, 1.99, 6.61],
                        [1355.67, 2.02, 7.62],
                        [2802, 1.98, 13.39],
                        [5879, 2.03, 19.05]])
}

# 设置柱状图的宽度
bar_width = 0.2
index = np.arange(len(windows))

# 绘制四个独立的图
metrics = list(data.keys())
for i, metric in enumerate(metrics):
    plt.figure(figsize=(6, 4))
    
    # 绘制柱状图
    for j in range(len(algorithms)):
        plt.bar(index + j * bar_width, data[metric][:, j], bar_width, 
                label=algorithms[j], alpha=1.0)
    
    # 设置刻度和标签
    plt.xticks(index + bar_width, windows, fontsize=14, rotation=15)
    
    # 添加标签
    plt.xlabel('Window Size', fontsize=16)
    
    if metric == 'FE Time':
        plt.yscale('log')
        plt.ylabel(f'{metric} (log scale, s)', fontsize=16)
        # FE Time 图例放在左上角
        plt.legend(fontsize=14, loc='upper left')
    else:
        plt.ylim(0, 100)  # 对于百分比指标设置y轴范围
        # 添加50%的水平虚线
        plt.axhline(y=50, color='gray', linestyle='--', linewidth=2)
        plt.ylabel(metric, fontsize=16)
        # 其他图表的图例放在右下角
        plt.legend(fontsize=14, loc='lower right')
    
    plt.tick_params(axis='y', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    # 保存为pdf格式
    plt.savefig(f'fig/Window-{metric}.pdf', format='pdf', dpi=300, transparent=False)
    plt.close()
