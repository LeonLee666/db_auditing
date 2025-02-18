import numpy as np
import matplotlib.pyplot as plt

# 数据密度和算法
alphas = ['1.2', '1.4', '1.6', '1.8', '2.0']  # 新的x轴值：zipf参数alpha
algorithms = ['LSTM-Euler', 'LSTM-HASH-Grid', 'LSTM-Centroid']

# 示例数据，每个指标对应一个二维数组：行表示不同的alpha值，列表示不同的算法
data = {
    'Precision': np.array([[49.8, 50.55, 94.7],    
                          [50.28, 50, 99.3],
                          [49.79, 50.1, 98.98],
                          [50.1, 57.9, 100],
                          [49.9, 49.47, 94.6]]),
    'Recall': np.array([[54.9093, 76.5479, 94.9002],    
                       [50.94436, 95.68765, 99.3],
                       [53.7, 88.04034, 98.96],
                       [48.44 , 87.85696, 100],
                       [46.78064, 90.54, 94.8004]]),
    'F1-score': np.array([[52.23, 60.89, 94.8],     
                         [50.61, 65.68, 99.3],
                         [51.24, 63.86, 98.97],
                         [48.72, 69.8, 100],
                         [48.29, 63.98, 94.7]]),
    'FE Time': np.array([[2844, 2.03, 18.7],
                        [2839, 2.03, 10.37],
                        [2822.81, 1.99, 8.18],
                        [2814, 2, 7.64],
                        [2814.33, 2, 7.39]])
}

# 设置柱状图的宽度
bar_width = 0.2
index = np.arange(len(alphas))

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
        plt.xticks(index + bar_width, alphas, fontsize=14, rotation=15)
    
    if metric != 'FE Time':
        plt.xticks(index, alphas, fontsize=14, rotation=15)
    
    # 添加标签
    plt.xlabel('Zipf Parameter α', fontsize=16)  # 更新x轴标签
    if metric == 'FE Time':
        plt.ylabel(f'{metric} (log scale, s)', fontsize=16)
    else:
        plt.ylabel(metric, fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    
    # 调整布局
    plt.tight_layout()
    # 保存为pdf格式
    plt.savefig(f'fig/RQ2-Zipf-{metric}.pdf', format='pdf', dpi=300, transparent=False)
    plt.close()
