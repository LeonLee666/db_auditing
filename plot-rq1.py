import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 读取CSV文件
df = pd.read_csv('rq1_data.csv')

# 数据集和算法
labels = ['TPC-C', 'Voter', 'YCSB', 'Twitter', 'Wikipedia']
algorithms = ['LSTM-Euler', 'LSTM-HASH-Grid', 'LSTM-Centroid']

# 设置柱状图的宽度
bar_width = 0.2
index = np.arange(len(labels))

# 获取所有唯一的指标
metrics = df['Metric'].unique()

# 绘制图表
for metric in metrics:
    # 创建新的图形
    plt.figure(figsize=(6, 4))
    
    # 获取当前指标的数据
    metric_data = df[df['Metric'] == metric]
    
    # 绘制柱状图
    for j, algorithm in enumerate(algorithms):
        values = metric_data[algorithm].values
        plt.bar(index + j * bar_width, values, bar_width, 
                label=algorithm, alpha=1.0)
    
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
