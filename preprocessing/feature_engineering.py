from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors

import config

def read_file(file1, file2):
    cols = ['mean']
    df = pd.read_csv(file1, usecols=cols, index_col=False)
    df2 = pd.read_csv(file2, usecols=cols, index_col=False)
    return df, df2

def my_plot(df, df2):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['mean2'], label='mean2-positive', marker='o')
    plt.plot(df2.index, df2['mean2'], label='mean2-negative', marker='x')
    plt.plot(df.index, df['mean4'], label='mean4-positive', marker='v')
    plt.plot(df2.index, df2['mean4'], label='mean4-negative', marker=',')
    plt.plot(df.index, df['mean6'], label='mean6-positive', marker='*')
    plt.plot(df2.index, df2['mean6'], label='mean6-negative', marker='.')
    plt.legend()
    plt.title('feature sequences')
    plt.xlabel('Index')
    plt.ylabel('feature values')
    plt.grid()
    # 保存为 PNG 文件
    plt.savefig('my_plot.png')
    # 关闭图形
    plt.close()

def are_points_close(point1, point2, num_bins=1000):
    """
    判断两个高维空间中的点是否距离很近。
    参数:
    - point1: 第一个点的NumPy数组。
    - point2: 第二个点的NumPy数组。
    - num_bins: 每个维度的等分区间数，默认为1000。
    返回:
    - 如果两个点在所有维度上都处于相同的等分区间，则返回True；否则返回False。
    """
    # 将每个点的每个维度的值映射到等分区间的索引
    bin_indices1 = (point1 * num_bins).astype(int)
    bin_indices2 = (point2 * num_bins).astype(int)
    # 检查所有维度是否都在相同的等分区间内
    for idx1, idx2 in zip(bin_indices1, bin_indices2):
        if idx1 != idx2:
            return False
    return True

def neighbor_count(df, index, threshold=0.001):
    start = max(0, index - 200)
    end = min(len(df), index)
    # 获取当前行的值作为一个空间点
    current_row = df.iloc[index].values
    # 计算符合条件的个数
    count = 0
    for i in range(start, end):
        if i != index:  # 排除自身
            neighbor_row = df.iloc[i].values
            # 计算二范式距离
            distance = np.linalg.norm(current_row - neighbor_row)
            if distance <= threshold:
                count += 1
    return count

# 创建一个字典来存储不同窗口大小的 hashmap
hashmaps = {
    200: defaultdict(int),
    400: defaultdict(int),
    800: defaultdict(int)
}

def nn_cnt(df, index, window_size, n_neighbors=5):
    """
    使用局部敏感哈希（LSH）来近似查找近邻
    """
    start = max(0, index - window_size)
    # 只选择BERT编码的列
    bert_columns = [f'feature_{i}' for i in range(768)]
    current_row = df.iloc[index][bert_columns].values.reshape(1, -1)
    
    # 使用 NearestNeighbors 查找近邻
    if len(df.iloc[start:index]) == 0:
        return 0
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(df.iloc[start:index][bert_columns].values)
    distances, indices = nbrs.kneighbors(current_row)
    
    # 返回距离小于某个阈值的近邻个数
    threshold = 0.1  # 可以根据实际数据分布调整这个阈值
    return np.sum(distances <= threshold)

def calculate_features(df, window_size):
    """
    计算特定窗口大小的特征，适用于高维数据
    """
    cnt_col = f'cnt{window_size}'
    mean1_col = f'mean{window_size}_1'
    mean2_col = f'mean{window_size}_2'
    
    df[cnt_col] = Parallel(n_jobs=-1)(
        delayed(nn_cnt)(df, idx, window_size) for idx in tqdm(df.index)
    )
    df[mean1_col] = df[cnt_col].rolling(window=window_size).max()
    df[mean2_col] = df[mean1_col].rolling(window=window_size).mean()
    
    return df
