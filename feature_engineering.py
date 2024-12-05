from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import config

def read_file(file1, file2):
    cols = ['mean']
    df = pd.read_csv(file1, usecols=cols, index_col=False)
    df2 = pd.read_csv(file2, usecols=cols, index_col=False)
    return df, df2

def my_plot(df, df2):
    plt.figure(figsize=(10, 5))
    window_sizes = [200, 400, 800]
    markers = ['o', 'x', 'v', ',']
    
    for i, window_size in enumerate(window_sizes):
        mean_col = f'mean{window_size}_2'
        plt.plot(df.index, df[mean_col], label=f'mean{window_size}-positive', marker=markers[i])
        plt.plot(df2.index, df2[mean_col], label=f'mean{window_size}-negative', marker=markers[i+1])
    
    plt.legend()
    plt.title('Feature Sequences')
    plt.xlabel('Index')
    plt.ylabel('Feature Values')
    plt.grid()
    plt.savefig('my_plot.png')
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

def nn_cnt(df, index, window_size):
    """
    统一的近邻计数函数，替代原来的三个独立函数
    """
    num_bins = 1000
    current_row = df.iloc[index][['value1', 'value2']].values
    bin_idx = (current_row * num_bins).astype(int)
    hashmap = hashmaps[window_size]
    hashmap[tuple(bin_idx)] += 1
    
    if index >= window_size:
        outofdate_row = df.iloc[index - window_size][['value1', 'value2']].values
        bin_idx2 = (outofdate_row * num_bins).astype(int)
        hashmap[tuple(bin_idx2)] -= 1
    
    return hashmap[tuple(bin_idx)]

def calculate_features(df, window_size):
    """
    计算特定窗口大小的特征
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

def preprocess(infile, outfile):
    scaler = MinMaxScaler()
    df = pd.read_csv(infile, usecols=['value1', 'value2'], index_col=False)
    
    # 数据预处理
    for col in ['value1', 'value2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[[col]] = scaler.fit_transform(df[[col]])
    
    # 计算不同窗口大小的特征
    window_sizes = [200, 400, 800]
    for window_size in window_sizes:
        df = calculate_features(df, window_size)
    
    df.to_csv(outfile)
    return df

def extract_features():
    df = preprocess(config.POSITIVE_FILE, config.POSITIVE_FEATURES)
    df2 = preprocess(config.NEGATIVE_FILE, config.NEGATIVE_FEATURES)
    my_plot(df, df2)
