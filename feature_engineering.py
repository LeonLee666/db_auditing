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

hashmap200 = defaultdict(int)
hashmap400 = defaultdict(int)
hashmap800 = defaultdict(int)

def nn_cnt200(df, index):
    num_bins = 1000
    current_row = df.iloc[index][['value1', 'value2']].values
    bin_idx = (current_row * num_bins).astype(int)
    hashmap200[tuple(bin_idx)] += 1
    if index >= 200:
        outofdate_row = df.iloc[index - 200][['value1', 'value2']].values
        bin_idx2 = (outofdate_row * num_bins).astype(int)
        hashmap200[tuple(bin_idx2)] -= 1
    return hashmap200[tuple(bin_idx)]

def nn_cnt400(df, index):
    num_bins = 1000
    current_row = df.iloc[index][['value1', 'value2']].values
    bin_idx = (current_row * num_bins).astype(int)
    hashmap400[tuple(bin_idx)] += 1
    if index >= 400:
        outofdate_row = df.iloc[index - 400][['value1', 'value2']].values
        bin_idx2 = (outofdate_row * num_bins).astype(int)
        hashmap400[tuple(bin_idx2)] -= 1
    return hashmap400[tuple(bin_idx)]

def nn_cnt800(df, index):
    num_bins = 1000
    current_row = df.iloc[index][['value1', 'value2']].values
    bin_idx = (current_row * num_bins).astype(int)
    hashmap800[tuple(bin_idx)] += 1
    if index >= 800:
        outofdate_row = df.iloc[index - 800][['value1', 'value2']].values
        bin_idx2 = (outofdate_row * num_bins).astype(int)
        hashmap800[tuple(bin_idx2)] -= 1
    return hashmap800[tuple(bin_idx)]

def preprocess(infile, outfile):
    scaler = MinMaxScaler()
    df = pd.read_csv(infile, usecols=['value1', 'value2'], index_col=False)
    df['value1'] = pd.to_numeric(df['value1'], errors='coerce')
    df['value2'] = pd.to_numeric(df['value2'], errors='coerce')
    df[['value1']] = scaler.fit_transform(df[['value1']])
    df[['value2']] = scaler.fit_transform(df[['value2']])
    df['cnt200'] = Parallel(n_jobs=-1)(
        delayed(nn_cnt200)(df, idx) for idx in tqdm(df.index)
    )
    df['mean1'] = df['cnt200'].rolling(window=200).mean()
    df['mean2'] = df['mean1'].rolling(window=200).mean()
    df['cnt400'] = Parallel(n_jobs=-1)(
        delayed(nn_cnt400)(df, idx) for idx in tqdm(df.index)
    )
    df['mean3'] = df['cnt400'].rolling(window=400).mean()
    df['mean4'] = df['mean3'].rolling(window=400).mean()
    df['cnt800'] = Parallel(n_jobs=-1)(
        delayed(nn_cnt800)(df, idx) for idx in tqdm(df.index)
    )
    df['mean5'] = df['cnt800'].rolling(window=800).mean()
    df['mean6'] = df['mean5'].rolling(window=800).mean()
    df.to_csv(outfile)
    return df

def extract_features():
    df = preprocess(config.POSITIVE_FILE, config.POSITIVE_FEATURES)
    df2 = preprocess(config.NEGATIVE_FILE, config.NEGATIVE_FEATURES)
    my_plot(df, df2)
