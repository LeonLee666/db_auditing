from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import ast
from gensim.models import Word2Vec

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
    判断��个高维空间中的点是否距离很近。
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
    # 选择所有以 'value' 开头的列
    value_columns = [col for col in df.columns if col.startswith('value_')]
    current_row = df.iloc[index][value_columns].values
    bin_idx = (current_row * num_bins).astype(int)
    hashmap = hashmaps[window_size]
    hashmap[tuple(bin_idx)] += 1
    
    if index >= window_size:
        outofdate_row = df.iloc[index - window_size][value_columns].values
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
    
    print(f'正在计算窗口大小为 {window_size} 的特征...')
    # 移除并行处理，使用普通的列表推导式确保顺序性
    df[cnt_col] = [nn_cnt(df, idx, window_size) for idx in tqdm(df.index)]
    df[mean1_col] = df[cnt_col].rolling(window=window_size).max()
    df[mean2_col] = df[mean1_col].rolling(window=window_size).mean()
    
    return df

def preprocess(infile, outfile):
    df = pd.read_csv(infile)
    print(f'df.shape: {df.shape}')
    
    # 将literals转换为词嵌入并同时转换为字符串
    print('正在转换literals为词嵌入...')
    literals_str = [[str(item) for item in ast.literal_eval(row)] 
                   for row in tqdm(df['literals'], desc='转换literals并处理样本')]
    vector_size = 1
    w2v_model = Word2Vec(sentences=literals_str, 
                        vector_size=vector_size,  
                        window=1,
                        min_count=1,
                        workers=4)
    
    # 保留所有词向量而不是取平均值
    embedded_vectors = []
    max_sequence_length = max(len(sequence) for sequence in literals_str)
    
    for sequence in tqdm(literals_str, desc='生成词向量'):
        # 获取每个token的词向量
        sequence_vectors = [w2v_model.wv[token] for token in sequence]
        # 填充序列至最大长度
        if len(sequence_vectors) < max_sequence_length:
            padding = [np.zeros(32) for _ in range(max_sequence_length - len(sequence_vectors))]
            sequence_vectors.extend(padding)
        # 将所有向量展平
        flat_vector = np.concatenate(sequence_vectors)
        embedded_vectors.append(flat_vector)
    # 修改列名生成逻辑以匹配新的维度
    values_df = pd.DataFrame(
        embedded_vectors,
        columns=[f'value_{i+1}' for i in range(max_sequence_length * vector_size)]
    )
    
    # 数据预处理
    scaler = MinMaxScaler()
    for col in values_df.columns:
        # 检查并转换 category 类型
        if values_df[col].dtype.name == 'category':
            values_df[col] = values_df[col].cat.codes
        values_df[col] = pd.to_numeric(values_df[col], errors='coerce')
        values_df[[col]] = scaler.fit_transform(values_df[[col]])
    
    # 计算不同窗口大小的特征
    window_sizes = [200, 400, 800]
    for window_size in window_sizes:
        df = calculate_features(values_df, window_size)
    
    df.to_csv(outfile)
    return df

def extract_features():
    print('开始处理正样本数据...')
    df = preprocess(config.POSITIVE_FILE, config.POSITIVE_FEATURES)
    print('正样本数据处理完成')
    
    print('开始处理负样本数据...')
    df2 = preprocess(config.NEGATIVE_FILE, config.NEGATIVE_FEATURES)
    print('负样本数据处理完成')
    
    print('开始绘制特征分布图...')
    my_plot(df, df2)
    print('特征工程全部完成')
