from collections import defaultdict
from multiprocessing import Manager
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
import ast
from gensim.models import Word2Vec


def read_file(file1, file2):
    cols = ['mean']
    df = pd.read_csv(file1, usecols=cols, index_col=False)
    df2 = pd.read_csv(file2, usecols=cols, index_col=False)
    return df, df2

def my_plot(df, df2):
    plt.figure(figsize=(10, 5))
    window_sizes = [256, 512, 1024]
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
    判断个高维空间中的点是否距离很近。
    参数:
    - point1: 第一个点的NumPy数组。
    - point2: 第二个点的NumPy数组。
    - num_bins: 每个维度的等分区间数，默认为1000。
    返回:
    - 如果两个点在所有维度上都处于相同的等分区间，则返回True；否则返回False。
    """
    # 将每个点的每维度的值映射到等分区间的索引
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

def calculate_features_chunk(chunk, window_size):
    """
    使用CPU处理的数据块处理函数
    """
    cnt_col = f'cnt{window_size}'
    mean1_col = f'mean{window_size}_1'
    mean2_col = f'mean{window_size}_2'
    
    value_columns = [col for col in chunk.columns if col.startswith('value_')]
    counts = []
    
    # 使用numpy数组替代GPU数组
    chunk_data = chunk[value_columns].values
    num_bins = 1000
    
    # 使用字典来记录bin的计数
    bin_counts = defaultdict(int)
    
    for idx in range(len(chunk)):
        current_row = chunk_data[idx]
        bin_idx = tuple((current_row * num_bins).astype(np.int32))
        
        # 获取当前bin的计数
        knn = bin_counts[bin_idx]
        bin_counts[bin_idx] += 1
        
        if idx >= window_size:
            outofdate_row = chunk_data[idx - window_size]
            old_bin_idx = tuple((outofdate_row * num_bins).astype(np.int32))
            bin_counts[old_bin_idx] -= 1
        
        counts.append(knn)
    
    chunk[cnt_col] = counts
    chunk[mean1_col] = chunk[cnt_col].rolling(window=window_size).mean()
    chunk[mean2_col] = chunk[mean1_col].rolling(window=window_size).mean()
    
    return chunk

def calculate_features_parallel(df, window_size, n_jobs=16):
    """
    并行计算特征
    """
    print(f'正在并行计算窗口大小为 {window_size} 的特征...')
    chunk_size = len(df) // n_jobs
    
    chunks = []
    results = []
    
    # 划分数据块
    for i in range(n_jobs):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        
        if i == 0:
            chunk = df.iloc[start_idx:end_idx].copy()
            chunks.append(chunk)
        else:
            # 对于非第一个块，包含前一个块的最后window_size个点
            chunk = df.iloc[start_idx-window_size:end_idx].copy()
            chunks.append(chunk)
    
    # 并行处理
    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_features_chunk)(chunk, window_size) 
        for i, chunk in enumerate(chunks)
    )
    
    # 合并结果
    final_df = df.copy()
    cols_to_update = [f'cnt{window_size}', f'mean{window_size}_1', f'mean{window_size}_2']
    
    for i, result_chunk in enumerate(results):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        
        if i == 0:
            # 第一个chunk直接使用全部结果
            final_df.loc[start_idx:end_idx-1, cols_to_update] = result_chunk[cols_to_update].iloc[:end_idx-start_idx].values
        else:
            # 非第一个chunk需要丢弃前window_size行的数据
            result_values = result_chunk[cols_to_update].iloc[window_size:window_size+(end_idx-start_idx)].values
            final_df.loc[start_idx:end_idx-1, cols_to_update] = result_values
    
    return final_df

def preprocess(infile, outfile):
    df = pd.read_csv(infile)

    literals_parsed = []
    for row in df['literals']:
        try:
            parsed = ast.literal_eval(row)
            literals_parsed.append(parsed)
        except (ValueError, SyntaxError) as e:
            print(f'解析错误: {e}')
            literals_parsed.append([])
    values_df = pd.DataFrame(literals_parsed, columns=[f'value_{i}' for i in range(len(literals_parsed[0]))])
    # 检查并处理非数值型列
    for col in values_df.columns:
        if not np.issubdtype(values_df[col].dtype, np.number):
            print(f'检测到非数值型列 {col}, 正在进行编码...')
            # 对非数值列进行标签编码
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            values_df[col] = le.fit_transform(values_df[col].astype(str))
            print(f'列 {col} 编码完成')
    # 数据预处理
    scaler = MinMaxScaler()
    values_df = pd.DataFrame(
        scaler.fit_transform(values_df),
        columns=values_df.columns
    )
    
    # 计算不同窗口大小的特征
    window_sizes = [256, 512, 1024]
    for window_size in window_sizes:
        start_time = time.time()
        values_df = calculate_features_parallel(values_df, window_size, n_jobs=32)
        end_time = time.time()
        print(f'窗口大小 {window_size} 的特征计算耗时: {end_time - start_time:.2f}秒')
    
    # 打印列名以进行调试
    print("可用的列名:", values_df.columns.tolist())
    
    values_df.to_csv(outfile, index=False)
    return values_df
