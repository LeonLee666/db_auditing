from collections import defaultdict
from multiprocessing import Manager
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
import config
from sql_extract import extract_sql_file

def plot_features(df, df2):
    # 计算需要的子图行数和列数
    n_plots = len(config.WINDOW_SIZES)
    n_cols = 2  # 每行2个子图
    n_rows = (n_plots + 1) // 2  # 向上取整
    # 创建一个更大的图形以容纳所有子图
    plt.figure(figsize=(15, 5*n_rows))
    markers = ['o', 'x']
    
    for i, window_size in enumerate(config.WINDOW_SIZES):
        # 创建子图
        plt.subplot(n_rows, n_cols, i+1)
        mean_col = f'mean{window_size}_2'
        plt.plot(df.index, df[mean_col], label='positive', marker=markers[0], markersize=4)
        plt.plot(df2.index, df2[mean_col], label='negative', marker=markers[1], markersize=4)
        plt.legend()
        plt.title(f'Window Size = {window_size}')
        plt.xlabel('Index')
        plt.ylabel('Feature Values')
        plt.grid(True)
        # plt.xlim(20000, 30000)
    
    # 调整子图之间的间距
    plt.tight_layout()
    plt.savefig('plot_features.png')
    plt.close()

def neighbor_count(df, index, window_size, threshold=0.001):
    start = max(0, index - window_size)
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
    
    chunk_result = chunk.copy()
    chunk_result[cnt_col] = counts
    chunk_result[mean1_col] = chunk_result[cnt_col].rolling(window=window_size).mean()
    chunk_result[mean2_col] = chunk_result[mean1_col].rolling(window=window_size).mean()
    
    return chunk_result

def calculate_features_chunk_neighbor(chunk, window_size):
    """
    使用neighbor_count方法的数据块处理函数
    """
    cnt_col = f'cnt{window_size}'
    mean1_col = f'mean{window_size}_1'
    mean2_col = f'mean{window_size}_2'
    
    counts = []
    for idx in range(len(chunk)):
        count = neighbor_count(
            chunk, 
            idx,
            window_size,
            threshold=0.001
        )
        counts.append(count)
    
    chunk[cnt_col] = counts
    chunk[mean1_col] = chunk[cnt_col].rolling(window=window_size).mean()
    chunk[mean2_col] = chunk[mean1_col].rolling(window=window_size).mean()
    return chunk

def calculate_features_parallel(df, window_size, n_jobs=32):
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
    
    # 根据配置选择处理函数
    process_func = (calculate_features_chunk_neighbor 
                   if config.FEATURE_ALGORITHM == 'neighbor_count' 
                   else calculate_features_chunk)
    
    # 并行处理
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_func)(chunk, window_size) 
        for i, chunk in enumerate(chunks)
    )
    
    # 定义需要更新的列名
    cols_to_update = [f'cnt{window_size}', f'mean{window_size}_1', f'mean{window_size}_2']
    
    # 初始化一个空的DataFrame来存储结果
    final_df = pd.DataFrame()
    
    # 合并结果
    for i, result_chunk in enumerate(results):
        if i == 0:
            # 第一个chunk直接添加
            final_df = pd.concat([final_df, result_chunk[cols_to_update]], ignore_index=True)
        else:
            # 非第一个chunk需要丢弃前window_size行的数据
            chunk_start = window_size
            final_df = pd.concat([final_df, result_chunk[cols_to_update].iloc[chunk_start:]], ignore_index=True)
    
    # 确保结果长度正确
    if len(final_df) > len(df):
        final_df = final_df.iloc[:len(df)]
    elif len(final_df) < len(df):
        # 如果长度不足，用最后一行的值填充
        rows_to_add = len(df) - len(final_df)
        last_row = final_df.iloc[[-1]]
        final_df = pd.concat([final_df] + [last_row] * rows_to_add, ignore_index=True)
        final_df = final_df.iloc[:len(df)]
    
    for col in cols_to_update:
        df[col] = final_df[col].values
    
    return df

def extract_and_parse_sql_file(infile):
    """
    从SQL文件中提取数据并将literals列转换为数值列
    
    参数:
    infile: SQL文件路径
    
    返回:
    DataFrame: 包含解析后的数值列（value_0, value_1, ...）
    """
    # 添加计时开始
    start_time = time.time()
    
    # 提取原始数据
    df = extract_sql_file(infile)
    
    # 获取最大长度并创建结果数组
    max_length = max(len(x) for x in df['literals'])
    result = np.zeros((len(df), max_length))
    
    # 填充数据
    for i, row in enumerate(df['literals']):
        result[i, :len(row)] = row
    
    # 计算耗时并打印
    end_time = time.time()
    print(f'Time taken for SQL file extraction and parsing: {end_time - start_time:.2f} seconds')
    
    # 创建并返回DataFrame
    return pd.DataFrame(
        result,
        columns=[f'value_{i}' for i in range(max_length)]
    )

def preprocess(infile, outfile):
    # 使用新的合并函数
    values_df = extract_and_parse_sql_file(infile)
    
    # 数据预处理
    scaler = MinMaxScaler()
    values_df = pd.DataFrame(
        scaler.fit_transform(values_df),
        columns=values_df.columns
    )
    
    # 计算不同窗口大小的特征
    for window_size in config.WINDOW_SIZES:
        start_time = time.time()
        values_df = calculate_features_parallel(values_df, window_size, n_jobs=config.JOBS)
        end_time = time.time()
        print(f'Time taken for window size {window_size}: {end_time - start_time:.2f} seconds')
    
    # 打印列名以进行调试
    print("Available columns:", values_df.columns.tolist())
    
    values_df.to_csv(outfile, index=False)
    return values_df
