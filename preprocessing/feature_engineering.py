import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors

def nn_cnt(df, index, window_size, n_neighbors=5, threshold=0.3):
    """
    计算窗口内与当前样本距离大于阈值的样本数量
    """
    if index >= len(df):
        return 0
        
    bert_columns = [f'feature_{i}' for i in range(768)]
    current_row = df.iloc[index][bert_columns].values.reshape(1, -1)
    
    # Get data within the window
    start = max(0, index - window_size)
    window_data = df.iloc[start:index][bert_columns].values
    
    if len(window_data) < 1:
        return 0
        
    actual_n_neighbors = min(n_neighbors, len(window_data))
    
    try:
        # 移除并行处理
        nbrs = NearestNeighbors(
            n_neighbors=actual_n_neighbors,
            algorithm='kd_tree',
            metric='cosine',
            leaf_size=30,
            n_jobs=40
        ).fit(window_data)
        
        distances, _ = nbrs.kneighbors(current_row)
        return np.sum(distances[0] > threshold)
    except ValueError:
        return 0

def calculate_features(df, window_size):
    """
    Calculate features for a specific window size, suitable for high-dimensional data
    """
    cnt_col = f'cnt{window_size}'
    mean1_col = f'mean{window_size}_1'
    mean2_col = f'mean{window_size}_2'
    
    # 串行计算
    cnt_values = [nn_cnt(df, idx, window_size) for idx in tqdm(range(len(df)))]
    
    # 计算滚动平均值
    mean1_values = pd.Series(cnt_values).rolling(window=window_size).mean()
    mean2_values = mean1_values.rolling(window=window_size).mean()
    
    # 使用pd.concat一次性添加所有列
    new_features = pd.DataFrame({
        cnt_col: cnt_values,
        mean1_col: mean1_values,
        mean2_col: mean2_values
    })
    
    # 合并原始DataFrame和新特征
    result = pd.concat([df, new_features], axis=1)
    return result

def find_reasonable_threshold(df, sample_size=1000):
    # Get BERT feature columns
    bert_columns = [f'feature_{i}' for i in range(768)]
    
    # Random sampling
    total_samples = len(df)
    if total_samples < sample_size:
        sample_data = df[bert_columns].values
    else:
        # 随机选择起始索引，确保有足够空间取sample_size个连续样本
        start_idx = np.random.randint(0, total_samples - sample_size)
        # 获取从start_idx开始的连续sample_size个样本
        sample_data = df.iloc[start_idx:start_idx + sample_size][bert_columns].values
    
    # Ensure actual sample size
    actual_sample_size = len(sample_data)
    if actual_sample_size < 2:
        raise ValueError(f"Sample size too small, currently only {actual_sample_size} samples")
    
    nbrs = NearestNeighbors(n_neighbors=min(10, actual_sample_size), algorithm='ball_tree')
    nbrs.fit(sample_data)
    
    # Calculate distances
    distances, _ = nbrs.kneighbors(sample_data)
    
    # Analyze distance distribution
    print(f"Using {actual_sample_size} randomly sampled data points")
    print(f"Distance statistics:\n"
          f"Minimum: {distances[:, 1:].min()}\n"  # Exclude self-distance
          f"5th percentile: {np.percentile(distances[:, 1:], 5)}\n"
          f"15th: {np.percentile(distances[:, 1:], 15)}\n"
          f"25th percentile: {np.percentile(distances[:, 1:], 25)}\n"
          f"Maximum: {distances[:, 1:].max()}")
    
    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(distances[:, 1:].flatten(), bins=50)
    plt.title(f'Distance Distribution Histogram (Random samples: {actual_sample_size})')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    
    # Save plot to file instead of displaying
    plt.savefig('distance_distribution.png')
    plt.close()  # Close plot to free memory
    
    return np.percentile(distances[:, 1:], 25)  # Suggest using 25th percentile as threshold
