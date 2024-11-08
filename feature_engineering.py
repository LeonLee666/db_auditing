import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import config

def read_file(file1, file2):
    cols = ['mean']
    df = pd.read_csv(file1, usecols=cols, index_col=False)
    df2 = pd.read_csv(file2, usecols=cols, index_col=False)
    return df, df2

def my_plot(df, df2):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['mean2'], label='positive', marker='o')
    plt.plot(df2.index, df2['mean2'], label='negative', marker='x')
    plt.legend()
    plt.title('feature sequences')
    plt.xlabel('Index')
    plt.ylabel('feature values')
    plt.grid()
    plt.show()

def neighbor_count(df, index, threshold=0.001):
    # 计算前后各20行的范围
    start = max(0, index - 100)
    end = min(len(df), index + 101)
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

def process(infile, outfile):
    scaler = MinMaxScaler()
    df = pd.read_csv(infile, usecols=['value1', 'value2'], index_col=False)
    df['value1'] = pd.to_numeric(df['value1'], errors='coerce')
    df['value2'] = pd.to_numeric(df['value2'], errors='coerce')
    df[['value1']] = scaler.fit_transform(df[['value1']])
    df[['value2']] = scaler.fit_transform(df[['value2']])
    # 并行应用
    df['cnt'] = Parallel(n_jobs=-1)(
        delayed(neighbor_count)(df, idx) for idx in tqdm(df.index)
    )
    df['mean1'] = df['cnt'].rolling(window=200).mean()
    df['mean2'] = df['mean1'].rolling(window=200).mean()
    df.to_csv(outfile)
    return df

def extract_features():
    df = process(config.POSITIVE_FILE, config.POSITIVE_FEATURES)
    df2 = process(config.NEGATIVE_FILE, config.NEGATIVE_FEATURES)
    my_plot(df, df2)
