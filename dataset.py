import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import config
from tqdm import tqdm

def neighbor_count(df, index, threshold=0.001):
    # 计算前后各10行的范围
    start = max(0, index - 20)
    end = min(len(df), index + 21)
    current_value = df.at[index, 'value2']
    # 计算符合条件的个数
    count = ((df['value2'][start:end] - current_value).abs() <= threshold).sum()
    return count

def ReadFileAsDataFrame(file):
    df = pd.read_csv(file, usecols=['value2'], index_col=False)
    df['value2'] = pd.to_numeric(df['value2'], errors='coerce')
    scaler = MinMaxScaler()
    df[['value2']] = scaler.fit_transform(df[['value2']])
    tqdm.pandas()
    print(f"preprocessing data for {file}")
    df['cnt'] = df.index.to_series().progress_apply(lambda idx: neighbor_count(df=df, index=idx, threshold=0.001))
    df['mean'] = df['cnt'].rolling(window=100).mean()
    df = df.dropna(subset=['mean'])
    df = df.drop(columns=['value2','cnt'])
    return df

# get data from csv file as a list, in which every item is a tuple(nparray[][],nparray[])
def PrepareData():
    positive_df = ReadFileAsDataFrame(config.SPIDER_FILE)
    negative_df = ReadFileAsDataFrame(config.NORMAL_FILE)
    data_list = []
    cnt = 0
    for i in range(0, len(positive_df) - config.SEQ_LENGTH + 1, 10):
        cnt = cnt + 1
        sample = positive_df.iloc[i:i + config.SEQ_LENGTH]
        sample = sample.to_numpy()
        data_list.append((sample, 1))
        if cnt == config.POSITIVE_SIZE:
            break
    cnt = 0
    for i in range(0, len(negative_df) - config.SEQ_LENGTH + 1, 10):
        cnt = cnt + 1
        sample = negative_df.iloc[i:i + config.SEQ_LENGTH]
        sample = sample.to_numpy()
        data_list.append((sample, 0))
        if cnt == config.NEGATIVE_SIZE:
            break
    train_set, test_set = train_test_split(data_list, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.25)
    return train_set, val_set, test_set

# define dataset, the input args for dataset type is a list of (sample,label)
class DataWrapper(Dataset):
    def __init__(self, dataset):
        # the type of dataset is a list of tuple(sample, lable)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, encode_label = self.dataset[idx]
        return torch.Tensor(sample), torch.tensor(encode_label, dtype=torch.int64)
