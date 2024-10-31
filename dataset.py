import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import config

def ReadFileAsDataFrame(file):
    df = pd.read_csv(file, usecols=['value2'], index_col=False)
    # df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    # df['nsTime'] = pd.to_numeric(df['nsTime'], errors='coerce')
    # df['value1'] = pd.to_numeric(df['value1'], errors='coerce')
    df['value2'] = pd.to_numeric(df['value2'], errors='coerce')
    scaler = MinMaxScaler()
    df[['value2']] = scaler.fit_transform(df[['value2']])
    return df


# get data from csv file as a list, in which every item is a tuple(nparray[][],nparray[])
def PrepareData():
    positive_df = ReadFileAsDataFrame(config.SPIDER_FILE)
    negative_df = ReadFileAsDataFrame(config.NORMAL_FILE)
    positive_list = []
    cnt = 0
    for i in range(len(positive_df) - config.SEQ_LENGTH + 1):
        cnt = cnt + 1
        sample = positive_df.iloc[i:i + config.SEQ_LENGTH]
        sample = sample.to_numpy()
        positive_list.append((sample, 1))
        if cnt == config.POSITIVE_SIZE:
            break

    negative_list = []
    cnt = 0
    for i in range(len(negative_df) - config.SEQ_LENGTH + 1):
        cnt = cnt + 1
        sample = negative_df.iloc[i:i + config.SEQ_LENGTH]
        sample = sample.to_numpy()
        negative_list.append((sample, 0))
        if cnt == config.NEGATIVE_SIZE:
            break
    positive_train, positive_test = train_test_split(positive_list, test_size=0.2)
    negative_train, negative_test = train_test_split(negative_list, test_size=0.2)
    train_set = positive_train + negative_train
    test_set = positive_test + negative_test
    return train_set, test_set

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
