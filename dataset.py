import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import config
from feature_engineering import preprocess, my_plot

def ReadFileAsDataFrame(file):
    usecols = ['mean200_2', 'mean400_2', 'mean800_2']
    df = pd.read_csv(file, usecols=usecols, index_col=False)
    df = df.dropna(subset=usecols)
    return df

# get data from csv file as a list, in which every item is a tuple(nparray[][],nparray[])
def PrepareData():
    if config.NEED_CALC_FEATURES == 1:
        print('开始处理正样本数据...')
        positive_df = preprocess(config.POSITIVE_FILE, config.POSITIVE_FEATURES)
        print('正样本数据处理完成')
        
        print('开始处理负样本数据...')
        negative_df = preprocess(config.NEGATIVE_FILE, config.NEGATIVE_FEATURES)
        print('负样本数据处理完成')
        
        print('开始绘制特征分布图...')
        my_plot(positive_df, negative_df)
        print('特征工程全部完成')

    positive_df = ReadFileAsDataFrame(config.POSITIVE_FEATURES)
    negative_df = ReadFileAsDataFrame(config.NEGATIVE_FEATURES)
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
