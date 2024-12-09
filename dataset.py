import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import config
from preprocessing.encode_feature import encode_and_extract_features
import ipdb

def PrepareData():
    usecols = ['mean200_2', 'mean400_2', 'mean800_2']
    if config.REGENERATE_FEATURES:
        positive_features = encode_and_extract_features(config.POSITIVE_FILE)
        positive_features.dropna(inplace=True)
        positive_features.to_csv('positive_features.csv', index=False)
        positive_features = positive_features[usecols]
        negative_features = encode_and_extract_features(config.NEGATIVE_FILE)
        negative_features.dropna(inplace=True)
        negative_features.to_csv('negative_features.csv', index=False)
        negative_features = negative_features[usecols]
    else:
        # 从保存的CSV文件中读取特征
        positive_features = pd.read_csv('positive_features.csv', usecols=usecols, index_col=False)
        negative_features = pd.read_csv('negative_features.csv', usecols=usecols, index_col=False)
    print(f"Feature整理已经完成: 正样本数量={len(positive_features)}, 负样本数量={len(negative_features)}")
    
    data_list = []
    cnt = 0
    for i in range(0, len(positive_features) - config.SEQ_LENGTH + 1, 10):
        cnt = cnt + 1
        sample = positive_features.iloc[i:i + config.SEQ_LENGTH]
        sample = sample.to_numpy()
        data_list.append((sample, 1))
        if cnt == config.POSITIVE_SIZE:
            break
    print(f"正样本已经处理完成，共{cnt}个样本")
    cnt = 0
    for i in range(0, len(negative_features) - config.SEQ_LENGTH + 1, 10):
        cnt = cnt + 1
        sample = negative_features.iloc[i:i + config.SEQ_LENGTH]
        sample = sample.to_numpy()
        data_list.append((sample, 0))
        if cnt == config.NEGATIVE_SIZE:
            break
    print(f"负样本已经处理完成，共{cnt}个样本")
    train_set, test_set = train_test_split(data_list, test_size=0.2)
    train_set, val_set = train_test_split(train_set, test_size=0.25)
    return train_set, val_set, test_set

class DataWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample, encode_label = self.dataset[idx]
        return torch.Tensor(sample), torch.tensor(encode_label, dtype=torch.int64)
