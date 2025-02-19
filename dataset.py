import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import config
from feature_engineering import extract_and_parse_sql_file, preprocess, plot_features

def PrepareData():
    if config.VANILLA_MODE:
        print('使用原始特征处理样本...')
        positive_df = extract_and_parse_sql_file(config.POSITIVE_FILE)
        negative_df = extract_and_parse_sql_file(config.NEGATIVE_FILE)
        # 只选择 value_0 和 value_1
        usecols = ['value_0', 'value_1']
    else:
        print('Processing positive samples...')
        positive_df = preprocess(config.POSITIVE_FILE)
        print('Positive samples processing completed')
        
        print('Processing negative samples...')
        negative_df = preprocess(config.NEGATIVE_FILE)
        print('Negative samples processing completed')
        
        print('Starting to plot feature distributions...')
        plot_features(positive_df, negative_df)
        print('Feature engineering completed')

        usecols = [f'mean{size}_2' for size in config.WINDOW_SIZES]
        
        # 根据配置决定是否加载grid特征
        if config.FEATURE_ALGORITHM == 'centroid':
            usecols = []
            grid_mean_cols = []
            for size in config.WINDOW_SIZES:
                grid_mean_base = f'grid_centroid{size}'
                for dim in range(len([col for col in positive_df.columns 
                                    if col.startswith('value_')])):
                    grid_mean_cols.append(f'{grid_mean_base}_dim_{dim}_mean2')
            usecols.extend(grid_mean_cols)
    
    # 直接使用已处理的DataFrame，并选择所需列
    positive_df = positive_df[usecols].dropna(subset=usecols)
    negative_df = negative_df[usecols].dropna(subset=usecols)
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
    return train_set, test_set
