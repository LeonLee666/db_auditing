import pandas as pd
from .feature_engineering import calculate_features
from .encoder import encode_texts_in_batches
from .sql_extract import extract_sql_file 

def encode_and_extract_features(input_file, batch_size=32):
    # 直接读取CSV文件
    # df = pd.read_csv(input_file, usecols=['sql'], index_col=False)
    df = extract_sql_file(df)
    sql_list = df['sql'].tolist()
    
    # 使用encoder中的函数进行编码，直接获得DataFrame
    df_encoded = encode_texts_in_batches(sql_list, batch_size=batch_size)
    
    # 计算特征
    df_features = df_encoded
    # 计算所有窗口大小的特征
    for window_size in [200, 400, 800]:
        df_features = calculate_features(df_features, window_size)
    
    # 只保留特征列
    feature_cols = [col for col in df_features.columns if col.startswith(('cnt', 'mean'))]
    df_features = df_features[feature_cols]
    
    return df_features
