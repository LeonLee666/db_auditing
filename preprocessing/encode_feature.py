import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .feature_engineering import calculate_features, find_reasonable_threshold
from .encoder import encode_texts_in_batches
from .sql_extract import extract_sql_file 

def encode_and_extract_features(input_file, batch_size=32):
    # 直接读取CSV文件
    df = pd.read_csv(input_file, usecols=['sql'], index_col=False)
    # df = extract_sql_file(df)
    sql_list = df['sql'].tolist()
    
    # 使用encoder中的函数进行编码，直接获得DataFrame
    df_encoded = encode_texts_in_batches(sql_list, batch_size=batch_size)
    # 对每一列进行归一化处理
    scaler = MinMaxScaler()
    feature_columns = [f'feature_{i}' for i in range(768)]
    df_encoded[feature_columns] = scaler.fit_transform(df_encoded[feature_columns])
    
    # 计算特征
    df_features = df_encoded

    threshold = find_reasonable_threshold(df_features, sample_size=1000)

    print(f"合理的阈值: {threshold}")
    # 计算所有窗口大小的特征
    for window_size in [200, 400, 800]:
        print(f"正在计算窗口大小为 {window_size} 的特征...")
        df_features = calculate_features(df_features, window_size)
    
    # 只保留特征列
    feature_cols = [col for col in df_features.columns if col.startswith(('cnt', 'mean'))]
    df_features = df_features[feature_cols]
    
    return df_features
