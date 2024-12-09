POSITIVE_FILE = 'spider-short.csv'
POSITIVE_SIZE = 4500

NEGATIVE_FILE = 'normal-short.csv'
NEGATIVE_SIZE = 4500

# 单个时序样本的长度，也就是一个时序样本包含多少条sql
SEQ_LENGTH = 20000

# 模型参数
CONV_OUT_CHANNELS = 16
KERNEL_SIZE = 3
POOL_SIZE = 2


# 时序metric的打点种类，对应sql中的变量个数
INPUT_SIZE = 3
# 隐藏层神经元个数
HIDDEN_SIZE = 32
# 隐藏层层数
LSTM_LAYER = 3
# 输出神经元个数
OUTPUT_CLASSES = 2
DROPOUT = 0.2

# 训练参数
TRAINING_BATCH_SIZE = 32
N_EPOCH = 5
LEARNING_RATE = 1e-4

# 特征生成控制
REGENERATE_FEATURES = False  # 设置为True时重新生成特征，False时从文件读取特征
