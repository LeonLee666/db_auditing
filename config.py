FEATURE_ALGORITHM = 'centroid'
POSITIVE_SIZE = 100000
NEGATIVE_SIZE = 100000

JOBS = 32
# 单个时序样本的长度，也就是一个时序样本包含多少条sql
SEQ_LENGTH = 10000

# 时序metric的打点种类，对应sql中的变量个数
WINDOW_SIZES = [256]
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

VANILLA_MODE = False  # 默认为False，使用特征工程后的特征