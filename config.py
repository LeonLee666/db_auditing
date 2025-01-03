
POSITIVE_FEATURES = 'positive.csv'
POSITIVE_SIZE = 450000

NEGATIVE_FEATURES = 'negative.csv'
NEGATIVE_SIZE = 450000


JOBS = 1
NEED_CALC_FEATURES = False
# 单个时序样本的长度，也就是一个时序样本包含多少条sql
SEQ_LENGTH = 30000

# 时序metric的打点种类，对应sql中的变量个数
WINDOW_SIZES = [256, 1024, 4096, 16384]
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