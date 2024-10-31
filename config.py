SPIDER_FILE = 'spider_part.csv'
NORMAL_FILE = 'normal_part.csv'
# 数据集参数
# 爬虫要本数量
POSITIVE_SIZE = 800000
# 正常样本数量
NEGATIVE_SIZE = 800000
# 单个时序样本的长度，也就是一个时序样本包含多少条sql
SEQ_LENGTH = 500

# 模型参数
CONV_OUT_CHANNELS = 16
KERNEL_SIZE = 3
POOL_SIZE = 2
# 时序metric的打点种类，对应sql中的变量个数
INPUT_SIZE = 1
# 隐藏层神经元个数
HIDDEN_SIZE = 32
# 隐藏层层数
LSTM_LAYER = 5
# 输出神经元个数
OUTPUT_CLASSES = 2
DROPOUT = 0.0

# 训练参数
TRAINING_BATCH_SIZE = 32
N_EPOCH = 10
LEARNING_RATE = 0.001
