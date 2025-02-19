#!/bin/bash

DEFAULT_SEQ_LENGTH=10000
DEFAULT_WINDOW_SIZES="256,1024"
DEFAULT_FEATURE_ALGORITHM="grid"

# 获取命令行参数
seq_length=${1:-$DEFAULT_SEQ_LENGTH}
window_sizes=${2:-$DEFAULT_WINDOW_SIZES}
feature_algorithm=${3:-$DEFAULT_FEATURE_ALGORITHM}

# 创建配置文件
cat > config.py << EOF
FEATURE_ALGORITHM = '${feature_algorithm}'
POSITIVE_SIZE = 150000
NEGATIVE_SIZE = 150000

JOBS = 32
# 单个时序样本的长度，也就是一个时序样本包含多少条sql
SEQ_LENGTH = ${seq_length}

# 时序metric的打点种类，对应sql中的变量个数
WINDOW_SIZES = [${window_sizes}]
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
EOF

echo "配置文件已生成: config.py" 