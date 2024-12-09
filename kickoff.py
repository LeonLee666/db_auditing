import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import config
from mylstm import MyLSTM

def get_trainer_config(use_gpu: bool):
    """获取trainer的基础配置"""
    base_config = {
        "logger": TensorBoardLogger(save_dir="metrics", name="audit"),
        "max_epochs": config.N_EPOCH,
        "log_every_n_steps": 1,
        "enable_model_summary": True,
        "enable_progress_bar": True
    }
    
    if use_gpu:
        base_config.update({
            "devices": [1],
            "accelerator": "gpu"
        })
    else:
        base_config.update({
            "accelerator": "cpu"
        })
    return base_config

def train_and_test(model: pl.LightningModule, use_gpu: bool):
    """训练和测试模型"""
    trainer_config = get_trainer_config(use_gpu)
    
    # 训练
    trainer = pl.Trainer(**trainer_config)
    try:
        # 取消注释以下行来启用学习率查找器
        # tuner = Tuner(trainer)
        # tuner.lr_find(model=model)
        trainer.fit(model)
    except Exception as e:
        print(f"训练过程发生错误: {str(e)}")
        sys.exit(1)
    
    # 测试
    test_config = {k: v for k, v in trainer_config.items() if k not in ['max_epochs', 'log_every_n_steps']}
    tester = pl.Trainer(**test_config)
    try:
        tester.test(model)
    except Exception as e:
        print(f"测试过程发生错误: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="auditlog ai agent.")
    parser.add_argument('--cuda', action='store_true', help='是否使用GPU')
    parser.add_argument('--fe', action='store_true', help='是否重新生成特征')
    args = parser.parse_args()

    # 设置特征重生成标志
    config.REGENERATE_FEATURES = args.fe
    
    pl.seed_everything(22)
    
    model = MyLSTM(
        batch_size=config.TRAINING_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        basic=True
    )
    
    train_and_test(model, args.cuda)
