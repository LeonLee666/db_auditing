import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import config
from mymodel import MyModel

# 添加 EarlyStoppingCallback
class LossBasedEarlyStopping(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['loss']
        if loss <= 0.1:
            print(f"\nLoss {loss:.4f} is less than 0.1, early stopping")
            trainer.should_stop = True

def get_trainer_config(use_cuda, is_test=False):
    base_config = {
        'logger': TensorBoardLogger(save_dir="metrics", name="audit"),
        'enable_progress_bar': True,
        'callbacks': [LossBasedEarlyStopping()],  # 添加callback
    }
    
    if use_cuda:
        base_config.update({
            'devices': [0],
            'accelerator': 'gpu'
        })
    else:
        base_config['accelerator'] = 'cpu'
    
    if not is_test:
        base_config.update({
            'max_epochs': config.N_EPOCH,
            'log_every_n_steps': 1,
            'enable_model_summary': True,
        })
    
    return base_config

def main():
    parser = argparse.ArgumentParser(description="auditlog ai agent.")
    parser.add_argument('--positive', type=str, required=True, help='positive log file path')
    parser.add_argument('--negative', type=str, required=True, help='negative log file path')
    parser.add_argument('--fe', action='store_true', help='is need re-calc features')
    parser.add_argument('--cuda', action='store_true', help='running device')
    args = parser.parse_args()
    config.NEED_CALC_FEATURES = args.fe
    config.POSITIVE_FILE = args.positive
    config.NEGATIVE_FILE = args.negative
    if config.FEATURE_ALGORITHM == 'centroid':
        config.INPUT_SIZE = len(config.WINDOW_SIZES) * 2
    else:
        config.INPUT_SIZE = len(config.WINDOW_SIZES)
    
    pl.seed_everything(42)    
    model = MyModel(
        batch_size=config.TRAINING_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE
    )    
    # Training
    trainer = pl.Trainer(**get_trainer_config(args.cuda))
    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        print("\n键盘中断,正在退出训练...")
        print("继续执行测试阶段...")
    finally:
        # Testing
        tester = pl.Trainer(**get_trainer_config(args.cuda, is_test=True))
        tester.test(model)

if __name__ == '__main__':
    main()
