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
    parser.add_argument('--algorithm', type=str, default='bin_count',
                       choices=['neighbor_count', 'bin_count'],
                       help='feature extraction algorithm (neighbor_count or bin_count)')
    args = parser.parse_args()
    
    config.NEED_CALC_FEATURES = args.fe
    config.POSITIVE_FILE = args.positive
    config.NEGATIVE_FILE = args.negative
    config.INPUT_SIZE = len(config.WINDOW_SIZES)
    config.FEATURE_ALGORITHM = args.algorithm
    pl.seed_everything(22)
    
    model = MyModel(
        batch_size=config.TRAINING_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE
    )
    
    # Training
    trainer = pl.Trainer(**get_trainer_config(args.cuda))
    trainer.fit(model)
    
    # Testing
    tester = pl.Trainer(**get_trainer_config(args.cuda, is_test=True))
    tester.test(model)

if __name__ == '__main__':
    main()
