import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import config
from mylstm import MyLSTM

def get_trainer_config(use_cuda, is_test=False):
    base_config = {
        'logger': TensorBoardLogger(save_dir="metrics", name="audit"),
        'enable_progress_bar': True,
    }
    
    if use_cuda:
        base_config.update({
            'devices': [1],
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
    parser.add_argument('--fe', action='store_true', help='is need re-calc features')
    parser.add_argument('--cuda', action='store_true', help='running device')
    args = parser.parse_args()
    
    config.NEED_CALC_FEATURES = args.fe
    pl.seed_everything(22)
    
    model = MyLSTM(
        batch_size=config.TRAINING_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        basic=True
    )
    
    # Training
    trainer = pl.Trainer(**get_trainer_config(args.cuda))
    trainer.fit(model)
    
    # Testing
    tester = pl.Trainer(**get_trainer_config(args.cuda, is_test=True))
    tester.test(model)

if __name__ == '__main__':
    main()
