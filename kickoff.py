import sys
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.tuner import Tuner
import config
from mylstm import MyLSTM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="auditlog ai agent.")
    parser.add_argument('--fe', action='store_true', help='is need re-calc features')
    parser.add_argument('--cuda', action='store_true', help='running device')
    args = parser.parse_args()
    config.NEED_CALC_FEATURES = args.fe
    pl.seed_everything(22)
    logger = TensorBoardLogger(save_dir="metrics", name="audit")
    # logger = CSVLogger("csv_log", name="lstm")
    model = MyLSTM(batch_size=config.TRAINING_BATCH_SIZE, learning_rate=config.LEARNING_RATE, basic=True)
    if args.cuda:
        trainer = pl.Trainer(
            logger=logger,
            devices=[1],
            accelerator='gpu',
            max_epochs=config.N_EPOCH,
            log_every_n_steps=1,
            enable_model_summary=True,
            enable_progress_bar=True
        )
        # tuner = Tuner(trainer)
        # tuner.lr_find(model=model)
        trainer.fit(model)
        tester = pl.Trainer(
            logger=logger,
            devices=[1],
            accelerator='gpu',
            enable_progress_bar=True
        )
        tester.test(model)
    else:
        trainer = pl.Trainer(
            logger=logger,
            accelerator='cpu',
            max_epochs=config.N_EPOCH,
            log_every_n_steps=1,
            enable_model_summary=True,
            enable_progress_bar=True
        )
        # tuner = Tuner(trainer)
        # tuner.lr_find(model=model)
        trainer.fit(model)
        tester = pl.Trainer(
            logger=logger,
            accelerator='cpu',
            enable_progress_bar=True
        )
        tester.test(model)
