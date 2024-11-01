import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import config
from mylstm import MyLSTM

if __name__ == '__main__':
    pl.seed_everything(22)
    logger = TensorBoardLogger(save_dir="metrics", name="lstm")
    model = MyLSTM(batch_size=config.TRAINING_BATCH_SIZE, learning_rate=config.LEARNING_RATE, basic=True)
    trainer = pl.Trainer(
        logger=logger,

        devices=1,
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
        devices=1,
        accelerator='gpu',
        enable_progress_bar=True
    )
    tester.test(model)
