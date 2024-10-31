import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import config
from mylstm import MyLSTM

if __name__ == '__main__':
    pl.seed_everything(22)
    logger = TensorBoardLogger(save_dir="metrics", name="lstm")
    model = MyLSTM(basic=True)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.N_EPOCH,
        log_every_n_steps=1,
        enable_model_summary=True,
        enable_progress_bar=True
    )
    trainer.fit(model)
    trainer.test(model)
