import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from joblib.externals.loky import cpu_count
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score

import config
from dataset import PrepareData, DataWrapper

criterion = nn.CrossEntropyLoss()
dropout = nn.Dropout(config.DROPOUT)
acc = Accuracy(task='binary')
f1_score = F1Score(num_classes=config.OUTPUT_CLASSES, task='binary')


class MyLSTM(pl.LightningModule):
    def __init__(self, batch_size, learning_rate):
        super(MyLSTM, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_set, self.val_set, self.test_set = PrepareData()        

        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.LSTM_LAYER,
            batch_first=True,
            dropout=config.DROPOUT
        )
        self.dense1 = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.dense2 = nn.Linear(config.HIDDEN_SIZE, config.HIDDEN_SIZE)
        self.dense3 = nn.Linear(config.HIDDEN_SIZE, config.OUTPUT_CLASSES)

    def forward(self, x, labels=None):
        # x.shape = [batch size, seq length X feature size]
        self.lstm.flatten_parameters()
        h_0 = torch.zeros(config.LSTM_LAYER, x.size(0), config.HIDDEN_SIZE).to(x.device).requires_grad_()
        c_0 = torch.zeros(config.LSTM_LAYER, x.size(0), config.HIDDEN_SIZE).to(x.device).requires_grad_()
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # get the last time step output, the shape is [batchsize, hidden size]
        out = self.dense1(out)
        out = F.relu(out)
        out = dropout(out)
        out = self.dense2(out)
        out = F.relu(out)
        out = dropout(out)
        out = self.dense3(out)
        loss = 0
        if labels is not None:
            loss = criterion(out, labels)
        return loss, out

    def __common__calc__(self, batch, batch_idx):
        features, labels = batch
        loss, outputs = self(features, labels)
        predictions = torch.argmax(outputs, dim=1)
        device = features.device
        acc_device = acc.to(device)
        f1_score2_device = f1_score.to(device)
        step_acc = acc_device(predictions, labels)
        step_f1 = f1_score2_device(predictions, labels)
        return loss, step_acc, step_f1

    def training_step(self, batch, batch_idx):
        loss, step_acc, step_f1 = self.__common__calc__(batch, batch_idx)
        self.log("fit loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("fit accuracy", step_acc, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("fit f1 score", step_f1, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        if loss < 0.002:
            self.trainer.should_stop = True  # 手动停止训练
        return {"loss": loss, "accuracy": step_acc, "f1": step_f1}

    def validation_step(self, batch, batch_idx):
        loss, step_acc, step_f1 = self.__common__calc__(batch, batch_idx)
        self.log("validation loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation accuracy", step_acc, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("validation f1 score", step_f1, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "accuracy": step_acc, "f1": step_f1}

    def test_step(self, batch, batch_idx):
        loss, step_acc, step_f1 = self.__common__calc__(batch, batch_idx)
        self.log("test loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test accuracy", step_acc, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("test f1 score", step_f1, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss, "accuracy": step_acc, "f1": step_f1}

    def train_dataloader(self):
        return DataLoader(
            dataset=DataWrapper(self.train_set),
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=cpu_count()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=DataWrapper(self.val_set),
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=DataWrapper(self.test_set),
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=cpu_count()
        )

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
