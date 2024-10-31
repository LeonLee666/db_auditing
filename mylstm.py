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
train_set, test_set = PrepareData()


class MyLSTM(pl.LightningModule):
    def __init__(self, basic=True):
        super(MyLSTM, self).__init__()
        self.basic = basic
        if not self.basic:
            # 卷积层
            self.conv1 = nn.Conv1d(in_channels=config.INPUT_SIZE, out_channels=config.CONV_OUT_CHANNELS,
                                   kernel_size=config.KERNEL_SIZE)
            self.pool = nn.MaxPool1d(kernel_size=config.POOL_SIZE)  # 池化层
            config.INPUT_SIZE = config.CONV_OUT_CHANNELS

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
        if not self.basic:
            x = x.permute(0, 2, 1)  # 转换为 [batch size, feature size, seq length] 以适应 Conv1D
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)  # 池化层
            x = x.permute(0, 2, 1)  # 转回为 [batch size, seq length, feature size] 以适应 LSTM
        self.lstm.flatten_parameters()
        h_0 = torch.zeros(config.LSTM_LAYER, x.size(0), config.HIDDEN_SIZE).to(x.device)
        c_0 = torch.zeros(config.LSTM_LAYER, x.size(0), config.HIDDEN_SIZE).to(x.device)
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

    def training_step(self, batch, batch_idx):
        features, labels = batch
        loss, outputs = self(features, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_acc = acc(predictions, labels)
        step_f1 = f1_score(predictions, labels)
        self.log("fit loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("fit accuracy", step_acc, on_step=True, prog_bar=True, logger=True)
        self.log("fit f1 score", step_f1, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_acc, "f1": step_f1}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        loss, outputs = self(features, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_acc = acc(predictions, labels)
        step_f1 = f1_score(predictions, labels)
        self.log("validation loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("validation accuracy", step_acc, on_step=True, prog_bar=True, logger=True)
        self.log("validation f1 score", step_f1, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_acc, "f1": step_f1}

    def test_step(self, batch, batch_idx):
        features, labels = batch
        loss, outputs = self(features, labels)
        predictions = torch.argmax(outputs, dim=1)
        step_acc = acc(predictions, labels)
        step_f1 = f1_score(predictions, labels)
        self.log("test loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test accuracy", step_acc, on_step=True, prog_bar=True, logger=True)
        self.log("test f1 score", step_f1, on_step=True, prog_bar=True, logger=True)
        return {"loss": loss, "accuracy": step_acc, "f1": step_f1}

    def train_dataloader(self):
        return DataLoader(
            dataset=DataWrapper(train_set),
            batch_size=config.TRAINING_BATCH_SIZE,
            shuffle=True,
            persistent_workers=True,
            num_workers=cpu_count()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=DataWrapper(test_set),
            batch_size=config.TRAINING_BATCH_SIZE,
            shuffle=False,
            persistent_workers=True,
            num_workers=cpu_count()
        )

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=config.LEARNING_RATE)
