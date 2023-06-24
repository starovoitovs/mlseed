import os

import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader, default_collate
import torch
from mlseed.utils import DEVICE

# automatically load all tensor on the device
# https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t
collate_fn = lambda x: tuple(y.to(DEVICE) for y in default_collate(x))


class SP500Dataset(LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/close.csv')
        self.df = pd.read_csv(filename)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.set_index('date')[['CTXS', 'VTR']]

        self.train_dataset = TensorDataset(torch.tensor(self.df.to_numpy().astype(np.float32), device=DEVICE))
        self.val_dataset = TensorDataset(torch.tensor(self.df.to_numpy().astype(np.float32), device=DEVICE))
        self.test_dataset = TensorDataset(torch.tensor(self.df.to_numpy().astype(np.float32), device=DEVICE))

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, collate_fn=collate_fn, drop_last=True, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, collate_fn=collate_fn, batch_size=100000)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, collate_fn=collate_fn, batch_size=100000)
