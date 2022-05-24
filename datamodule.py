from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import PersonalityDataset
from params import LocationConfig


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        train_dir: Path = None,
        val_dir: Path = None
    ):
        """
        :param batch_size: batch size
        :param train_dir: directory with pickle files for train set
        :param val_path: directory with pickle files for validation set
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.val_dir = val_dir

    def setup(self, val_only=False, stage=None):
        if not val_only:
            self.train_set = PersonalityDataset(self.train_dir)
        self.val_set = PersonalityDataset(self.val_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
        )