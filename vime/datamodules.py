import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .datasets import LabeledDataset, UnlabeledDataset


class VIMESelfDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X: np.ndarray,
        train_size: float = 0.9,
        batch_size: int = 512,
        seed: int = 26,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.X = X
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str) -> None:
        X_train, X_val = train_test_split(self.X, train_size=self.hparams.train_size, random_state=self.hparams.seed)
        self.train_dataset = UnlabeledDataset(X_train)
        self.val_dataset = UnlabeledDataset(X_val)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False)


class VIMESemiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: np.ndarray,
        y: np.ndarray,
        X_predict: np.ndarray,
        train_size: float = 0.9,
        labeled_batch_size: int = 256,
        unlabeled_batch_size: int = 512,
        seed: int = 26,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.X_unlabeled = X_unlabeled
        self.X_labeled = X_labeled
        self.y = y
        self.X_predict = X_predict
        self.train_unlabeled_dataset = None
        self.train_labeled_dataset = None
        self.val_labeled_dataset = None
        self.predict_dataset = None

    def setup(self, stage: str) -> None:
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_labeled, self.y, train_size=self.hparams.train_size, random_state=self.hparams.seed, stratify=self.y
        )
        self.train_unlabeled_dataset = UnlabeledDataset(self.X_unlabeled)
        self.train_labeled_dataset = LabeledDataset(X_train, y_train)
        self.val_labeled_dataset = LabeledDataset(X_val, y_val)
        self.predict_dataset = UnlabeledDataset(self.X_predict)

    def train_dataloader(self) -> CombinedLoader:
        labeled_dataloader = DataLoader(
            self.train_labeled_dataset, batch_size=self.hparams.labeled_batch_size, shuffle=True
        )
        unlabeled_dataloader = DataLoader(
            self.train_unlabeled_dataset, batch_size=self.hparams.unlabeled_batch_size, shuffle=True
        )
        dataloaders = {"labeled": labeled_dataloader, "unlabeled": unlabeled_dataloader}
        if len(labeled_dataloader) >= len(unlabeled_dataloader):
            combined_loader = CombinedLoader(dataloaders, mode="max_size_cycle")
        else:
            combined_loader = CombinedLoader(dataloaders, mode="min_size")
        return combined_loader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_labeled_dataset, batch_size=self.hparams.labeled_batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset, batch_size=self.hparams.unlabeled_batch_size, shuffle=False)
