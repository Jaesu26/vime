from typing import Any, Dict, List, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import check_random_state
from torch import Tensor

from .models import VIMESelfModule, VIMESemiModule
from .utils import mask_generator, pretext_generator


class VIMESelf(pl.LightningModule):
    def __init__(
        self,
        in_features_list: List[int],
        out_features_list: List[int],
        learning_rate: float = 5e-3,
        p_masking: float = 0.3,
        alpha: float = 2.0,
        log_interval: int = 10,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = VIMESelfModule(in_features_list, out_features_list)
        self.random_state = check_random_state(seed)
        self.feature_criterion = nn.MSELoss()
        self.mask_criterion = nn.BCEWithLogitsLoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def on_before_batch_transfer(self, batch: np.ndarray, dataloader_idx: int) -> Dict[str, Tensor]:
        X = batch
        mask = mask_generator(X, self.hparams.p_masking, self.random_state)
        X_tilde, mask = pretext_generator(X, mask, self.random_state)
        X, X_tilde, mask = torch.FloatTensor(X), torch.FloatTensor(X_tilde), torch.FloatTensor(mask)
        batch = {"X": X, "X_tilde": X_tilde, "mask": mask}
        return batch

    def _shared_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        X, X_tilde, mask = batch["X"], batch["X_tilde"], batch["mask"]
        X_hat, mask_hat = self(X_tilde)
        mask_vector_estimation_loss = self.mask_criterion(mask_hat, mask)
        reconstruction_loss = self.feature_criterion(X_hat, X)
        loss = mask_vector_estimation_loss + self.hparams.alpha * reconstruction_loss
        return {"loss": loss, "l_m": mask_vector_estimation_loss, "l_r": reconstruction_loss}

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        output = self._shared_step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.training_step_outputs]).mean()
        mean_loss_m = torch.stack([output["l_m"] for output in self.training_step_outputs]).mean()
        mean_loss_r = torch.stack([output["l_r"] for output in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        if self.current_epoch % self.hparams.log_interval == 0:
            print(
                f"Epoch {self.current_epoch} | Train Loss: {mean_loss:.4f}"
                f" | Train Loss_m: {mean_loss_m:.4f} | Train Loss_r: {mean_loss_r:.4f}",
                end=" " * 2,
            )

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        output = self._shared_step(batch, batch_idx)
        self.validation_step_outputs.append(output)

    def validation_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.validation_step_outputs]).mean()
        mean_loss_m = torch.stack([output["l_m"] for output in self.validation_step_outputs]).mean()
        mean_loss_r = torch.stack([output["l_r"] for output in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log_dict({"loss": mean_loss})
        if self.current_epoch % self.hparams.log_interval == 0:
            print(f"Val Loss: {mean_loss:.4f} | Val Loss_m: {mean_loss_m:.4f} | Val Loss_r: {mean_loss_r:.4f}")

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]
