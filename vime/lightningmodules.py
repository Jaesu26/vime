from typing import Any, Dict, List, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import check_random_state
from torch import Tensor

from .models import Predictor, VIMESelfModule, VIMESemiModule
from .utils import mask_generator, pretext_generator


class VIMESelf(pl.LightningModule):
    def __init__(
        self,
        in_features_list: List[int],
        out_features_list: List[int],
        learning_rate: float = 1e-2,
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
        self.training_step_outputs: List[Tensor] = []
        self.validation_step_outputs: List[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

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
        self.log_dict({"val_loss": mean_loss})
        if self.current_epoch % self.hparams.log_interval == 0:
            print(f"Val Loss: {mean_loss:.4f} | Val Loss_m: {mean_loss_m:.4f} | Val Loss_r: {mean_loss_r:.4f}")

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]


def consistency_criterion(y_hat: Tensor) -> Tensor:
    loss = torch.var(y_hat, dim=0).mean()
    return loss


class VIMESemi(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        in_features_list: List[int],
        out_features_list: List[int],
        num_classes: int,
        learning_rate: float = 5e-3,
        p_masking: float = 0.3,
        K: int = 3,
        beta: float = 1.0,
        log_interval: int = 10,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = VIMESemiModule(encoder, Predictor(in_features_list, out_features_list, num_classes))
        self.random_state = check_random_state(seed)
        self.supervised_criterion = nn.BCEWithLogitsLoss()
        self.consistency_criterion = consistency_criterion
        self.training_step_outputs: List[Tensor] = []
        self.validation_step_outputs: List[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.trainer.training:
            X_labeled, y = batch["labeled"]
            batch["labeled"] = torch.FloatTensor(X_labeled), torch.FloatTensor(y)
            X_unlabeled = batch["unlabeled"]
            X_augmented = []
            for _ in self.hparams.K:
                mask = mask_generator(X_unlabeled, self.hparams.p_masking, self.random_state)
                X_tilde, _ = pretext_generator(X_unlabeled, mask, self.random_state)
                X_tilde = torch.FloatTensor(X_tilde)
                X_augmented.append(X_tilde)
            batch["unlabeled"] = torch.stack(X_augmented)
        elif self.trainer.validating:
            X, y = batch
            batch = torch.FloatTensor(X), torch.FloatTensor(y)
        elif self.trainer.predicting:
            X = batch
            batch = torch.FloatTensor(X)
        return batch

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        X_labeled, y = batch["labeled"]
        X_unlabeled = batch["unlabeled"]
        y_hat_from_original = self(X_labeled)
        y_hat_from_corruption = self(X_unlabeled)
        supervised_loss = self.supervised_criterion(y_hat_from_original, y)
        consistency_loss = self.consistency_criterion(y_hat_from_corruption)
        loss = supervised_loss + self.hparams.beta * consistency_loss
        output = {"loss": loss, "l_s": supervised_loss, "l_u": consistency_loss}
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.training_step_outputs]).mean()
        mean_loss_s = torch.stack([output["l_s"] for output in self.training_step_outputs]).mean()
        mean_loss_u = torch.stack([output["l_u"] for output in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        if self.current_epoch % self.hparams.log_interval == 0:
            print(
                f"Epoch {self.current_epoch} | Train Loss: {mean_loss:.4f}"
                f" | Train Loss_s: {mean_loss_s:.4f} | Train Loss_u: {mean_loss_u:.4f}",
                end=" " * 2,
            )

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        X, y = batch
        y_hat = self(X)
        supervised_loss = self.supervised_criterion(y_hat, y)
        output = {"loss_s": supervised_loss}
        self.validation_step_outputs.append(output)

    def validation_epoch_end(self) -> None:
        mean_loss_s = torch.stack([output["loss_s"] for output in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log_dict({"val_loss": mean_loss_s})
        if self.current_epoch % self.hparams.log_interval == 0:
            print(f"Val Loss_s: {mean_loss_s:.4f}")

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        X = batch
        y_hat = self(X)
        return y_hat

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]
