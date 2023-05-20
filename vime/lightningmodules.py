from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchmetrics import Accuracy

from .losses import CELoss, ConsistencyLoss
from .models import MLP, VIMESelfNetwork, VIMESemiNetwork
from .utils import check_rng, mask_generator, pretext_generator


class VIMESelf(pl.LightningModule):
    """VIME for self-supervised learning.

    Args:
        input_dim: Number of features.
        hidden_dims: Number of features each hidden state.
        cat_indices: Categorical features indices.
            If the list is empty, no embeddings will be done.
        cat_dims: Number of unique values for each categorical feature.
        cat_embedding_dim: Embedding dimension for each categorical feature.
            If int, the same embedding dimension will be used for all categorical features.
        lr: Learning rate for the optimizer.
        p_masking: Probability of masking a feature.
        alpha: Hyperparameter to control weights of mask vector estimation loss and reconstruction loss.
        log_interval: Logging frequency.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        cat_indices: Optional[List[int]] = None,
        cat_dims: Optional[List[int]] = None,
        cat_embedding_dim: Union[int, List[int]] = 2,
        lr: float = 1e-2,
        p_masking: float = 0.3,
        alpha: float = 2.0,
        log_interval: int = 10,
        seed: int = 26,
    ) -> None:
        super().__init__()
        pl.seed_everything(seed)
        self.save_hyperparameters()
        self.net = VIMESelfNetwork(input_dim, hidden_dims, cat_indices, cat_dims, cat_embedding_dim)
        self.cont_indices = self.net.encoder.embedder.cont_indices
        self.cat_indices = self.net.encoder.embedder.cat_indices
        self.cat_dims = self.net.encoder.embedder.cat_dims
        self.cat_embedding_dims = self.net.encoder.embedder.cat_embedding_dims
        self.total_cat_dim = self.net.encoder.embedder.total_cat_dim
        self.weight = self.net.encoder.embedder.num_cat_features / input_dim
        cat_dims = [0] + self.cat_dims
        self.start_indices = np.cumsum(cat_dims)[:-1]
        self.end_indices = np.cumsum(cat_dims)[1:]
        self.rng = check_rng(seed)
        self.continuous_feature_criterion = nn.MSELoss()
        self.categorical_feature_criterion = CELoss()
        self.mask_criterion = nn.BCELoss()
        self.training_step_outputs: List[Dict[str, Tensor]] = []
        self.validation_step_outputs: List[Dict[str, Tensor]] = []
        self.mean_loss = None
        self.mean_loss_m = None
        self.mean_loss_r = None

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def on_before_batch_transfer(self, batch: Tensor, dataloader_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        if self.trainer.training:
            x = batch
            mask = mask_generator(self.hparams.p_masking, x.shape, self.rng)
            x_tilde, mask = pretext_generator(x, mask, self.rng)
            batch = x, x_tilde, mask
        elif self.trainer.validating:
            # Do not transform validation data for non-stochastic validation
            x = batch
            mask = torch.zeros_like(x)
            x_tilde = x
            batch = x, x_tilde, mask
        return batch

    def _shared_fit_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, x_tilde, mask = batch
        x_hat, mask_hat = self(x_tilde)
        loss, l_m, l_r = self.compute_loss(mask_hat, mask, x_hat, x)
        return {"loss": loss, "l_m": l_m, "l_r": l_r}

    def compute_loss(self, mask_hat: Tensor, mask: Tensor, x_hat: Tensor, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mask_vector_estimation_loss = self._compute_mask_loss(mask_hat, mask)
        reconstruction_loss = self._compute_reconstruction_loss(x_hat, x)
        loss = mask_vector_estimation_loss + self.hparams.alpha * reconstruction_loss
        return loss, mask_vector_estimation_loss, reconstruction_loss

    def _compute_mask_loss(self, mask_hat: Tensor, mask: Tensor) -> Tensor:
        return self.mask_criterion(mask_hat, mask)

    def _compute_reconstruction_loss(self, x_hat: Tensor, x: Tensor) -> Tensor:
        x_continuous = x[:, self.cont_indices]
        x_hat_continuous = x_hat[:, self.total_cat_dim :]
        reconstruction_loss_cont = self.continuous_feature_criterion(x_hat_continuous, x_continuous)
        reconstruction_loss_cat = 0.0
        for cat_index, start_index, end_index in zip(self.cat_indices, self.start_indices, self.end_indices):
            x_categorical = x[:, cat_index].long()
            x_hat_categorical = x_hat[:, start_index:end_index]
            loss = self.categorical_feature_criterion(x_hat_categorical, x_categorical)
            reconstruction_loss_cat += loss / self.num_cat_features
        reconstruction_loss = (1 - self.weight) * reconstruction_loss_cont + self.weight * reconstruction_loss_cat
        return reconstruction_loss

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        output = self._shared_fit_step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def _on_shared_fit_epoch_end(self) -> None:
        outputs = self.training_step_outputs if self.trainer.training else self.validation_step_outputs
        self.mean_loss = torch.stack([output["loss"] for output in outputs]).mean()
        self.mean_loss_m = torch.stack([output["l_m"] for output in outputs]).mean()
        self.mean_loss_r = torch.stack([output["l_r"] for output in outputs]).mean()
        outputs.clear()

    def on_train_epoch_end(self) -> None:
        self._on_shared_fit_epoch_end()
        if self.should_log:
            print(
                f"Epoch {self.current_epoch + 1} | Train Loss: {self.mean_loss:.4f}"
                f" | Train Loss_m: {self.mean_loss_m:.4f} | Train Loss_r: {self.mean_loss_r:.4f}",
                end=" " * 2,
            )

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        output = self._shared_fit_step(batch, batch_idx)
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self) -> None:
        self._on_shared_fit_epoch_end()
        self.log("val_loss", self.mean_loss)
        if self.should_log:
            print(
                f"Val Loss: {self.mean_loss:.4f} | Val Loss_m: {self.mean_loss_m:.4f}"
                f" | Val Loss_r: {self.mean_loss_r:.4f}"
            )

    @property
    def should_log(self) -> bool:
        return (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.hparams.log_interval == 0
            or self.current_epoch + 1 == self.trainer.max_epochs
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]

    @property
    def encoder(self) -> nn.Module:
        return self.net.encoder


class VIMESemi(pl.LightningModule):
    """VIME for semi-supervised learning.

    Args:
        pretrained_encoder: Pretrained encoder.
        hidden_dims: Number of features each hidden state.
        num_classes: Number of classes.
        supervised_criterion: Supervised loss function (i.g. torch.nn.CrossEntropyLoss()).
        lr: Learning rate for the optimizer.
        p_masking: Probability of masking a feature.
        K: Number of unlabeled augmentations.
        beta: Hyperparameter to control weights of supervised loss and consistency loss.
        log_interval: Logging frequency.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        pretrained_encoder: nn.Module,
        hidden_dims: List[int],
        num_classes: int,
        supervised_criterion: Callable[[Tensor, Tensor], Tensor],
        lr: float = 5e-3,
        p_masking: float = 0.3,
        K: int = 3,
        beta: float = 1.0,
        log_interval: int = 10,
        seed: int = 26,
    ) -> None:
        super().__init__()
        pl.seed_everything(seed)
        self.save_hyperparameters(ignore="pretrained_encoder")
        self.net = VIMESemiNetwork(pretrained_encoder, hidden_dims, num_classes)
        self.rng = check_rng(seed)
        self.supervised_criterion = supervised_criterion
        self.consistency_criterion = ConsistencyLoss()
        self.training_step_outputs: List[Dict[str, Tensor]] = []
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def on_fit_start(self) -> None:
        self.net.freeze_encoder()

    def on_train_epoch_start(self) -> None:
        self.net.pretrained_encoder.eval()

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.trainer.training:
            x_unlabeled = batch["unlabeled"]
            x_augmented = []
            for _ in range(self.hparams.K):
                mask = mask_generator(self.hparams.p_masking, x_unlabeled.shape, self.rng)
                x_tilde, _ = pretext_generator(x_unlabeled, mask, self.rng)
                x_augmented.append(x_tilde)
            batch["unlabeled"] = torch.stack(x_augmented)  # Shape: (K, B, C)
        return batch

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x_labeled, y = batch["labeled"]
        x_augmented = batch["unlabeled"]
        y_hat_from_original = self(x_labeled)
        y_hat_from_corruption = self(x_augmented)
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
        if self.should_log:
            print(
                f"Epoch {self.current_epoch + 1} | Train Loss: {mean_loss:.4f}"
                f" | Train Loss_s: {mean_loss_s:.4f} | Train Loss_u: {mean_loss_u:.4f}",
                end=" " * 2,
            )

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        supervised_loss = self.supervised_criterion(y_hat, y)
        output = {"loss_s": supervised_loss}
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self) -> None:
        mean_loss_s = torch.stack([output["loss_s"] for output in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log("val_loss", mean_loss_s)
        if self.should_log:
            print(f"Val Loss_s: {mean_loss_s:.4f}")

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch
        y_hat = self(x)
        return y_hat

    @property
    def should_log(self) -> bool:
        return (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.hparams.log_interval == 0
            or self.current_epoch + 1 == self.trainer.max_epochs
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.net.predictor.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]


class MLPClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        cat_indices: Optional[List[int]] = None,
        cat_dims: Optional[List[int]] = None,
        cat_embedding_dim: Union[int, List[int]] = 2,
        lr: float = 1e-3,
        log_interval: int = 10,
        seed: int = 26,
    ) -> None:
        super().__init__()
        pl.seed_everything(seed)
        self.save_hyperparameters()
        self.net = MLP(input_dim, hidden_dims, num_classes, cat_indices, cat_dims, cat_embedding_dim)
        task = "binary" if num_classes == 1 else "multiclass"
        if task == "binary":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.macro_accuracy = Accuracy(task=task, num_classes=num_classes, average="macro")
        self.training_step_outputs: List[Dict[str, Tensor]] = []
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def _shared_fit_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        output = self._shared_fit_step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        if self.should_log:
            print(f"Epoch {self.current_epoch + 1} | Train Loss: {mean_loss:.4f}", end=" " * 2)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        output = self._shared_fit_step(batch, batch_idx)
        self.validation_step_outputs.append(output)
        self.macro_accuracy.update(output["y_hat"], output["y"])

    def on_validation_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.validation_step_outputs]).mean()
        macro_acc = self.macro_accuracy.compute()
        self.log_dict({"val_loss": mean_loss, "val_macro_acc": macro_acc})
        self.validation_step_outputs.clear()
        self.macro_accuracy.reset()
        if self.should_log:
            print(f"Val Loss: {mean_loss:.4f} | Val Macro Acc: {macro_acc:.4f}")

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch
        y_hat = self(x)
        return y_hat

    @property
    def should_log(self) -> bool:
        return (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.hparams.log_interval == 0
            or self.current_epoch + 1 == self.trainer.max_epochs
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]
