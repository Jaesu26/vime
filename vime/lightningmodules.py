from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import check_random_state
from torch import Tensor
from torchmetrics import Accuracy

from .loss import ConsistencyLoss
from .models import MLP, VIMESelfNetwork, VIMESemiNetwork
from .utils import mask_generator, pretext_generator


class VIMESelf(pl.LightningModule):
    """The VIME for self-supervised learning.

    Args:
        input_dim: The number of features.
        hidden_dims: The number of features each hidden layer.
        cat_indices: The positional index for each categorical features.
        cat_dims: The number of modalities for each categorical features.
            If the list is empty, no embeddings will be done.
        cat_embedding_dim: The embedding dimension for each categorical features.
            If int, the same embedding dimension will be used for all categorical features.
        learning_rate: The learning rate for the optimizer.
        p_masking: The probability of masking a feature.
        alpha: The hyperparameter to control the weights of mask vector estimation loss and reconstruction loss.
        log_interval: The logging frequency.
        seed: The random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        cat_indices: Sequence[int] = (),
        cat_dims: Sequence[int] = (),
        cat_embedding_dim: Union[Sequence[int], int] = 2,
        learning_rate: float = 1e-2,
        p_masking: float = 0.3,
        alpha: float = 2.0,
        log_interval: int = 10,
        seed: int = 26,
    ) -> None:
        super().__init__()
        pl.seed_everything(seed)
        self.save_hyperparameters()
        self.vime_self = VIMESelfNetwork(input_dim, hidden_dims, cat_indices, cat_dims, cat_embedding_dim)
        self.cont_indices = self.vime_self.encoder.embeddings.cont_indices
        self.cat_indices = self.vime_self.encoder.embeddings.cat_indices
        self.cat_dims = self.vime_self.encoder.embeddings.cat_dims
        self.cat_embedding_dims = self.vime_self.encoder.embeddings.cat_embedding_dims
        self.total_cat_dim = self.vime_self.encoder.embeddings.cat_dims
        self.random_state = check_random_state(seed)
        self.continuous_feature_criterion = nn.MSELoss()
        self.categorical_feature_criterion = nn.CrossEntropyLoss()
        self.mask_criterion = nn.BCEWithLogitsLoss()
        self.training_step_outputs: List[Dict[str, Tensor]] = []
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.vime_self(x)

    def on_before_batch_transfer(self, batch: Tensor, dataloader_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        X = batch
        mask = mask_generator(self.hparams.p_masking, X.shape, self.random_state)
        X_tilde, mask = pretext_generator(X, mask, self.random_state)
        batch = X, X_tilde, mask
        return batch

    def _shared_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        X, X_tilde, mask = batch
        X_hat, mask_hat = self(X_tilde)
        mask_vector_estimation_loss = self.mask_criterion(mask_hat, mask)
        X_continuous = X[:, self.cont_indices]
        X_hat_continuous = X_hat[:-self.total_cat_dim]
        X_hat_categorical = X_hat[-self.total_cat_dim:]
        Xs_categorical_hat = [
            X_hat_categorical[start_index:end_index]
            for start_index, end_index in zip(self.cat_dims[:-1], self.cat_dims[1:])
        ]
        reconstruction_loss_cat = 0.0
        for cat_index, X_hat_cat in zip(self.cat_indices, Xs_categorical_hat):
            loss = self.categorical_feature_criterion(X[:, cat_index], X_hat_cat)
            reconstruction_loss_cat += loss
        reconstruction_loss_cont = self.continuous_feature_criterion(X_hat_continuous, X_continuous)
        reconstruction_loss = reconstruction_loss_cont + reconstruction_loss_cat
        loss = mask_vector_estimation_loss + self.hparams.alpha * reconstruction_loss
        return {"loss": loss, "l_m": mask_vector_estimation_loss, "l_r": reconstruction_loss}

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        output = self._shared_step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.training_step_outputs]).mean()
        mean_loss_m = torch.stack([output["l_m"] for output in self.training_step_outputs]).mean()
        mean_loss_r = torch.stack([output["l_r"] for output in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        if self.should_log:
            print(
                f"Epoch {self.current_epoch + 1} | Train Loss: {mean_loss:.4f}"
                f" | Train Loss_m: {mean_loss_m:.4f} | Train Loss_r: {mean_loss_r:.4f}",
                end=" " * 2,
            )

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> None:
        output = self._shared_step(batch, batch_idx)
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.validation_step_outputs]).mean()
        mean_loss_m = torch.stack([output["l_m"] for output in self.validation_step_outputs]).mean()
        mean_loss_r = torch.stack([output["l_r"] for output in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log_dict({"val_loss": mean_loss})
        if self.should_log:
            print(f"Val Loss: {mean_loss:.4f} | Val Loss_m: {mean_loss_m:.4f} | Val Loss_r: {mean_loss_r:.4f}")

    @property
    def should_log(self):
        return (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.hparams.log_interval == 0
            or self.current_epoch + 1 == self.trainer.max_epochs
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]

    @property
    def encoder(self) -> nn.Module:
        return self.vime_self.encoder


class VIMESemi(pl.LightningModule):
    """The VIME for semi-supervised learning.

    Args:
        pretrained_encoder: The pretrained encoder.
        input_dim: The number of features.
        hidden_dims: The number of features each hidden layer.
        num_classes: The number of classes.
        supervised_criterion: The supervised loss function (i.g. torch.nn.CrossEntropyLoss()).
        learning_rate: The learning rate for the optimizer.
        p_masking: The probability of masking a feature.
        K: The number of augmented samples.
        beta: The hyperparameter to control the weights of supervised loss and consistency loss.
        log_interval: The logging frequency.
        seed: The random seed for reproducibility.
    """

    def __init__(
        self,
        pretrained_encoder: nn.Module,
        input_dim: int,
        hidden_dims: List[int],
        num_classes: int,
        supervised_criterion: Callable[[Tensor, Tensor], Tensor],
        learning_rate: float = 5e-3,
        p_masking: float = 0.3,
        K: int = 3,
        beta: float = 1.0,
        log_interval: int = 10,
        seed: int = 26,
    ) -> None:
        super().__init__()
        pl.seed_everything(seed)
        self.save_hyperparameters()
        self.vime_semi = VIMESemiNetwork(pretrained_encoder, input_dim, hidden_dims, num_classes)
        self.random_state = check_random_state(seed)
        self.supervised_criterion = supervised_criterion
        self.consistency_criterion = ConsistencyLoss()
        self.training_step_outputs: List[Dict[str, Tensor]] = []
        self.validation_step_outputs: List[Dict[str, Tensor]] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.vime_semi(x)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.trainer.training:
            X_unlabeled = batch["unlabeled"]
            X_augmented = []
            for _ in range(self.hparams.K):
                mask = mask_generator(self.hparams.p_masking, X_unlabeled.shape, self.random_state)
                X_tilde, _ = pretext_generator(X_unlabeled, mask, self.random_state)
                X_augmented.append(X_tilde)
            batch["unlabeled"] = torch.stack(X_augmented)  # Shape: (K, B, C)
        return batch

    def on_train_epoch_start(self) -> None:
        self.vime_semi.freeze_encoder()

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        X_labeled, y = batch["labeled"]
        X_augmented = batch["unlabeled"]
        y_hat_from_original = self(X_labeled)
        y_hat_from_corruption = self(X_augmented)
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
        X, y = batch
        y_hat = self(X)
        supervised_loss = self.supervised_criterion(y_hat, y)
        output = {"loss_s": supervised_loss}
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self) -> None:
        mean_loss_s = torch.stack([output["loss_s"] for output in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log_dict({"val_loss": mean_loss_s})
        if self.should_log:
            print(f"Val Loss_s: {mean_loss_s:.4f}")

    def predict_step(self, batch: Tensor, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        X = batch
        y_hat = self(X)
        return y_hat

    @property
    def should_log(self):
        return (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.hparams.log_interval == 0
            or self.current_epoch + 1 == self.trainer.max_epochs
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.vime_semi.predictor.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]


class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.mlp_classifier = MLP(input_dim, hidden_dim, num_classes)
        task = "binary" if num_classes == 1 else "multiclass"
        if task == "binary":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.macro_accuracy = Accuracy(task=task, num_classes=num_classes, average="macro")

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.mlp_classifier(x)
        return y_hat

    def _shared_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        return {"loss": loss, "y_hat": y_hat, "y": y}

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        output = self._shared_step(batch, batch_idx)
        self.training_step_outputs.append(output)
        return output

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.training_step_outputs]).mean()
        self.training_step_outputs.clear()
        if self.should_log:
            print(f"Epoch {self.current_epoch + 1} | Train Loss: {mean_loss:.4f}", end=" " * 2)

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        output = self._shared_step(batch, batch_idx)
        self.macro_accuracy.update(output["y_hat"], output["y"])
        self.validation_step_outputs.append(output)

    def on_validation_epoch_end(self) -> None:
        mean_loss = torch.stack([output["loss"] for output in self.validation_step_outputs]).mean()
        macro_acc = self.macro_accuracy.compute()
        self.validation_step_outputs.clear()
        self.log_dict({"val_loss": mean_loss, "val_macro_acc": macro_acc})
        self.macro_accuracy.reset()
        if self.should_log:
            print(f"Val Loss: {mean_loss:.4f} | Val Macro Acc: {macro_acc:.4f}")

    @property
    def should_log(self):
        return (
            self.current_epoch == 0
            or (self.current_epoch + 1) % self.hparams.log_interval == 0
            or self.current_epoch + 1 == self.trainer.max_epochs
        )

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]
