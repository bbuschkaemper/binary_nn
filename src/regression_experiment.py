from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import SupportsFloat, cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import nn

from output_paths import checkpoint_root
from regression_data import RegressionDataConfig, RegressionDataModule


type Batch = tuple[torch.Tensor, torch.Tensor]
type RegressionModelBuilder = Callable[[int, tuple[int, ...]], nn.Module]


@dataclass(slots=True)
class TrainingConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 75
    seed: int = 42
    accelerator: str = "auto"
    devices: int | str = 1
    precision: str | int = "32-true"
    enable_progress_bar: bool = False


@dataclass(slots=True)
class RegressionMetrics:
    mse: float
    mae: float
    rmse: float
    r2: float


@dataclass(slots=True)
class RegressionRuntime:
    fit_seconds: float
    test_seconds: float
    predict_seconds: float
    total_seconds: float
    parameter_count: int


@dataclass(slots=True)
class RegressionRunResult:
    model: "RegressionLightningModule"
    device: str
    history: list[dict[str, float]]
    test_loss: float
    test_metrics: RegressionMetrics
    runtime: RegressionRuntime
    naive_test_metrics: RegressionMetrics
    data_config: RegressionDataConfig
    training_config: TrainingConfig


def compute_regression_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> RegressionMetrics:
    errors = predictions - targets
    mse = float(np.mean(np.square(errors)))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))
    target_variance = float(np.sum(np.square(targets - np.mean(targets))))
    residual_variance = float(np.sum(np.square(errors)))
    r2 = 1.0 - residual_variance / target_variance if target_variance > 0.0 else 0.0
    return RegressionMetrics(mse=mse, mae=mae, rmse=rmse, r2=r2)


class MetricHistoryCallback(Callback):
    def __init__(self) -> None:
        self.history: list[dict[str, float]] = []
        self._train_loss_by_epoch: dict[int, float] = {}

    @staticmethod
    def _metric_to_float(metric: object) -> float:
        if isinstance(metric, torch.Tensor):
            return float(metric.item())
        return float(cast(SupportsFloat, metric))

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            self._train_loss_by_epoch[trainer.current_epoch] = self._metric_to_float(
                train_loss
            )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        epoch_summary = {
            "epoch": float(trainer.current_epoch + 1),
            "train_loss": self._train_loss_by_epoch.get(
                trainer.current_epoch, float("nan")
            ),
            "val_loss": self._metric_to_float(metrics["val_loss"]),
            "val_rmse": self._metric_to_float(metrics["val_rmse"]),
            "val_r2": self._metric_to_float(metrics["val_r2"]),
        }
        self.history.append(epoch_summary)


class RegressionLightningModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        target_mean: float = 0.0,
        target_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_mean = target_mean
        self.target_scale = target_scale
        self.model = model
        self.loss_fn = nn.MSELoss()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _restore_original_target_scale(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.target_scale + self.target_mean

    @staticmethod
    def _batch_r2(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_centered = targets - torch.mean(targets)
        target_variance = torch.sum(torch.square(target_centered))
        residual_variance = torch.sum(torch.square(predictions - targets))
        if torch.isclose(
            target_variance, torch.zeros((), device=target_variance.device)
        ):
            return torch.tensor(0.0, device=targets.device)
        return 1.0 - residual_variance / target_variance

    def _shared_step(self, batch: Batch, stage: str) -> torch.Tensor:
        features, targets = batch
        predictions = self(features)
        loss = self.loss_fn(predictions, targets)
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=stage != "train",
        )

        predictions_original = self._restore_original_target_scale(predictions.detach())
        targets_original = self._restore_original_target_scale(targets.detach())
        rmse = torch.sqrt(
            torch.mean(torch.square(predictions_original - targets_original))
        )
        r2 = self._batch_r2(predictions_original, targets_original)

        if stage != "train":
            self.log(
                f"{stage}_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False
            )
            self.log(f"{stage}_r2", r2, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def predict_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        features, _ = batch
        return self(features)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def extra_parameter_count(self) -> int:
        extra_parameter_count = getattr(self.model, "extra_parameter_count", None)
        if callable(extra_parameter_count):
            return int(extra_parameter_count())
        return 0

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        clip_weights = getattr(self.model, "clip_weights_", None)
        if callable(clip_weights):
            clip_weights()
        apply_discrete_updates = getattr(self.model, "apply_discrete_updates_", None)
        if callable(apply_discrete_updates):
            apply_discrete_updates()


def trainer_device_name(trainer: L.Trainer) -> str:
    strategy_root_device = getattr(trainer.strategy, "root_device", None)
    return str(strategy_root_device) if strategy_root_device is not None else "cpu"


def target_scaler_stats(data_module: RegressionDataModule) -> tuple[float, float]:
    assert data_module.bundle is not None
    target_scaler = data_module.bundle.target_scaler
    target_mean = float(np.asarray(target_scaler.mean_).reshape(-1)[0])
    target_scale = float(np.asarray(target_scaler.scale_).reshape(-1)[0])
    return target_mean, target_scale


def build_lightning_model(
    data_module: RegressionDataModule,
    training_config: TrainingConfig,
    model_builder: RegressionModelBuilder,
) -> RegressionLightningModule:
    assert data_module.bundle is not None
    model = model_builder(data_module.bundle.input_dim, training_config.hidden_dims)
    return build_lightning_model_from_model(data_module, training_config, model)


def build_lightning_model_from_model(
    data_module: RegressionDataModule,
    training_config: TrainingConfig,
    model: nn.Module,
) -> RegressionLightningModule:
    assert data_module.bundle is not None
    target_mean, target_scale = target_scaler_stats(data_module)
    return RegressionLightningModule(
        model=model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        target_mean=target_mean,
        target_scale=target_scale,
    )


def build_trainer(
    training_config: TrainingConfig,
    checkpoint_dir: str,
    callbacks: list[Callback],
) -> L.Trainer:
    return L.Trainer(
        default_root_dir=checkpoint_dir,
        max_epochs=training_config.epochs,
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        precision=training_config.precision,
        logger=False,
        enable_progress_bar=training_config.enable_progress_bar,
        enable_model_summary=False,
        deterministic=True,
        callbacks=callbacks,
    )


def load_best_model_weights(
    model: RegressionLightningModule,
    checkpoint_callback: ModelCheckpoint,
) -> RegressionLightningModule:
    if not checkpoint_callback.best_model_path:
        return model

    checkpoint = torch.load(
        checkpoint_callback.best_model_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model


def predict_regression_metrics(
    trainer: L.Trainer,
    model: RegressionLightningModule,
    data_loader,
    original_targets: np.ndarray,
) -> RegressionMetrics:
    predictions_scaled_batches = trainer.predict(model, dataloaders=data_loader)
    tensor_batches = [
        batch
        for batch in (predictions_scaled_batches or [])
        if isinstance(batch, torch.Tensor)
    ]
    if not tensor_batches:
        raise RuntimeError("Prediction returned no tensor batches.")

    predictions_scaled = torch.cat(tensor_batches, dim=0).to(torch.float32).cpu().numpy()
    predictions_original = (
        predictions_scaled * model.target_scale + model.target_mean
    ).reshape(-1)
    return compute_regression_metrics(predictions_original, original_targets)


def naive_regression_metrics(
    train_targets: np.ndarray, test_targets: np.ndarray
) -> RegressionMetrics:
    naive_predictions = np.full_like(
        test_targets,
        fill_value=float(np.mean(train_targets)),
    )
    return compute_regression_metrics(naive_predictions, test_targets)


def _synchronize_if_cuda_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _count_model_parameters(model: nn.Module) -> int:
    total = sum(parameter.numel() for parameter in model.parameters())
    extra_parameter_count = getattr(model, "extra_parameter_count", None)
    if callable(extra_parameter_count):
        total += int(extra_parameter_count())
    return total


def train_regression_model(
    model_builder: RegressionModelBuilder | None = None,
    *,
    model: nn.Module | None = None,
    data_config: RegressionDataConfig | None = None,
    training_config: TrainingConfig | None = None,
) -> RegressionRunResult:
    data_config = data_config or RegressionDataConfig()
    training_config = training_config or TrainingConfig()
    if model_builder is None and model is None:
        raise ValueError("Either model_builder or model must be provided.")
    if model_builder is not None and model is not None:
        raise ValueError("Provide either model_builder or model, not both.")
    L.seed_everything(training_config.seed, workers=True)

    data_module = RegressionDataModule(data_config)
    data_module.setup("fit")
    assert data_module.bundle is not None
    data = data_module.bundle
    if model is None:
        assert model_builder is not None
        lightning_model = build_lightning_model(
            data_module, training_config, model_builder
        )
    else:
        lightning_model = build_lightning_model_from_model(
            data_module, training_config, model
        )
    parameter_count = _count_model_parameters(lightning_model)

    with tempfile.TemporaryDirectory(dir=str(checkpoint_root())) as checkpoint_dir:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        )
        history_callback = MetricHistoryCallback()
        trainer = build_trainer(
            training_config,
            checkpoint_dir,
            callbacks=[checkpoint_callback, history_callback],
        )

        _synchronize_if_cuda_available()
        fit_start = time.perf_counter()
        trainer.fit(lightning_model, datamodule=data_module)
        _synchronize_if_cuda_available()
        fit_seconds = time.perf_counter() - fit_start

        best_model = load_best_model_weights(lightning_model, checkpoint_callback)

        _synchronize_if_cuda_available()
        test_start = time.perf_counter()
        test_results = trainer.test(best_model, datamodule=data_module, verbose=False)
        _synchronize_if_cuda_available()
        test_seconds = time.perf_counter() - test_start
        test_loss = float(test_results[0]["test_loss"])

        _synchronize_if_cuda_available()
        predict_start = time.perf_counter()
        test_metrics = predict_regression_metrics(
            trainer=trainer,
            model=best_model,
            data_loader=data.test_loader,
            original_targets=data.test_targets_original,
        )
        _synchronize_if_cuda_available()
        predict_seconds = time.perf_counter() - predict_start
        device_name = trainer_device_name(trainer)
        history = history_callback.history

    naive_metrics = naive_regression_metrics(
        data.train_targets_original,
        data.test_targets_original,
    )
    runtime = RegressionRuntime(
        fit_seconds=fit_seconds,
        test_seconds=test_seconds,
        predict_seconds=predict_seconds,
        total_seconds=fit_seconds + test_seconds + predict_seconds,
        parameter_count=parameter_count,
    )

    return RegressionRunResult(
        model=best_model,
        device=device_name,
        history=history,
        test_loss=test_loss,
        test_metrics=test_metrics,
        runtime=runtime,
        naive_test_metrics=naive_metrics,
        data_config=data_config,
        training_config=training_config,
    )
