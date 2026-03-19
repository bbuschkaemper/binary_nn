from __future__ import annotations

import copy
import tempfile
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import SupportsFloat, cast

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import nn

from output_paths import checkpoint_root
from regression_data import (
    RegressionDataConfig,
    RegressionDataModule,
    create_regression_dataloaders,
)


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
    stage_benchmarks: RegressionStageBenchmarks | None = None


@dataclass(slots=True)
class StepBenchmarkConfig:
    repetitions: int = 5
    warmup_steps: int = 2
    timed_steps: int = 10


@dataclass(slots=True)
class StageBenchmarkResult:
    benchmark_device: str
    precision: str | int
    batch_size: int
    repetitions: int
    warmup_steps: int
    timed_steps: int
    mean_step_ms: float
    std_step_ms: float
    min_step_ms: float
    max_step_ms: float
    samples_per_second: float
    peak_memory_mb: float | None


@dataclass(slots=True)
class RegressionStageBenchmarks:
    fit: StageBenchmarkResult | None = None
    test: StageBenchmarkResult | None = None
    predict: StageBenchmarkResult | None = None


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


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _count_model_parameters(model: nn.Module) -> int:
    total = sum(parameter.numel() for parameter in model.parameters())
    extra_parameter_count = getattr(model, "extra_parameter_count", None)
    if callable(extra_parameter_count):
        total += int(extra_parameter_count())
    return total


def _resolve_benchmark_device(device_name: str) -> torch.device:
    return torch.device(device_name)


def _autocast_dtype(
    precision: str | int,
    device: torch.device,
) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    normalized = str(precision)
    if normalized in {"bf16", "bf16-mixed", "bf16-true"}:
        return torch.bfloat16
    if normalized in {"16", "16-mixed", "16-true"}:
        return torch.float16
    return None


def _precision_context(
    precision: str | int,
    device: torch.device,
):
    autocast_dtype = _autocast_dtype(precision, device)
    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=autocast_dtype)


def _uses_fp16_grad_scaler(precision: str | int, device: torch.device) -> bool:
    return device.type == "cuda" and str(precision) in {"16", "16-mixed"}


def _apply_model_post_step_hooks(model: nn.Module) -> None:
    clip_weights = getattr(model, "clip_weights_", None)
    if callable(clip_weights):
        clip_weights()
    apply_discrete_updates = getattr(model, "apply_discrete_updates_", None)
    if callable(apply_discrete_updates):
        apply_discrete_updates()


def _prepare_model_for_evaluation(model: nn.Module) -> None:
    prepare_for_evaluation = getattr(model, "prepare_for_evaluation_", None)
    if callable(prepare_for_evaluation):
        prepare_for_evaluation()


def _prepare_model_for_fit_stage_benchmark(model: nn.Module) -> None:
    for module in model.modules():
        prepare_for_fit_stage_benchmark = getattr(
            module,
            "prepare_for_fit_stage_benchmark_",
            None,
        )
        if callable(prepare_for_fit_stage_benchmark):
            prepare_for_fit_stage_benchmark()


def _fit_stage_benchmark_config_for_model(
    model: nn.Module,
    config: StepBenchmarkConfig,
) -> StepBenchmarkConfig:
    cycle_lengths: list[int] = []
    for module in model.modules():
        fit_stage_benchmark_cycle_length = getattr(
            module,
            "fit_stage_benchmark_cycle_length",
            None,
        )
        if not callable(fit_stage_benchmark_cycle_length):
            continue
        cycle_length = int(fit_stage_benchmark_cycle_length())
        if cycle_length > 1:
            cycle_lengths.append(cycle_length)
    if not cycle_lengths:
        return config

    cycle_length = max(cycle_lengths)
    timed_steps = max(config.timed_steps, cycle_length * 2)
    remainder = timed_steps % cycle_length
    if remainder != 0:
        timed_steps += cycle_length - remainder
    if timed_steps == config.timed_steps:
        return config
    return replace(config, timed_steps=timed_steps)


def _clone_model_for_stage_benchmark(
    run_result: RegressionRunResult,
    device: torch.device,
) -> nn.Module:
    model = copy.deepcopy(run_result.model.model)
    model.to(device)
    return model


def _move_batch_to_device(
    batch: Batch,
    device: torch.device,
) -> Batch:
    features, targets = batch
    return (
        features.to(device=device, non_blocking=True),
        targets.to(device=device, non_blocking=True),
    )


def _benchmark_step_runner(
    *,
    benchmark_device: torch.device,
    precision: str | int,
    batch_size: int,
    config: StepBenchmarkConfig,
    step_factory: Callable[[], Callable[[], None]],
) -> StageBenchmarkResult:
    if config.repetitions < 1:
        raise ValueError("Step benchmark repetitions must be at least 1.")
    if config.timed_steps < 1:
        raise ValueError("Step benchmark timed_steps must be at least 1.")

    step_times_ms: list[float] = []
    peak_memory_mb: float | None = None
    for _ in range(config.repetitions):
        step = step_factory()
        for _ in range(config.warmup_steps):
            step()
        _synchronize_device(benchmark_device)
        if benchmark_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(benchmark_device)
        start = time.perf_counter()
        for _ in range(config.timed_steps):
            step()
        _synchronize_device(benchmark_device)
        step_time_ms = (time.perf_counter() - start) * 1000.0 / config.timed_steps
        step_times_ms.append(step_time_ms)
        if benchmark_device.type == "cuda":
            peak_memory_mb = max(
                peak_memory_mb or 0.0,
                float(torch.cuda.max_memory_allocated(benchmark_device))
                / (1024.0 * 1024.0),
            )

    mean_step_ms = float(np.mean(step_times_ms))
    std_step_ms = float(np.std(step_times_ms))
    return StageBenchmarkResult(
        benchmark_device=str(benchmark_device),
        precision=precision,
        batch_size=batch_size,
        repetitions=config.repetitions,
        warmup_steps=config.warmup_steps,
        timed_steps=config.timed_steps,
        mean_step_ms=mean_step_ms,
        std_step_ms=std_step_ms,
        min_step_ms=float(np.min(step_times_ms)),
        max_step_ms=float(np.max(step_times_ms)),
        samples_per_second=(batch_size * 1000.0 / mean_step_ms),
        peak_memory_mb=peak_memory_mb,
    )


def benchmark_regression_run_result_stages(
    run_result: RegressionRunResult,
    benchmark_config: StepBenchmarkConfig,
) -> RegressionStageBenchmarks:
    benchmark_device = _resolve_benchmark_device(run_result.device)
    bundle = create_regression_dataloaders(run_result.data_config)
    train_batch = _move_batch_to_device(next(iter(bundle.train_loader)), benchmark_device)
    test_batch = _move_batch_to_device(next(iter(bundle.test_loader)), benchmark_device)
    precision = run_result.training_config.precision
    fit_benchmark_config = _fit_stage_benchmark_config_for_model(
        run_result.model.model,
        benchmark_config,
    )

    train_batch_size = int(train_batch[0].shape[0])
    test_batch_size = int(test_batch[0].shape[0])

    def build_fit_step() -> Callable[[], None]:
        model = _clone_model_for_stage_benchmark(run_result, benchmark_device)
        model.train()
        _prepare_model_for_fit_stage_benchmark(model)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=run_result.training_config.learning_rate,
            weight_decay=run_result.training_config.weight_decay,
        )
        scaler = (
            torch.cuda.amp.GradScaler()
            if _uses_fp16_grad_scaler(precision, benchmark_device)
            else None
        )
        loss_fn = nn.MSELoss()
        features, targets = train_batch

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(precision, benchmark_device):
                predictions = model(features)
                loss = loss_fn(predictions, targets)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            _apply_model_post_step_hooks(model)

        return step

    def build_test_step() -> Callable[[], None]:
        model = _clone_model_for_stage_benchmark(run_result, benchmark_device)
        model.eval()
        loss_fn = nn.MSELoss()
        features, targets = test_batch

        def step() -> None:
            with torch.no_grad():
                with _precision_context(precision, benchmark_device):
                    predictions = model(features)
                    loss_fn(predictions, targets)

        return step

    def build_predict_step() -> Callable[[], None]:
        model = _clone_model_for_stage_benchmark(run_result, benchmark_device)
        model.eval()
        features, _ = test_batch

        def step() -> None:
            with torch.no_grad():
                with _precision_context(precision, benchmark_device):
                    model(features)

        return step

    return RegressionStageBenchmarks(
        fit=_benchmark_step_runner(
            benchmark_device=benchmark_device,
            precision=precision,
            batch_size=train_batch_size,
            config=fit_benchmark_config,
            step_factory=build_fit_step,
        ),
        test=_benchmark_step_runner(
            benchmark_device=benchmark_device,
            precision=precision,
            batch_size=test_batch_size,
            config=benchmark_config,
            step_factory=build_test_step,
        ),
        predict=_benchmark_step_runner(
            benchmark_device=benchmark_device,
            precision=precision,
            batch_size=test_batch_size,
            config=benchmark_config,
            step_factory=build_predict_step,
        ),
    )


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
        _prepare_model_for_evaluation(best_model.model)

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
