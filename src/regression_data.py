from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass(slots=True)
class RegressionDataConfig:
    n_samples: int = 4096
    n_features: int = 10
    n_informative: int = 10
    noise: float = 12.0
    batch_size: int = 128
    num_workers: int = 0
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    random_state: int = 42


@dataclass(slots=True)
class RegressionDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    train_targets_original: np.ndarray
    val_targets_original: np.ndarray
    test_targets_original: np.ndarray


def create_regression_dataloaders(config: RegressionDataConfig) -> RegressionDataBundle:
    features, targets = make_regression(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_informative=config.n_informative,
        noise=config.noise,
        random_state=config.random_state,
    )

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        targets,
        test_size=config.test_fraction,
        random_state=config.random_state,
    )
    relative_val_fraction = config.val_fraction / (1.0 - config.test_fraction)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_fraction,
        random_state=config.random_state,
    )

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    x_train_scaled = feature_scaler.fit_transform(x_train)
    x_val_scaled = feature_scaler.transform(x_val)
    x_test_scaled = feature_scaler.transform(x_test)

    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).astype(
        np.float32
    )
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).astype(np.float32)
    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_scaled.astype(np.float32)),
        torch.from_numpy(y_train_scaled),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val_scaled.astype(np.float32)),
        torch.from_numpy(y_val_scaled),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(x_test_scaled.astype(np.float32)),
        torch.from_numpy(y_test_scaled),
    )

    return RegressionDataBundle(
        train_loader=DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        ),
        val_loader=DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        test_loader=DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
        input_dim=config.n_features,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        train_targets_original=y_train.astype(np.float32),
        val_targets_original=y_val.astype(np.float32),
        test_targets_original=y_test.astype(np.float32),
    )


class RegressionDataModule(L.LightningDataModule):
    def __init__(self, config: RegressionDataConfig) -> None:
        super().__init__()
        self.config = config
        self.bundle: RegressionDataBundle | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.bundle is None:
            self.bundle = create_regression_dataloaders(self.config)

    def train_dataloader(self) -> DataLoader:
        self.setup("fit")
        assert self.bundle is not None
        return self.bundle.train_loader

    def val_dataloader(self) -> DataLoader:
        self.setup("validate")
        assert self.bundle is not None
        return self.bundle.val_loader

    def test_dataloader(self) -> DataLoader:
        self.setup("test")
        assert self.bundle is not None
        return self.bundle.test_loader
