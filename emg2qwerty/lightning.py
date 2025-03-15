# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection
import torch.nn.functional as F

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.modules import TDSFullyConnectedBlock
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# RNN Module
class RNNCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_type: str = "LSTM",
        optimizer=DictConfig,
        lr_scheduler=DictConfig,
        decoder=DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # (T, N, bands=2, electrode_channels=16, freq)
        self.pre_model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            # (T, N, num_features)
        )

        # Replace TDSConvEncoder Use RNN
        if rnn_type.upper() == "LSTM":
            self.encoder = nn.LSTM(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
                bidirectional=False,
            )
        elif rnn_type.upper() == "GRU":
            self.encoder = nn.GRU(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )
        else:
            self.encoder = nn.RNN(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )

        self.output_fc = nn.Linear(rnn_hidden_size, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

        self._optimizer_cfg = optimizer
        self._lr_scheduler_cfg = lr_scheduler

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, bands=2, electrode_channels=16, freq)
        output: (T, N, num_classes)
        """
        x = self.pre_model(inputs)  # (T, N, num_features)

        x, _ = self.encoder(x)      # (T, N, rnn_hidden_size)

        x = self.output_fc(x)       # (T, N, num_classes)
        x = self.log_softmax(x)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


# CNN + RNN Module
# class TDSConvRNNCTCModule(pl.LightningModule):
#     NUM_BANDS: ClassVar[int] = 2
#     ELECTRODE_CHANNELS: ClassVar[int] = 16

#     def __init__(
#         self,
#         in_features: int,
#         mlp_features: Sequence[int],
#         block_channels: Sequence[int],
#         kernel_width: int,
#         optimizer: DictConfig,
#         lr_scheduler: DictConfig,
#         decoder: DictConfig,
#         rnn_hidden_size: int = 128,
#         rnn_num_layers: int = 1,
#         rnn_type: str = "LSTM",
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()

#         num_features = self.NUM_BANDS * mlp_features[-1]

#         # construct model
#         self.model = nn.Sequential(
#             # (T, N, bands=2, electrode_channels=16, freq)
#             SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
#             # (T, N, bands=2, mlp_features[-1])
#             MultiBandRotationInvariantMLP(
#                 in_features=in_features,
#                 mlp_features=mlp_features,
#                 num_bands=self.NUM_BANDS,
#             ),
#             # (T, N, num_features)
#             nn.Flatten(start_dim=2),
#             # (T, N, num_features)
#             TDSConvEncoder(
#                 num_features=num_features,
#                 block_channels=block_channels,
#                 kernel_width=kernel_width,
#             ),
#         )
#         # RNN Layer, support LSTM & GRU
#         if rnn_type.upper() == "LSTM":
#             self.rnn = nn.LSTM(
#                 input_size=num_features,
#                 hidden_size=rnn_hidden_size,
#                 num_layers=rnn_num_layers,
#                 batch_first=False,
#                 bidirectional=False,
#             )
#         elif rnn_type.upper() == "GRU":
#             self.rnn = nn.GRU(
#                 input_size=num_features,
#                 hidden_size=rnn_hidden_size,
#                 num_layers=rnn_num_layers,
#                 batch_first=False,
#             )
#         else:
#             self.rnn = nn.RNN(
#                 input_size=num_features,
#                 hidden_size=rnn_hidden_size,
#                 num_layers=rnn_num_layers,
#                 batch_first=False,
#             )

#         # (T, N, num_classes)
#         self.output_fc = nn.Linear(rnn_hidden_size, charset().num_classes)
#         self.log_softmax = nn.LogSoftmax(dim=-1)

#         # Criterion
#         self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

#         # Decoder
#         self.decoder = instantiate(decoder)

#         # Metrics
#         metrics = MetricCollection([CharacterErrorRates()])
#         self.metrics = nn.ModuleDict(
#             {
#                 f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
#                 for phase in ["train", "val", "test"]
#             }
#         )

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         # inputs: (T, N, bands, electrode_channels, freq)
#         x = self.model(inputs)
#         # RNN Layer: (T, N, rnn_hidden_size)
#         x, _ = self.rnn(x)
#         # Output Layer: (T, N, num_classes)
#         x = self.output_fc(x)
#         x = self.log_softmax(x)
#         return x

#     def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
#         inputs = batch["inputs"]
#         targets = batch["targets"]
#         input_lengths = batch["input_lengths"]
#         target_lengths = batch["target_lengths"]
#         N = len(input_lengths)  # batch size

#         emissions = self.forward(inputs)

#         # Shrink input lengths by an amount equivalent to the conv encoder's
#         # temporal receptive field to compute output activation lengths for CTCLoss.
#         # NOTE: This assumes the encoder doesn't perform any temporal downsampling
#         # such as by striding.
#         T_diff = inputs.shape[0] - emissions.shape[0]
#         emission_lengths = input_lengths - T_diff

#         loss = self.ctc_loss(
#             log_probs=emissions,
#             targets=targets.transpose(0, 1),
#             input_lengths=emission_lengths,
#             target_lengths=target_lengths,
#         )

#         # Decode emissions
#         predictions = self.decoder.decode_batch(
#             emissions=emissions.detach().cpu().numpy(),
#             emission_lengths=emission_lengths.detach().cpu().numpy(),
#         )

#         # Update metrics
#         metrics = self.metrics[f"{phase}_metrics"]
#         targets_np = targets.detach().cpu().numpy()
#         target_lengths_np = target_lengths.detach().cpu().numpy()
#         for i in range(N):
#             # Unpad targets (T, N) for batch entry
#             target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
#             metrics.update(prediction=predictions[i], target=target)

#         self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
#         return loss

#     def training_step(self, *args, **kwargs) -> torch.Tensor:
#         return self._step("train", *args, **kwargs)

#     def validation_step(self, *args, **kwargs) -> torch.Tensor:
#         return self._step("val", *args, **kwargs)

#     def test_step(self, *args, **kwargs) -> torch.Tensor:
#         return self._step("test", *args, **kwargs)

#     def on_train_epoch_end(self) -> None:
#         self._epoch_end("train")

#     def on_validation_epoch_end(self) -> None:
#         self._epoch_end("val")

#     def on_test_epoch_end(self) -> None:
#         self._epoch_end("test")

#     def _epoch_end(self, phase: str) -> None:
#         metrics = self.metrics[f"{phase}_metrics"]
#         self.log_dict(metrics.compute(), sync_dist=True)
#         metrics.reset()

#     def configure_optimizers(self) -> dict[str, Any]:
#         return utils.instantiate_optimizer_and_scheduler(
#             self.parameters(),
#             optimizer_config=self.hparams.optimizer,
#             lr_scheduler_config=self.hparams.lr_scheduler,
#         )


class RNNCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_type: str = "LSTM",
        optimizer=DictConfig,
        lr_scheduler=DictConfig,
        decoder=DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # (T, N, bands=2, electrode_channels=16, freq)
        self.pre_model = nn.Sequential(
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            nn.Flatten(start_dim=2),
            # (T, N, num_features)
        )

        # Replace TDSConvEncoder Use RNN
        if rnn_type.upper() == "LSTM":
            self.encoder = nn.LSTM(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
                bidirectional=False,
            )
        elif rnn_type.upper() == "GRU":
            self.encoder = nn.GRU(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )
        else:
            self.encoder = nn.RNN(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )

        self.output_fc = nn.Linear(rnn_hidden_size, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

        self._optimizer_cfg = optimizer
        self._lr_scheduler_cfg = lr_scheduler

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (T, N, bands=2, electrode_channels=16, freq)
        output: (T, N, num_classes)
        """
        x = self.pre_model(inputs)  # (T, N, num_features)

        x, _ = self.encoder(x)      # (T, N, rnn_hidden_size)

        x = self.output_fc(x)       # (T, N, num_classes)
        x = self.log_softmax(x)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class SimpleTransformerEncoder(nn.Module):

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transformer_encoder(x)
        return x


class TDSConvTransformerCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.tds_out_dim = self.NUM_BANDS * mlp_features[-1]

        self.specnorm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS
        )

        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        # (T, N, bands=2, mlp_features[-1])
        self.flatten = nn.Flatten(start_dim=2)
        # (T, N, num_features)

        self.tds_encoder = TDSConvEncoder(
            num_features=self.tds_out_dim,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )
        # (T, N, tds_out_dim/num_features)

        self.linear_projection = None
        if self.tds_out_dim != transformer_d_model:
            self.linear_projection = nn.Linear(self.tds_out_dim, transformer_d_model)

        self.transformer = SimpleTransformerEncoder(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=2048,
            dropout=0.1,
        )

        self.fc_out = nn.Linear(transformer_d_model, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

        self._optimizer_cfg = optimizer
        self._lr_scheduler_cfg = lr_scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, N, 2, 16, freq)
        output: (T, N, num_classes)
        """
        # 1) 
        x = self.specnorm(x)  # (T, N, 2, 16, freq)

        # 2) 
        x = self.mlp(x)       # (T, N, 2 * mlp_features[-1])
        x = self.flatten(x)   # (T, N, self.tds_out_dim)

        # 3) TDSConvEncoder
        x = self.tds_encoder(x)  # (T, N, self.tds_out_dim)

        if self.linear_projection is not None:
            x = self.linear_projection(x)  # (T, N, transformer_d_model)

        # 4) Transformer
        x = self.transformer(x)  # (T, N, transformer_d_model)

        # 5) 
        x = self.fc_out(x)       # (T, N, num_classes)
        x = self.log_softmax(x)  # (T, N, num_classes)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TransformerCTCModule(pl.LightningModule):
    """
    Only transformer, no TDSConv
    """
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        transformer_d_model: int,
        transformer_nhead: int,
        transformer_num_layers: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()

        self.specnorm = SpectrogramNorm(
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS
        )

        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)

        self.mlp_out_dim = self.NUM_BANDS * mlp_features[-1]

        self.linear_projection = None
        if self.mlp_out_dim != transformer_d_model:
            self.linear_projection = nn.Linear(self.mlp_out_dim, transformer_d_model)

        self.transformer = SimpleTransformerEncoder(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=2048,
            dropout=0.1,
        )

        self.fc_out = nn.Linear(transformer_d_model, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # CTC Loss
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

        self._optimizer_cfg = optimizer
        self._lr_scheduler_cfg = lr_scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.specnorm(x)  # (T, N, 2, 16, freq)

        x = self.mlp(x)       # (T, N, 2 * mlp_features[-1])
        x = self.flatten(x)   # (T, N, self.mlp_out_dim)

        if self.linear_projection is not None:
            x = self.linear_projection(x)  # (T, N, transformer_d_model)

        x = self.transformer(x)  # (T, N, transformer_d_model)

        x = self.fc_out(x)       # (T, N, num_classes)
        x = self.log_softmax(x)  # (T, N, num_classes)
        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class Upsampler(nn.Module):
    """
    Upsamples a tensor along a specified dimension using interpolation.
    For example, if your input tensor is (T, N, ...), and you want to upsample
    the time axis (T), set upsample_dim=0.
    """

    def __init__(self, factor: int, upsample_dim: int = 0):
        super().__init__()
        self.factor = factor
        self.upsample_dim = upsample_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute so that the dimension to be upsampled is at index 2 (after batch)
        dims = list(range(x.dim()))
        dims.pop(self.upsample_dim)
        dims.insert(1, self.upsample_dim)
        x = x.permute(*dims)  # now shape: (batch, T, ...)

        # Collapse dimensions after T into one
        b, T, *rest = x.shape
        x = x.contiguous().view(b, T, -1)  # shape: (b, T, C)

        # Transpose to (b, C, T) to use F.interpolate
        x = x.transpose(1, 2)  # shape: (b, C, T)

        # Upsample: use linear interpolation along T with scale factor = self.factor.
        # You can also compute the new size as: new_size = T * self.factor
        x = F.interpolate(x, scale_factor=self.factor,
                          mode='linear', align_corners=False)

        # Transpose back to (b, T_new, C)
        x = x.transpose(1, 2)  # shape: (b, T_new, C)

        # Reshape to restore any collapsed dimensions
        new_shape = [b, x.shape[1]] + rest
        x = x.view(*new_shape)

        # Inverse permutation: restore original order of dimensions
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        x = x.permute(*inv_dims)
        return x


class Downsampler(nn.Module):
    """
    Downsamples a tensor along a specified dimension using average pooling.
    For example, if your input is (T, N, ...) and you want to downsample the time axis (T),
    set downsample_dim=0.
    """

    def __init__(self, factor: int, downsample_dim: int = 0):
        super().__init__()
        self.factor = factor
        self.downsample_dim = downsample_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute so that the dimension to be downsampled is at index 2
        # This makes it easier to apply 1D pooling.
        dims = list(range(x.dim()))
        # Move the downsampled dim to position 2 (after batch and channel dims)
        dims.pop(self.downsample_dim)
        dims.insert(1, self.downsample_dim)
        x = x.permute(*dims)  # now shape: (batch, T, ...)

        # Collapse all dimensions after the time dimension into one dimension
        b, T, *rest = x.shape
        x = x.contiguous().view(b, T, -1)  # shape: (b, T, C)

        # Transpose to (b, C, T) so that we can use F.avg_pool1d
        x = x.transpose(1, 2)  # shape: (b, C, T)
        x = F.avg_pool1d(x, kernel_size=self.factor, stride=self.factor)
        # x now has shape: (b, C, T_new)

        # Transpose back and reshape to original dimensions (except with T downsampled)
        x = x.transpose(1, 2)  # shape: (b, T_new, C)
        T_new = x.shape[1]
        # Restore the other dimensions:
        new_shape = [b, T_new] + rest
        x = x.view(*new_shape)

        # Inverse permutation: put the dimensions back to original order.
        # Build inverse of dims permutation.
        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        x = x.permute(*inv_dims)
        return x


class ChannelSlicer(nn.Module):
    def __init__(self, start_channel: int, end_channel: int):
        super().__init__()
        self.start_channel = start_channel
        self.end_channel = end_channel

    def forward(self, x):
        # Assuming input shape: (T, N, bands, channels, freq)
        return x[:, :, :, self.start_channel:self.end_channel, :]


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
        jitter: bool = True,
        downsample_factor: int = 2,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvRNNCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 1,
        rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # construct model
        self.model = nn.Sequential(
            # (T, N, bands=2, electrode_channels=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            # (T, N, num_features)
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
        )
        # RNN Layer, support LSTM & GRU
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
                bidirectional=True,
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )
        else:
            self.rnn = nn.RNN(
                input_size=num_features,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                batch_first=False,
            )

        # (T, N, num_classes)
        self.fc_block1 = TDSFullyConnectedBlock(rnn_hidden_size*2)
        self.fc_block2 = nn.Linear(rnn_hidden_size*2, num_features)
        self.output_fc = nn.Linear(num_features, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, bands, electrode_channels, freq)
        x = self.model(inputs)
        # RNN Layer: (T, N, rnn_hidden_size)
        x, _ = self.rnn(x)
        # Output Layer: (T, N, num_classes)
        x = self.fc_block1(x)
        x = self.fc_block2(x)
        x = self.output_fc(x)
        x = self.log_softmax(x)
        return x
    
    def _step(self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,
            targets=targets.transpose(0, 1),
            input_lengths=emission_lengths,
            target_lengths=target_lengths,
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lengths_np = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
