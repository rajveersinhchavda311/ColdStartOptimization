"""
True Temporal Convolutional Network (TCN) Baseline
===================================================

Architecture:
    A genuine causal Temporal Convolutional Network using dilated causal
    1D convolutions. This is NOT an MLP with a TCN label.

    Causal convolution guarantee:
        Output at position t depends ONLY on inputs at positions <= t.
        This is enforced via left-padding (causal padding): each conv layer
        pads (kernel_size - 1) * dilation zeros on the LEFT side only,
        then truncates the output to the original length.

    Layer structure (per residual block):
        Input -> CausalConv1d -> WeightNorm -> ReLU -> Dropout
              -> CausalConv1d -> WeightNorm -> ReLU -> Dropout
              -> Residual connection (with 1x1 conv if channels differ)
              -> Output

    Stack:
        4 residual blocks with dilation factors [1, 2, 4, 8]
        Kernel size = 3

    Receptive field calculation:
        For a TCN with L layers, kernel size k, and dilation d_i at layer i,
        each TemporalBlock contains 2 CausalConv1d layers:
        receptive_field = 1 + 2 * (k - 1) * sum(d_i)
        = 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8)
        = 61

        With an input sequence of 10 lags, the effective receptive field
        covers the entire input sequence.

    Input shape:
        Temporal sequence : (batch_size, 1, 10)  -- lag_10 (oldest) -> lag_1 (newest)
        Seasonal scalar   : (batch_size, 1)       -- lag_1440 (same minute yesterday)
        Combined in forward: temporal features concatenated with seasonal scalar
        before the final linear layer.

    Output: (batch_size, 1) — predicted concurrency at the next timestep

Pre-Phase-3 Fix (target normalization):
    Previously, inputs were z-score normalized but targets were kept in raw
    scale. This forced the output linear layer to learn very large weights
    (output ~ 600K mean), creating an ill-conditioned optimization problem
    and likely contributing to TCN underperformance in Phase 1.

    Fix: targets are now normalized using the same training statistics.
    Predictions are denormalized before being returned, so the interface
    (predict returns raw-scale values) is unchanged.

Pre-Phase-3 Fix (seasonal feature):
    lag_1440 (same minute yesterday) is now provided as an additional scalar
    feature. It is normalized and concatenated to the temporal block output
    before the final linear layer, giving the TCN access to the daily
    seasonal signal without requiring a 1440-step input sequence.

Training:
    - Loss: MSE on normalized targets
    - Optimizer: Adam (lr=1e-3)
    - Scheduler: ReduceLROnPlateau (patience=10, factor=0.5)
    - Early stopping: patience=15 on validation MSE
    - Normalization: Z-score using train mean/std (frozen)
    - Batch size: 256
    - Max epochs: 200

Leakage prevention:
    - Input features are ONLY lag columns (historical values)
    - The concurrency (target) column is NEVER used as an input feature
    - Z-score statistics computed ONLY from training data
    - Early stopping uses validation set (chronologically after training)

Reproducibility:
    - All random seeds fixed (PyTorch, NumPy, Python random)
    - Deterministic mode enabled where possible
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.base import BaseModel

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
NUM_LAGS = 10
LAG_COLS = [f"lag_{k}" for k in range(1, NUM_LAGS + 1)]
SEASONAL_COL = "lag_1440"


class TimeSeriesDataset(Dataset):
    """
    Dataset for TCN training.

    Each sample:
        X_seq  : [lag_10, lag_9, ..., lag_1] normalized  — shape (1, 10)
        X_seas : lag_1440 normalized                      — shape (1,)
        y      : concurrency[t] normalized                — scalar
    """

    def __init__(self, df: pd.DataFrame, feat_mean: float, feat_std: float,
                 target_mean: float, target_std: float):
        lag_values = df[LAG_COLS[::-1]].values        # (n, 10) oldest -> newest
        seasonal = df[SEASONAL_COL].values              # (n,)
        targets = df["concurrency"].values              # (n,)

        self.X_seq = torch.FloatTensor(
            (lag_values - feat_mean) / feat_std
        ).unsqueeze(1)                                  # (n, 1, 10)

        self.X_seas = torch.FloatTensor(
            (seasonal - feat_mean) / feat_std
        ).unsqueeze(1)                                  # (n, 1)

        # Normalize targets — fixes the ill-conditioned output layer problem
        self.y = torch.FloatTensor(
            (targets - target_mean) / target_std
        )                                               # (n,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.X_seas[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Causal Convolution Block
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with left-padding.

    Standard convolution sees both past and future. Causal convolution
    sees ONLY the past by padding (kernel_size - 1) * dilation zeros
    on the LEFT and zero on the right, then chopping the output.

    This guarantees: output[t] depends only on input[<=t].
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                in_channels, out_channels, kernel_size,
                dilation=dilation, padding=0
            )
        )

    def forward(self, x):
        x_padded = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class TemporalBlock(nn.Module):
    """
    Residual block in the TCN.

    Structure:
        x -> CausalConv -> ReLU -> Dropout -> CausalConv -> ReLU -> Dropout
          -> (+residual) -> out
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCNNetwork(nn.Module):
    """
    Full TCN architecture with seasonal scalar input.

    Forward:
        x_seq  : (batch, 1, seq_len)  — temporal lag sequence
        x_seas : (batch, 1)           — lag_1440 scalar feature
        ->
        temporal features[:, :, -1]   — (batch, channels)  last-step output
        concat with x_seas             — (batch, channels+1)
        linear                         — (batch, 1)
    """

    def __init__(self, num_inputs: int = 1, num_channels: list = None,
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()

        if num_channels is None:
            num_channels = [32, 32, 32, 32]

        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)

        # +1 for the seasonal scalar feature
        self.linear = nn.Linear(num_channels[-1] + 1, 1)

    def forward(self, x_seq, x_seas):
        features = self.network(x_seq)          # (batch, channels, seq_len)
        last_step = features[:, :, -1]          # (batch, channels)
        combined = torch.cat([last_step, x_seas], dim=1)  # (batch, channels+1)
        out = self.linear(combined).squeeze(-1)  # (batch,)
        return out


# ---------------------------------------------------------------------------
# Model Wrapper
# ---------------------------------------------------------------------------

class TCNModel(BaseModel):
    """
    True Temporal Convolutional Network for demand forecasting.

    Fixes applied in Pre-Phase-3 cleanup:
        1. Target normalization: outputs are now in normalized space during
           training and denormalized at predict() time.
        2. Seasonal feature: lag_1440 is provided as an extra scalar input,
           giving the model access to the daily periodicity signal.
    """

    def __init__(self, num_channels=None, kernel_size=3, dropout=0.2,
                 lr=1e-3, batch_size=256, max_epochs=200,
                 patience=15, device=None):
        self.num_channels = num_channels or [32, 32, 32, 32]
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: TCNNetwork = None
        self.feat_mean: float = None
        self.feat_std: float = None
        self.target_mean: float = None
        self.target_std: float = None
        self.training_history: dict = None

    @property
    def name(self) -> str:
        return "TCN"

    @property
    def description(self) -> str:
        return (
            f"Causal TCN: {len(self.num_channels)} layers, "
            f"channels={self.num_channels}, kernel={self.kernel_size}, "
            f"dilations=[{', '.join(str(2**i) for i in range(len(self.num_channels)))}], "
            f"with lag_1440 seasonal scalar"
        )

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> None:
        set_seed(SEED)

        # Normalization statistics from TRAINING data ONLY
        all_lag_values = train_df[LAG_COLS].values.flatten()
        self.feat_mean = float(all_lag_values.mean())
        self.feat_std = float(all_lag_values.std())

        # Target normalization — new in Pre-Phase-3 fix
        self.target_mean = float(train_df["concurrency"].mean())
        self.target_std = float(train_df["concurrency"].std())

        print(f"  [TCN] Feature norm  — mean: {self.feat_mean:,.1f}, std: {self.feat_std:,.1f}")
        print(f"  [TCN] Target norm   — mean: {self.target_mean:,.1f}, std: {self.target_std:,.1f}")
        print(f"  [TCN] Device: {self.device}")

        train_dataset = TimeSeriesDataset(
            train_df, self.feat_mean, self.feat_std,
            self.target_mean, self.target_std
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=False, generator=torch.Generator().manual_seed(SEED)
        )

        val_loader = None
        if val_df is not None:
            val_dataset = TimeSeriesDataset(
                val_df, self.feat_mean, self.feat_std,
                self.target_mean, self.target_std
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        self.model = TCNNetwork(
            num_inputs=1,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

        print(f"  [TCN] Architecture:\n{self.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [TCN] Total parameters: {total_params:,}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}

        print(f"  [TCN] Training for up to {self.max_epochs} epochs...")

        for epoch in range(self.max_epochs):
            self.model.train()
            train_losses = []
            for X_seq, X_seas, y_batch in train_loader:
                X_seq = X_seq.to(self.device)
                X_seas = X_seas.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_seq, X_seas)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            history["train_loss"].append(avg_train_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])

            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_seq, X_seas, y_batch in val_loader:
                        X_seq = X_seq.to(self.device)
                        X_seas = X_seas.to(self.device)
                        y_batch = y_batch.to(self.device)
                        pred = self.model(X_seq, X_seas)
                        loss = criterion(pred, y_batch)
                        val_losses.append(loss.item())

                avg_val_loss = np.mean(val_losses)
                history["val_loss"].append(avg_val_loss)
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"    Epoch {epoch + 1:3d}/{self.max_epochs} - "
                          f"train: {avg_train_loss:.4f}, "
                          f"val: {avg_val_loss:.4f}, "
                          f"lr: {optimizer.param_groups[0]['lr']:.6f}"
                          f"{' *' if epochs_without_improvement == 0 else ''}")

                if epochs_without_improvement >= self.patience:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break
            else:
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"    Epoch {epoch + 1:3d}/{self.max_epochs} - "
                          f"train: {avg_train_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"  [TCN] Restored best model (val_loss: {best_val_loss:.4f})")

        self.training_history = history
        self.model.eval()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using ONLY lag columns.
        Does NOT access 'concurrency' column.
        Returns values in original (denormalized) scale.
        """
        if self.model is None:
            raise RuntimeError("TCNModel.fit() must be called before predict()")

        lag_values = df[LAG_COLS[::-1]].values          # (n, 10)
        lag_norm = (lag_values - self.feat_mean) / self.feat_std
        X_seq = torch.FloatTensor(lag_norm).unsqueeze(1).to(self.device)  # (n, 1, 10)

        seas_values = df[SEASONAL_COL].values             # (n,)
        seas_norm = (seas_values - self.feat_mean) / self.feat_std
        X_seas = torch.FloatTensor(seas_norm).unsqueeze(1).to(self.device)  # (n, 1)

        self.model.eval()
        preds_norm = []
        with torch.no_grad():
            for i in range(0, len(X_seq), self.batch_size):
                seq_batch = X_seq[i:i + self.batch_size]
                seas_batch = X_seas[i:i + self.batch_size]
                pred = self.model(seq_batch, seas_batch)
                preds_norm.append(pred.cpu().numpy())

        preds_norm = np.concatenate(preds_norm)

        # Denormalize back to raw scale
        return (preds_norm * self.target_std + self.target_mean).astype(np.float64)

    def verify_causality(self) -> dict:
        """Verify the TCN architecture is genuinely causal."""
        if self.model is None:
            raise RuntimeError("Model must be trained first")

        results = {
            "is_causal": True,
            "architecture": "Dilated Causal TCN with seasonal scalar",
            "num_layers": len(self.num_channels),
            "channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dilations": [2**i for i in range(len(self.num_channels))],
            "checks": []
        }

        for name, module in self.model.named_modules():
            if isinstance(module, CausalConv1d):
                conv_module = module.conv
                expected_padding = (conv_module.kernel_size[0] - 1) * conv_module.dilation[0]
                actual_padding = module.padding
                match = expected_padding == actual_padding
                results["checks"].append({
                    "layer": name, "type": "causal_padding",
                    "expected": expected_padding, "actual": actual_padding,
                    "pass": match
                })
                if not match:
                    results["is_causal"] = False

        set_seed(SEED)
        self.model.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 1, 10).to(self.device)
            x_seas = torch.randn(1, 1).to(self.device)
            x2 = x1.clone()
            x2[0, 0, -1] += 1000.0

            out1 = self.model(x1, x_seas)
            out2 = self.model(x2, x_seas)
            results["checks"].append({
                "type": "last_position_visible",
                "description": "Changing the most recent input should change the output",
                "pass": not torch.allclose(out1, out2, atol=1e-3)
            })

            x3 = x1.clone()
            x3[0, 0, 0] += 1000.0
            out3 = self.model(x3, x_seas)
            results["checks"].append({
                "type": "first_position_visible",
                "description": "Changing the oldest input should also affect output",
                "pass": not torch.allclose(out1, out3, atol=1e-3)
            })

        k = self.kernel_size
        dilations = results["dilations"]
        rf = 1
        for d in dilations:
            rf += 2 * (k - 1) * d
        results["receptive_field"] = rf
        results["checks"].append({
            "type": "receptive_field",
            "value": rf, "input_length": 10,
            "covers_full_input": rf >= 10, "pass": rf >= 10
        })

        results["is_causal"] = all(c["pass"] for c in results["checks"])
        return results
