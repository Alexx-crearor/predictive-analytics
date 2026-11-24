"""LSTM and DLinear reference baselines for the P100/H8 final-project protocol.

Input: [batch, lookback, channels]
Output: [batch, channels * horizon] (feature-major), like M0--M4.
"""
from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Sequence-to-vector LSTM; one shared linear head predicts all channels/leads."""

    def __init__(self, input_dim: int, seq_len: int, horizon: int,
                 hidden_size: int = 96, num_layers: int = 1):
        super().__init__()
        if min(input_dim, seq_len, horizon, hidden_size, num_layers) < 1:
            raise ValueError("All dimensions and layer counts must be positive")
        self.seq_len, self.input_dim, self.horizon = seq_len, input_dim, horizon
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.projection = nn.Linear(hidden_size, input_dim * horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"Expected [B,{self.seq_len},{self.input_dim}], got {tuple(x.shape)}")
        _, (h_n, _) = self.lstm(x)
        return self.projection(h_n[-1])


class MovingAvg(nn.Module):
    """Odd-sized moving average with edge-value replication."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be positive and odd")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        padded = torch.nn.functional.pad(x.transpose(1, 2), (pad, pad), mode="replicate")
        return self.pool(padded).transpose(1, 2)


class DLinearForecaster(nn.Module):
    """Decomposition-Linear, channel-independent seasonal/trend projections.

    Both projections operate along the lookback axis on every channel.
    The final flatten is feature-major. Never repeat a scalar across leads.
    """

    def __init__(self, input_dim: int, seq_len: int, horizon: int,
                 moving_avg: int = 25):
        super().__init__()
        if min(input_dim, seq_len, horizon) < 1:
            raise ValueError("All dimensions must be positive")
        self.seq_len, self.input_dim, self.horizon = seq_len, input_dim, horizon
        self.moving_avg = MovingAvg(moving_avg)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        self.linear_trend = nn.Linear(seq_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(f"Expected [B,{self.seq_len},{self.input_dim}], got {tuple(x.shape)}")
        trend = self.moving_avg(x)
        seasonal = x - trend
        # [B, C, H] -> flatten channel then time (feature-major)
        output = (self.linear_seasonal(seasonal.transpose(1, 2))
                  + self.linear_trend(trend.transpose(1, 2)))
        return output.contiguous().reshape(x.shape[0], -1)
