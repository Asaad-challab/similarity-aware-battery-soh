import torch
import torch.nn as nn


class LSTMSOH(nn.Module):
    """
    LSTM model for SOH estimation from charging-based health indicators.

    Input shape:
        (batch_size, sequence_length, input_dim)

    Output:
        SOH prediction for the last timestep
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, input_dim)
        """

        # LSTM forward pass
        out, _ = self.lstm(x)

        # Use output of last timestep
        last_hidden = out[:, -1, :]

        # Regress SOH
        soh = self.regressor(last_hidden)

        return soh.squeeze(-1)
