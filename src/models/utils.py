import torch
import torch.nn as nn


class EmbeddingDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = super(EmbeddingDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1).squeeze(2)
        return x


class LstmGruSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_lstm_size: int,
        hidden_gru_size: int,
        dropout_rate: float,
    ) -> None:
        super(LstmGruSequenceEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_lstm_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_lnorm = nn.LayerNorm(2 * hidden_lstm_size)

        self.gru = nn.GRU(
            input_size=2 * hidden_lstm_size,
            hidden_size=hidden_gru_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_lnorm = nn.LayerNorm(2 * hidden_gru_size)

        self.linears = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * input_size, input_size),
        )
        self.linears_lnorm = nn.LayerNorm(input_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, S, F]
        x = self.lstm_lnorm(x)  # [B, S, F]

        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # [B, S, F]
        x = self.gru_lnorm(x)  # [B, S, F]
        x_max, _ = torch.max(x, 1)  # [B, F]
        x_avg = torch.mean(x, 1)  # [B, F]
        x_last = x[:, -1]  # [B, F]

        x = torch.cat([x_max, x_avg, x_last], -1)  # [B, 3 * F]
        x = x + self.linears_lnorm(self.linears(x))

        return x
