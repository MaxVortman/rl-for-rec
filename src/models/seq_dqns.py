import torch
import torch.nn as nn


class SeqDQN(nn.Module):
    def __init__(
        self,
        action_n: int,
        embedding_dim: int = 32,
        padding_idx: int = 0,
        hidden_lstm_size: int = 64,
        hidden_gru_size: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super(SeqDQN, self).__init__()

        self.action_embedding = nn.Embedding(
            action_n + 1, embedding_dim, padding_idx=padding_idx  # + padding index
        )

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

        input_size = 3 * 2 * hidden_gru_size  # 2 pooling layers + last state

        self.linears = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * input_size, input_size),
        )
        self.linears_lnorm = nn.LayerNorm(input_size)
        self.head = nn.Linear(input_size, action_n + 1)

    def forward(self, state):
        x = self.action_embedding(state)  # [B, S] -> [B, S, F]

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)  # [B, S, F]
        x = self.lstm_lnorm(x)  # [B, S, F]

        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # [B, S, F]
        x = self.gru_lnorm(x)  # [B, S, F]
        x_max, _ = torch.max(x, 1)  # [B, F]
        x_avg = torch.mean(x, 1)  # [B, F]
        x_last = x[:, -1]  # [B, F]

        x = torch.cat([x_max, x_avg, x_last], -1) # [B, 3 * F]
        x = x + self.linears_lnorm(self.linears(x))

        x = self.head(x)
        
        return x
