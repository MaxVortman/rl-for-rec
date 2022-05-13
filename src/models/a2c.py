import torch
import torch.nn as nn
from .utils import EmbeddingDropout, LstmGruSequenceEncoder


class Critic(nn.Module):
    # value network
    def __init__(
        self,
        action_n,
        embedding_dim,
        hidden_lstm_size,
        hidden_gru_size,
        padding_idx,
        dropout_rate=0.1,
        init_w=3e-5,
    ):
        super(Critic, self).__init__()

        self.action_embedding = nn.Embedding(
            action_n + 1, embedding_dim, padding_idx=padding_idx  # + padding index
        )

        self.embedding_dropout = EmbeddingDropout(dropout_rate)

        input_size = 3 * 2 * hidden_gru_size  # 2 pooling layers + last state

        self.seq_encoder = LstmGruSequenceEncoder(
            input_size=input_size,
            embedding_dim=embedding_dim,
            hidden_lstm_size=hidden_lstm_size,
            hidden_gru_size=hidden_gru_size,
            dropout_rate=dropout_rate,
        )

        self.linears = nn.Sequential(
            nn.Linear(input_size + action_n + 1, 4 * input_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * input_size, input_size),
        )

        self.head = nn.Linear(input_size, 1)

        self.head.weight.data.uniform_(-init_w, init_w)
        self.head.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = self.action_embedding(state)  # [B, S] -> [B, S, F]
        x = self.embedding_dropout(x)

        x = self.seq_encoder(x)

        logits = torch.log(action)
        x = torch.cat([x, logits], 1)
        x = self.linears(x)

        x = self.head(x)
        return x


class Actor(nn.Module):
    # policy network
    def __init__(
        self,
        action_n,
        embedding_dim,
        hidden_lstm_size,
        hidden_gru_size,
        padding_idx,
        dropout_rate=0.1,
    ):
        super(Actor, self).__init__()

        self.action_embedding = nn.Embedding(
            action_n + 1, embedding_dim, padding_idx=padding_idx  # + padding index
        )

        self.embedding_dropout = EmbeddingDropout(dropout_rate)

        input_size = 3 * 2 * hidden_gru_size  # 2 pooling layers + last state

        self.seq_encoder = LstmGruSequenceEncoder(
            input_size=input_size,
            embedding_dim=embedding_dim,
            hidden_lstm_size=hidden_lstm_size,
            hidden_gru_size=hidden_gru_size,
            dropout_rate=dropout_rate,
        )

        self.head = nn.Sequential(
            nn.Linear(input_size, action_n + 1),  # + padding index
            nn.Softmax(dim=1),
        )

    def forward(self, state):
        x = self.action_embedding(state)  # [B, S] -> [B, S, F]
        x = self.embedding_dropout(x)

        x = self.seq_encoder(x)

        x = self.head(x)
        return x
