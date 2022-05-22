import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from .utils import EmbeddingDropout


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(
            ntoken + 1, d_model, padding_idx=padding_idx
        )  # + padding_idx

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [sequence length, batch_size, dim_model]
        """

        x = self.embedding(src) * math.sqrt(self.d_model)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model)
        x = x.permute(1, 0, 2)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # (sequence length, batch_size, dim_model)

        return x


class TransformerEmbeddingFreeze(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super(TransformerEmbeddingFreeze, self).__init__()
        self.training = False
        self.d_model = d_model

        self.embedding = nn.Embedding(
            ntoken + 1, d_model, padding_idx=padding_idx
        )  # + padding_idx

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        for module in self.children():
            module.eval()

    def train(self, mode):
        return self

    def forward(
        self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [sequence length, batch_size, dim_model]
        """

        x = self.embedding(src) * math.sqrt(self.d_model)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model)
        x = x.permute(1, 0, 2)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(
            x, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # (sequence length, batch_size, dim_model)

        return x


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"

        self.transformer_embedding = TransformerEmbedding(
            ntoken=ntoken,
            d_model=d_model,
            nhead=nhead,
            d_hid=d_hid,
            nlayers=nlayers,
            dropout=dropout,
            padding_idx=padding_idx,
        )

        self.head = nn.Linear(d_model, ntoken + 1)  # + padding_idx

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, num_tokens, sequence length]
        """

        x = self.transformer_embedding(
            src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )  # (sequence length, batch_size, dim_model)
        x = self.head(x)  # (sequence length, batch_size, num_tokens)

        # Permute to have batch size first again
        x = x.permute(1, 2, 0)  # (batch_size, num_tokens, sequence length)

        return x


class DqnFreezeTransformer(nn.Module):
    def __init__(
        self,
        transformer_embedding: TransformerEmbedding,
        ntoken: int,
        d_model: int = 512,
        dropout_rate: float = 0.1
    ):
        super(DqnFreezeTransformer, self).__init__()

        self.transformer_embedding = transformer_embedding

        self.embedding_dropout = EmbeddingDropout(dropout_rate)

        self.linears = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * d_model, d_model),
        )
        self.linears_lnorm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, ntoken + 1)  # + padding_idx

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, num_tokens, sequence length]
        """

        with torch.no_grad():
            x = self.transformer_embedding(
                src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )  # (sequence length, batch_size, dim_model)

        x = x.permute(1, 0, 2)  # (batch_size, sequence length, dim_model)
        x = self.embedding_dropout(x)

        x = x + self.linears_lnorm(self.linears(x))

        x = self.head(x)  # (batch_size, sequence length, num_tokens)

        x = x.permute(0, 2, 1)  # (batch_size, num_tokens, sequence length)

        return x


class DqnTransformerEmbedding(nn.Module):
    def __init__(
        self,
        transformer_embedding: TransformerEmbedding,
        ntoken: int,
        d_model: int = 512,
        dropout_rate: float = 0.1
    ):
        super(DqnTransformerEmbedding, self).__init__()

        self.transformer_embedding = transformer_embedding

        # self.embedding_dropout = EmbeddingDropout(dropout_rate)

        # self.linears = nn.Sequential(
        #     nn.Linear(d_model, 4 * d_model),
        #     nn.GELU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(4 * d_model, d_model),
        # )
        # self.linears_lnorm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, ntoken + 1)  # + padding_idx

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [batch_size, num_tokens, sequence length]
        """
        x = self.transformer_embedding(
                src=src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask
            )  # (sequence length, batch_size, dim_model)

        # x = x.permute(1, 0, 2)  # (batch_size, sequence length, dim_model)
        # x = self.embedding_dropout(x)

        # x = x + self.linears_lnorm(self.linears(x))

        # x = self.head(x)  # (batch_size, sequence length, num_tokens)
        x = self.head(x)  # (sequence length, batch_size, num_tokens)

        # x = x.permute(0, 2, 1)  # (batch_size, num_tokens, sequence length)
        x = x.permute(1, 2, 0)  # (batch_size, num_tokens, sequence length)

        return x


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


def create_pad_mask(matrix: Tensor, pad_token: int) -> Tensor:
    # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
    # [False, False, False, True, True, True]
    return matrix == pad_token
