from typing import Sequence
import torch
from torch.utils.data import Dataset
from .utils import pad_roll_sequences, pad_truncate_sequences, roll_sequences


class TransformerDatasetTrain(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        max_size: int = 512,
    ):
        self.sequences = roll_sequences(sequences, max_size + 1)
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        source, target = seq[:-1], seq[1:]

        return source, target


class TransformerDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
        max_size: int = 512,
        padding_idx: int = 0,
    ):
        self.sequences_tr = torch.tensor(
            pad_truncate_sequences(
                sequences_tr, max_len=max_size, value=padding_idx, padding="post"
            ),
            dtype=torch.long,
        )
        self.sequences_te = sequences_te

    def __len__(self) -> int:
        return len(self.sequences_tr)

    def __getitem__(self, index: int):
        source = self.sequences_tr[index]
        te = self.sequences_te[index]

        return source, te


class TransformerDatasetTrainCollator:
    def __init__(
        self,
        max_size: int = 512,
        padding_idx: int = 0,
    ):
        self.padding_idx = padding_idx
        self.max_size = max_size


    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")

        sources, targets = zip(*batch)

        sources = torch.tensor(
            pad_truncate_sequences(
                sources, max_len=self.max_size, value=self.padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

        targets = torch.tensor(
            pad_truncate_sequences(
                targets, max_len=self.max_size, value=self.padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

        return sources, targets


class TransformerDatasetTestCollator:
    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")
        source, te = zip(*batch)

        return torch.stack(source), te
