from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import pad_truncate_sequences


class FixedLengthDatasetTrain(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        padding_idx: int = 0,
        te_max_len: int = 128,
        tr_max_len: int = 512,
        test_prop: int = 0.2,
    ):
        tr_list, te_list = list(), list()
        for seq in sequences:
            n_items_u = len(seq)
            proportion = int((1 - test_prop) * n_items_u)

            tr_list.append(seq[:-proportion])
            te_list.append(seq[proportion:])

        self.y = torch.tensor(
            pad_truncate_sequences(
                sequences_te, max_len=te_max_len, value=padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

        sizes = [len(s) for s in sequences]
        sizes_t = torch.tensor(sizes)
        done = torch.zeros(len(sequences_fixed), dtype=torch.int)
        done[torch.cumsum(sizes_t - window_size, dim=0) - 1] = 1
        self.sequences = torch.tensor(sequences_fixed, dtype=torch.long)
        self.done = done

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        state = seq[:-1]
        next_state = seq[1:]
        action = seq[-1]

        return (
            state,
            action,
            torch.tensor(1),
            next_state,
            self.done[index],
            self.y[index],
        )


class FixedLengthDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
        window_size: int = 5,
        padding_idx: int = 0,
        te_max_len: int = 128,
    ):
        self.states = torch.tensor(
            pad_truncate_sequences(
                sequences_tr, max_len=window_size, value=padding_idx, padding="pre"
            ),
            dtype=torch.long,
        )
        self.y = torch.tensor(
            pad_truncate_sequences(
                sequences_te, max_len=te_max_len, value=padding_idx, padding="post"
            ),
            dtype=torch.long,
        )
        self.sequences_te = sequences_te

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        state = self.states[index]
        te = self.y[index]
        action = te[0]
        next_state = torch.cat((state[1:], action.unsqueeze(0)))
        done = torch.tensor(0)

        return state, action, torch.tensor(1), next_state, done, te


class FixedLengthDatasetCollator:
    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")

        res_batch = [torch.stack(tensor) for tensor in zip(*batch)]

        return res_batch
