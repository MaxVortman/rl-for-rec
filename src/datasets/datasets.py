from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import rolling_window, pad_sequences


class FixedLengthDatasetTrain(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        rewards: Sequence[Sequence[int]],
        window_size: int = 5,
    ):
        sequences_fixed = np.concatenate(
            [rolling_window(i, window_size + 1) for i in sequences], 0
        )
        rewards_fixed = np.concatenate(
            [rolling_window(i, window_size + 1) for i in rewards], 0
        )
        sizes = [len(s) for s in sequences]
        sizes_t = torch.tensor(sizes)
        done = torch.zeros(len(sequences_fixed), dtype=torch.int)
        done[torch.cumsum(sizes_t - window_size, dim=0) - 1] = 1
        self.sequences = torch.tensor(sequences_fixed, dtype=torch.long)
        self.rewards = torch.tensor(rewards_fixed, dtype=torch.int)
        self.done = done

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        reward_seq = self.rewards[index]
        state = seq[:-1]
        next_state = seq[1:]
        action = seq[-1]
        reward = reward_seq[-1]

        return state, action, reward, next_state, self.done[index]


class FixedLengthDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
        window_size: int = 5,
        padding_idx: int = 0,
    ):
        self.states = torch.tensor(
            pad_sequences(
                sequences_tr, max_len=window_size, value=padding_idx, padding="pre"
            ),
            dtype=torch.long,
        )
        self.y = torch.tensor(
            pad_sequences(
                sequences_te, max_len=window_size, value=padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return self.states[index], self.y[index]


class FixedLengthDatasetCollator:
    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")

        res_batch = [torch.stack(tensor) for tensor in zip(*batch)]

        return res_batch
