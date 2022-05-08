from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import rolling_window, pad_truncate_sequences


class FixedLengthDatasetTrain(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        window_size: int = 5,
    ):
        count_w = torch.tensor([len(s) for s in sequences]) - window_size
        cumsum_count_w = torch.cumsum(count_w, dim=0)
        done = torch.zeros(count_w.sum(0), dtype=torch.int)
        done[cumsum_count_w - 1] = 1

        self.sequences = [torch.tensor(s) for s in sequences]
        self.done = done
        self.reward = torch.tensor(1)
        self.seq_indexes = torch.cat([torch.repeat_interleave(torch.tensor(i), c) for i, c in enumerate(count_w)], dim=0)
        self.cumsum_count_w = cumsum_count_w
        self.window_size = window_size

    def __len__(self) -> int:
        return self.done.size(0)

    def __getitem__(self, index: int):
        seq_index = self.seq_indexes[index]
        full_seq = self.sequences[seq_index]
        partition_i = index - (self.cumsum_count_w[seq_index] - full_seq.size(0) + self.window_size)

        seq = full_seq[partition_i:partition_i + self.window_size + 1]

        te = full_seq[partition_i + self.window_size:]

        state = seq[:-1]
        next_state = seq[1:]
        action = seq[-1]

        return (
            state,
            action,
            self.reward,
            next_state,
            self.done[index],
            te,
        )


class FixedLengthDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
        window_size: int = 5,
        padding_idx: int = 0,
    ):
        self.states = torch.tensor(
            pad_truncate_sequences(
                sequences_tr, max_len=window_size, value=padding_idx, padding="pre"
            ),
            dtype=torch.long,
        )
        self.tes = [torch.tensor(te) for te in sequences_te]
        self.reward = torch.tensor(1)
        self.done = torch.tensor(0)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        state = self.states[index]
        te = self.tes[index]
        action = te[0]
        next_state = torch.cat((state[1:], action.unsqueeze(0)))

        return state, action, self.reward, next_state, self.done, te


class FixedLengthDatasetCollator:
    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")

        states, actions, rewards, next_states, dones, tes = zip(*batch)

        loss_batch = [torch.stack(tensor) for tensor in [states, actions, rewards, next_states, dones]]
        return loss_batch, tes
