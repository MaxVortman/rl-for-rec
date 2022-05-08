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

        self.sequences = sequences
        self.done = done
        self.seq_indexes = torch.cat([torch.repeat_interleave(torch.tensor(i), c) for i, c in enumerate(count_w)], dim=0)
        self.cumsum_count_w = cumsum_count_w
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq_index = self.seq_indexes[index]
        full_seq = self.sequences[seq_index]
        # partition_i = self.cumsum_count_w[seq_index] - index - 1
        partition_i = index - (self.cumsum_count_w[seq_index] - len(full_seq) + self.window_size)

        # seq = full_seq[-self.window_size - 1 - partition_i: -partition_i]

        # te = full_seq[-partition_i - 1:]

        seq = full_seq[partition_i:partition_i + self.window_size + 1]

        te = full_seq[partition_i + self.window_size:]

        state = seq[:-1]
        next_state = seq[1:]
        action = seq[-1]
        
        return (
            torch.tensor(state),
            torch.tensor(action),
            torch.tensor(1),
            torch.tensor(next_state),
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
        self.tes = sequences_te

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        state = self.states[index]
        te = self.tes[index]
        action = torch.tensor(te[0])
        next_state = torch.cat((state[1:], action.unsqueeze(0)))
        done = torch.tensor(0)

        return state, action, torch.tensor(1), next_state, done, te


class FixedLengthDatasetCollator:
    def __init__(
        self,
        items_n: int,
    ):
        self.items_n = items_n

    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")

        states, actions, rewards, next_states, dones, tes = zip(*batch)

        true_matrix = torch.zeros(
            size=(len(batch), self.items_n + 1) # + padding index
        )

        for i, te in enumerate(tes):
            true_matrix[i, te] = 1

        res_batch = [torch.stack(tensor) for tensor in [states, actions, rewards, next_states, dones]]
        res_batch.append(true_matrix)
        return res_batch
