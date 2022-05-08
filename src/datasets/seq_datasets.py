from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import pad_truncate_sequences


class SeqDatasetTrain(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        min_tr_size: int = 4,
    ):
        count_w = np.array([len(s) for s in sequences]) - min_tr_size
        cumsum_count_w = np.cumsum(count_w, axis=0)
        done = torch.zeros(count_w.sum(0), dtype=torch.int)
        done[cumsum_count_w - 1] = 1

        self.sequences = sequences
        self.done = done
        self.reward = torch.tensor(1)
        self.seq_indexes = list(
            np.concatenate([np.repeat(i, c) for i, c in enumerate(count_w)], axis=0)
        )
        self.cumsum_count_w = list(cumsum_count_w)
        self.min_tr_size = min_tr_size
        self.count_w = list(count_w)

    def __len__(self) -> int:
        return self.done.size(0)

    def __getitem__(self, index: int):
        seq_index = self.seq_indexes[index]
        full_seq = self.sequences[seq_index]
        partition_i = index - (self.cumsum_count_w[seq_index] - self.count_w[seq_index])

        seq = full_seq[: partition_i + self.min_tr_size + 1]

        te = full_seq[partition_i + self.min_tr_size :]

        state = seq[:-1]
        next_state = seq
        action = seq[-1]

        return (
            state,
            torch.tensor(action),
            self.reward,
            next_state,
            self.done[index],
            te,
        )


class SeqDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
    ):
        self.sequences_tr = sequences_tr
        self.sequences_te = sequences_te
        self.reward = torch.tensor(1)
        self.done = torch.tensor(0)

    def __len__(self) -> int:
        return len(self.sequences_tr)

    def __getitem__(self, index: int):
        state = self.sequences_tr[index]
        te = self.sequences_te[index]
        action = te[0]
        next_state = state + [action]

        return state, torch.tensor(action), self.reward, next_state, self.done, te


class SeqDatasetCollator:
    def __init__(
        self,
        padding_idx: int = 0,
        max_tr_size: int = 512,
    ):
        self.max_tr_size = max_tr_size
        self.padding_idx = padding_idx

    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")

        states, actions, rewards, next_states, dones, tes = zip(*batch)

        states_t = torch.tensor(
            pad_truncate_sequences(
                states, max_len=self.max_tr_size, value=self.padding_idx, padding="pre"
            ),
            dtype=torch.long,
        )

        next_states_t = torch.tensor(
            pad_truncate_sequences(
                next_states,
                max_len=self.max_tr_size,
                value=self.padding_idx,
                padding="pre",
            ),
            dtype=torch.long,
        )

        loss_batch = (
            states_t,
            torch.stack(actions),
            torch.stack(rewards),
            next_states_t,
            torch.stack(dones),
        )
        return loss_batch, tes
