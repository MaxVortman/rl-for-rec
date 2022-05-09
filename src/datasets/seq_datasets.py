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
        max_tr_size: int = 512,
        padding_idx: int = 0,
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
        self.max_tr_size = max_tr_size
        self.padding_idx = padding_idx
        self.count_w = list(count_w)

    def __len__(self) -> int:
        return self.done.size(0)

    def __getitem__(self, index: int):
        seq_index = self.seq_indexes[index]
        full_seq = self.sequences[seq_index]
        partition_i = index - (self.cumsum_count_w[seq_index] - self.count_w[seq_index])

        seq = full_seq[: partition_i + self.min_tr_size + 1]
        seq_tr = seq[-self.max_tr_size - 1 :]

        tr = seq[:-1]
        te = full_seq[partition_i + self.min_tr_size :]

        states_template = torch.full(
            size=(self.max_tr_size + 1,), fill_value=self.padding_idx
        )
        states_template[-len(seq_tr) :] = torch.tensor(seq_tr)

        state = states_template[:-1]
        next_state = states_template[1:]
        action = states_template[-1]

        return (
            state,
            action,
            self.reward,
            next_state,
            self.done[index],
            tr,
            te,
        )


class SeqDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
        padding_idx: int = 0,
        max_tr_size: int = 512,
    ):
        self.sequences_tr_pad = torch.tensor(
            pad_truncate_sequences(
                sequences_tr, max_len=max_tr_size, value=padding_idx, padding="pre"
            ),
            dtype=torch.long,
        )
        self.sequences_tr = sequences_tr
        self.sequences_te = sequences_te
        self.reward = torch.tensor(1)
        self.done = torch.tensor(0)

    def __len__(self) -> int:
        return len(self.sequences_tr)

    def __getitem__(self, index: int):
        state = self.sequences_tr_pad[index]
        tr = self.sequences_tr[index]
        te = self.sequences_te[index]
        action = torch.tensor(te[0])
        next_state = torch.cat((state[1:], action.unsqueeze(0)))

        return state, action, self.reward, next_state, self.done, tr, te


class SeqDatasetCollator:
    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch size should be greater than 0!")

        states, actions, rewards, next_states, dones, trs, tes = zip(*batch)

        loss_batch = (
            torch.stack(states),
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(next_states),
            torch.stack(dones),
        )
        return loss_batch, trs, tes
