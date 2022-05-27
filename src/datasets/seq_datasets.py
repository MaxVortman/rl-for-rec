from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import pad_truncate_sequences


class SeqDatasetTrain(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        rewards: Sequence[Sequence[int]],
        min_tr_size: int = 4,
        max_tr_size: int = 512,
    ):
        count_w = np.array([len(s) for s in sequences]) - min_tr_size
        cumsum_count_w = np.cumsum(count_w, axis=0)
        done = torch.zeros(count_w.sum(0), dtype=torch.int)
        done[cumsum_count_w - 1] = 1

        self.sequences = np.array(sequences, dtype=object)
        self.rewards = np.array(rewards, dtype=object)
        self.done = done
        self.seq_indexes = np.concatenate([np.repeat(i, c) for i, c in enumerate(count_w)], axis=0)
        self.cumsum_count_w = cumsum_count_w
        self.min_tr_size = min_tr_size
        self.max_tr_size = max_tr_size
        self.count_w = count_w

    def __len__(self) -> int:
        return self.done.size(0)

    def __getitem__(self, index: int):
        seq_index = self.seq_indexes[index]
        full_seq = self.sequences[seq_index]
        partition_i = index - (self.cumsum_count_w[seq_index] - self.count_w[seq_index])

        seq = full_seq[: partition_i + self.min_tr_size + 1]
        seq_tr = seq[-self.max_tr_size - 1 :]

        state = seq_tr[:-1]
        next_state = seq_tr[1:]
        action = seq_tr[-1]

        reward = self.rewards[seq_index][partition_i + self.min_tr_size]
    
        return (
            state,
            action,
            reward,
            next_state,
            self.done[index],
        )


class SeqDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
        rewards_te: Sequence[Sequence[int]],
    ):
        self.sequences_tr = sequences_tr
        self.sequences_te = sequences_te
        self.rewards_te = rewards_te
        self.done = torch.tensor(0)

    def __len__(self) -> int:
        return len(self.sequences_tr)

    def __getitem__(self, index: int):
        state = self.sequences_tr[index]
        tr = self.sequences_tr[index]
        te = self.sequences_te[index]
        action = te[0]
        next_state = state + [action]
        reward_te = self.rewards_te[index]
        reward = reward_te[0]

        return (
            state,
            action,
            reward,
            next_state,
            self.done,
            tr,
            te,
            reward_te,
        )


class SeqDatasetTrainCollator:
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

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            pad_truncate_sequences(
                states, max_len=self.max_size, value=self.padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

        next_states = torch.tensor(
            pad_truncate_sequences(
                next_states,
                max_len=self.max_size,
                value=self.padding_idx,
                padding="post",
            ),
            dtype=torch.long,
        )

        loss_batch = (
            states,
            torch.tensor(actions),
            torch.tensor(rewards),
            next_states,
            torch.stack(dones),
        )
        return loss_batch


class SeqDatasetTestCollator:
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

        states, actions, rewards, next_states, dones, trs, tes, reward_tes = zip(*batch)

        states = torch.tensor(
            pad_truncate_sequences(
                states, max_len=self.max_size, value=self.padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

        next_states = torch.tensor(
            pad_truncate_sequences(
                next_states,
                max_len=self.max_size,
                value=self.padding_idx,
                padding="post",
            ),
            dtype=torch.long,
        )

        loss_batch = (
            states,
            torch.tensor(actions),
            torch.tensor(rewards),
            next_states,
            torch.stack(dones),
        )
        return loss_batch, trs, tes, reward_tes
