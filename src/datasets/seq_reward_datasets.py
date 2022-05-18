from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import pad_truncate_sequences, slice_sequences


class SeqRewardDatasetTrain(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        rewards: Sequence[Sequence[int]],
        max_size: int = 512,
    ):
        self.sequences = slice_sequences(sequences, max_size + 1)
        self.rewards = slice_sequences(rewards, max_size + 1)

        # last_idx = [len(s) - 1 for s in self.sequences]
        count = np.array(
            [
                max(len(s) // (max_size + 1) + (len(s) % max_size >= 4), 1)
                for s in sequences
            ]
        )
        self.cumsum_count = set(np.cumsum(count, axis=0))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        state = seq[:-1]
        next_state = seq[1:]
        reward = self.rewards[index]
        if index in self.cumsum_count:
            done = [0] * (len(seq) - 2) + [1]
        else:
            done = [0] * (len(seq) - 1)

        return (
            state,
            reward,
            next_state,
            done,
        )


class SeqRewardDatasetTest(Dataset):
    def __init__(
        self,
        sequences_tr: Sequence[Sequence[int]],
        sequences_te: Sequence[Sequence[int]],
        rewards_tr: Sequence[Sequence[int]],
        rewards_te: Sequence[Sequence[int]],
        max_size: int = 512,
    ):
        self.sequences_tr = sequences_tr
        self.sequences_te = sequences_te
        self.rewards_tr = rewards_tr
        self.rewards_te = rewards_te
        self.done = torch.zeros(size=(max_size,))
        self.tr_last_ind = torch.tensor(
            [min(len(s), max_size) - 1 for s in sequences_tr]
        )

    def __len__(self) -> int:
        return len(self.sequences_tr)

    def __getitem__(self, index: int):
        state = self.sequences_tr[index]
        tr = self.sequences_tr[index]
        te = self.sequences_te[index]
        next_state = state[1:] + [te[0]]
        reward = self.rewards_tr[index]
        rewards_te = self.rewards_te[index]
        tr_last_ind = self.tr_last_ind[index]

        return state, reward, next_state, self.done, tr, te, rewards_te, tr_last_ind


class SeqRewardTestDatasetCollator:
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

        states, rewards, next_states, dones, trs, tes, rewards_tes, tr_last_ind = zip(
            *batch
        )

        states = torch.tensor(
            pad_truncate_sequences(
                states, max_len=self.max_size, value=self.padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

        rewards = torch.tensor(
            pad_truncate_sequences(
                rewards, max_len=self.max_size, value=0, padding="post"
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
            rewards,
            next_states,
            torch.stack(dones),
        )
        return loss_batch, trs, tes, rewards_tes, torch.stack(tr_last_ind)


class SeqRewardTrainDatasetCollator:
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

        states, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(
            pad_truncate_sequences(
                states, max_len=self.max_size, value=self.padding_idx, padding="post"
            ),
            dtype=torch.long,
        )

        rewards = torch.tensor(
            pad_truncate_sequences(
                rewards, max_len=self.max_size, value=0, padding="post"
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

        dones = torch.tensor(
            pad_truncate_sequences(
                dones,
                max_len=self.max_size,
                value=0,
                padding="post",
            ),
            dtype=torch.long,
        )

        loss_batch = (
            states,
            rewards,
            next_states,
            dones,
        )
        return loss_batch
