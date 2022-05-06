import numpy as np
from typing import Sequence, Type


def rolling_window(a, window):
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def pad_truncate_sequences(
    sequences: Sequence[Sequence[int]],
    max_len: int,
    value: int = 0,
    padding: str = "post",
    dtype: Type = np.int32,
) -> np.ndarray:
    """Pad sequences with specified value.

    Args:
        sequences (Sequence[Sequence[int]]): sequences to use for padding
        max_len (int): maximum length of sequence to use
        value (int): value to use as padding
        padding (str): type of padding, should be one of ``["pre", "post"]``,
            default is ``"post"``
        dtype (Type): output matrix values type


    Examples:

        >>> sequences = [[1, 2, 3], [4, 5], [6]]
        >>> pad_sequences(sequences, max_len=3, padding="post")
        array([[1, 2, 3],
        [4, 5, 0],
        [6, 0, 0]], dtype=int32)
        >>> pad_sequences(sequences, max_len=3, padding="pre")
        array([[1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]], dtype=int32)

    """

    if not max_len > 0:
        raise ValueError("`max_len` should be greater than 0")

    if padding not in {"pre", "post"}:
        raise ValueError("`padding` should be one of `pre` or `post`")

    features = np.full(shape=(len(sequences), max_len), fill_value=value, dtype=dtype)

    for idx, row in enumerate(sequences):
        if len(row):
            if padding == "pre":
                features[idx, -min(max_len, len(row)) :] = np.array(row)[-max_len:]
            else:
                features[idx, : min(max_len, len(row))] = np.array(row)[:max_len]

    return features
