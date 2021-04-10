import torch

from random import Random
from collections import deque, abc

from torch.utils.data.dataloader import default_collate as torch_collate


def to_device(batch, *, device):
    """Device mover based on `torch_collate`."""
    if isinstance(batch, tuple) and hasattr(batch, '_fields'):
        return type(batch)(*(to_device(value, device=device)
                           for value in batch))

    elif isinstance(batch, abc.Mapping):
        return {key: to_device(value, device=device)
                for key, value in batch.items()}

    elif isinstance(batch, abc.Sequence):
        return [to_device(value, device=device) for value in batch]

    elif isinstance(batch, torch.Tensor):
        return batch.to(device)

    # only torch.Tensors can be moved, everything else is kept as is
    return batch


class SimpleBuffer:
    """A simple buffer with arbitary schema which mimics a dataloader."""
    def __init__(self, n_capacity, n_batch_size=32, *, random=None):
        self.n_batch_size = n_batch_size
        self.random_ = random or Random()
        self.buffer = deque(maxlen=n_capacity)

    def commit(self, **kwdata):
        self.buffer.append(kwdata)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def __iter__(self):
        permutation = self.random_.sample(range(len(self)), len(self))
        for ix in range(0, len(self), self.n_batch_size):
            batch = permutation[ix:ix+self.n_batch_size]
            yield torch_collate([
                {**self[j], '_index': j} for j in batch  # add reference
            ])
