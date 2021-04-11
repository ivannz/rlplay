import torch

from random import Random
from collections import deque

from torch.utils.data.dataloader import default_collate as torch_collate


class SimpleBuffer:
    """A simple ring buffer with no schema which mimics a dataloader."""
    def __init__(self, n_capacity, n_batch_size=32, *, random=None):
        self.n_batch_size = n_batch_size
        self.random_ = random or Random()
        self.buffer = deque(maxlen=n_capacity)

    def commit(self, **kwdata):
        """Put key-value data into the buffer, evicting the oldest record if full."""
        self.buffer.append(kwdata)

    def __len__(self):
        """The used capacity of the buffer."""
        return len(self.buffer)

    def __getitem__(self, index):
        """Get the item at a particular index in the buffer."""
        return self.buffer[index]

    def __iter__(self):
        """Iterate over random batches of key-value data from the buffer."""
        permutation = self.random_.sample(range(len(self)), len(self))
        for ix in range(0, len(self), self.n_batch_size):
            batch = permutation[ix:ix+self.n_batch_size]

            # use torch's versatile default collate function
            yield torch_collate([
                {**self[j], '_index': j} for j in batch  # add back reference
            ])
