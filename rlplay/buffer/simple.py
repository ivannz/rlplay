import torch

from random import Random

from torch.utils.data.dataloader import default_collate as torch_collate


class SimpleBuffer:
    """A simple slow ring buffer with no schema which mimics a dataloader."""
    def __init__(self, n_capacity, n_batch_size=32, *, random=None):
        self.random_ = random or Random()
        self.n_capacity, self.n_batch_size = n_capacity, n_batch_size
        self.buffer, self.position = [], 0

    def commit(self, **kwdata):
        """Put key-value data into the buffer, evicting the oldest record if full."""
        if len(self.buffer) < self.n_capacity:
            self.buffer.append(kwdata)  # grow buffer

        else:
            self.buffer[self.position] = kwdata
        self.position = (1 + self.position) % self.n_capacity

    def __len__(self):
        """The used capacity of the buffer."""
        # XXX together with `__iter__`, this confuses tqdm.
        return len(self.buffer)

    def __getitem__(self, index):
        """Get the item at a particular index in the buffer."""
        return self.buffer[index]

    def __setitem__(self, index, value):
        # item attribute assignment is ingored by simple buffers
        pass

    def __iter__(self):
        """Iterate over random batches of key-value data from the buffer."""
        # XXX `Random.sample` is not very fast performing
        permutation = self.random_.sample(range(len(self)), len(self))
        for ix in range(0, len(self), self.n_batch_size):
            batch = permutation[ix:ix+self.n_batch_size]

            # use torch's versatile but slow default collate function
            # XXX moving it out of the loop does not seem to yield speed ups
            yield torch_collate([
                {**self[j], '_index': j} for j in batch  # add back reference
            ])
