import torch

from torch.utils.data.dataloader import default_collate as torch_collate


class SimpleBuffer:
    """A simple slow ring buffer with no schema which mimics a dataloader."""
    def __init__(self, capacity, *, generator=None):
        self.generator_ = generator or torch.default_generator
        assert isinstance(self.generator_, torch.Generator)

        self.buffer, self.capacity, self.position = [], capacity, 0

    def commit(self, _index=None, **kwdata):
        """Put key-value data into the buffer, evicting the oldest record if full."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(kwdata)  # grow buffer

        else:
            self.buffer[self.position] = kwdata
        self.position = (1 + self.position) % self.capacity

    def __len__(self):
        """The used capacity of the buffer."""
        return len(self.buffer)

    def __iter__(self):
        """iterate over the content."""
        return iter(self.buffer)

    def __getitem__(self, index):
        """Get the item at a particular index in the buffer."""
        return self.buffer[index]

    def __setitem__(self, index, meta):
        """Associate metadata with a key-value data at the given index."""
        # item attribute assignment is ingored by simple buffers
        pass

    def draw(self, batch_size):
        """Draw a random batch from the buffer WITH replacement."""
        # `.randperm` is slow for very large buffer, especially if only
        #  one batch is requested.
        batch = torch.randint(len(self), size=batch_size,
                              generator=self.generator_)
        return self.collate(batch.tolist())  # for-looping over a list fast

    def sample(self, batch_size):
        """Iterate over the buffer in batches drawn WITHOUT replacement."""
        n_batches = (len(self) + batch_size - 1) // batch_size

        permutation = torch.randperm(len(self), generator=self.generator_)
        # permutation[j::n_batches] is `statistically` the same permutation
        for j in range(n_batches):
            # XXX `torch_collate` on the whole buffer gives no speed up
            yield self.collate(permutation[j::n_batches].tolist())

    def collate(self, indices):
        # use torch's versatile, but slow-ish default collate function
        return torch_collate([
            {**self.buffer[j], '_index': j} for j in indices  # back reference
        ])
