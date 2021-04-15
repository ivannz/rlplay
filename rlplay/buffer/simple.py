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

    def sample_indices(self, batch_size, replacement):
        """Draw random indices into the current buffer."""
        if replacement:
            return torch.randint(len(self), generator=self.generator_,
                                 size=(batch_size,))

        # `.randperm` is slow for very large buffer, especially if only
        #  one batch is requested.
        permutation = torch.randperm(len(self), generator=self.generator_)
        return permutation[:batch_size]

    def draw(self, batch_size, replacement=True):
        """Draw a random batch from the buffer WITH replacement."""
        batch = self.sample_indices(batch_size, replacement)
        return self.collate(batch.tolist())  # for-looping over a list fast

    def sample(self, batch_size, replacement=False):
        """Iterate over the buffer in batches drawn WITHOUT replacement."""
        sample = self.sample_indices(len(self), replacement)
        for j in range(0, len(self), batch_size):
            # XXX `torch_collate` on the whole buffer gives no speed up
            yield self.collate(sample[j:j + batch_size].tolist())

    def collate(self, indices):
        # use torch's versatile, but slow-ish default collate function
        return torch_collate([
            {**self.buffer[j], '_index': j} for j in indices  # back reference
        ])
