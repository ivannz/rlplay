import torch

from torch.utils.data.dataloader import default_collate as torch_collate

from .base import BaseRingBuffer


class SimpleBuffer(BaseRingBuffer):
    """A simple slow ring buffer with no schema which mimics a dataloader."""
    def commit(self, _index=None, **kwdata):
        """Put key-value data into the buffer, evicting the oldest record if full."""
        super().commit(**kwdata)

    def sample_indices(self, batch_size, replacement):
        """Draw random ints indexing the current buffer."""
        if replacement:
            return torch.randint(len(self), generator=self.generator_,
                                 size=(batch_size,))

        # `.randperm` is slow for very large buffer, especially if only
        #  one batch is requested.
        permutation = torch.randperm(len(self), generator=self.generator_)
        return permutation[:batch_size]

    def collate(self, indices):
        # use torch's versatile, but slow-ish default collate function
        return torch_collate([
            {**self.buffer[j], '_index': j} for j in indices  # back reference
        ])
