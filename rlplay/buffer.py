import torch

from random import Random

from itertools import zip_longest
from collections import deque, abc

from torch.utils.data.dataloader import default_collate as torch_collate


def ensure_schema(batch, *, schema=None):
    """Ensure the specified device-dtype schema on a structured container.

    Details
    -------
    Replicates the logic of `torch_collate` in traversing structured batches:
        `/torch/utils/data/_utils/collate.py#L43-L86`
        as of commit `b80c6f863f2327c712c478f67c248b94d66b65ac` (2021-04-11)

    Operates
    """
    # protect against exhausted batch Sequence in `zip_longest`
    if batch is None:
        raise RuntimeError

    # undefined schema means keeping `source` specs
    if schema is None:
        return batch

    # if batch is a `torch.Tensor` then we're in the base case
    if isinstance(batch, torch.Tensor):
        if isinstance(schema, torch.dtype):
            return batch.to(dtype=schema)

        elif not isinstance(schema, dict):
            raise TypeError(f'`schema` for a torch.Tensor '
                            f'must be a `dict`. Got `{schema}`.')

        # if the specs match the tensor, then no copy is made
        return batch.to(**schema)

    # make sure the batch element adheres to the schema
    assert isinstance(batch, type(schema))

    if isinstance(batch, abc.Sequence):  # (tuple, namedtuple, list, etc)
        try:
            pairs = zip_longest(batch, schema, fillvalue=None)
            data = [ensure_schema(v, schema=s) for v, s in pairs]

        except RuntimeError:
            raise ValueError(
                f'Incompatible schema `{schema}` for `{batch}`') from None

        # namedtuples, unlike tuples, use positional args on `__init__`
        if isinstance(batch, tuple) and hasattr(batch, '_fields'):
            return type(batch)(*data)
            # their `._make(data)` method can be used to create new instances
            # assert callable(getattr(batch, '_make', None))
            # return batch._make(data)

        return type(batch)(data)

    elif isinstance(batch, abc.Mapping):  # (dict, OrderedDict, etc)
        # a `dict` schema is allowed to have missing keys
        return type(batch)({
            k: ensure_schema(v, schema=schema.get(k)) for k, v in batch.items()
        })

    # schema does not apply to anything else: keep it as is
    #  Since we are called **after** `torch_collate`, all atypical cases
    #  must have already been handled there.
    return batch


def to_device(batch, *, device):
    """Device mover based on `torch_collate`."""
    if device is None:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, abc.Sequence):
        data = [to_device(v, device=device) for v in batch]

        if isinstance(batch, tuple) and hasattr(batch, '_fields'):
            return type(batch)(*data)

        return type(batch)(data)

    elif isinstance(batch, abc.Mapping):
        return type(batch)({
            k: to_device(v, device=device) for k, v in batch.items()
        })

    # only `torch.Tensor` can be moved, everything else is kept as is
    return batch


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
