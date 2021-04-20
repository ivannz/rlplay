import torch

from torch.utils.data.dataloader import default_collate as torch_collate

from itertools import count
from heapq import heappush, heappop, heapify

from .base import BaseRingBuffer, _NoValue


def _update(op, tree, at, value):
    r"""Update the binary tree up from a changed value at the leaf.

    Details
    -------
    We assume a binary `tree` with $m$ leaves and $
        l := \lfloor \log_2 m \rfloor
    $ levels. The leaves are assigned arbitrary values. The value of each inner
    node, however, is equal to the sum (`op`) of its left and right child.
    Hence, at an inner level $k$ each node $j = 0, \cdots, 2^k-1$ represents
    the op-aggregate of the slice $
        [j 2^{l-k}, (j + 1) 2^{l-k})
    $ of the leaves, which reside at level $l$.
    """
    size = len(tree) // 2
    at = size + at if at < 0 else at
    assert 0 <= at < size

    at += size
    tree[at] = value
    while at > 1:
        at //= 2
        tree[at] = op(tree[2 * at], tree[2 * at + 1])


def _prefix(tree, value):
    """Find the largest index with all leaves before it aggregating to
    at most the specified value.

    Details
    -------
    This does not work unless the binary tree aggregates using `sum` and
    has exactly a power-of-two number of leaves.
    """

    # XXX the binary tree search logic is buggy for capacities that are not
    #  power of two.
    assert (len(tree) & (len(tree) - 1) == 0)  # power of two or zero

    j, size = 1, len(tree) // 2
    while j < size:
        # value is less than the sum of the left slice: search to the left
        if tree[2*j] >= value:
            j = 2 * j

        # value exceeds the left sum: search to the right for the residual
        else:
            value -= tree[2*j]
            j = 2 * j + 1

    return j - size


def _value(tree, at):
    """Get the leaf value at index."""
    size = len(tree) // 2
    at = size + at if at < 0 else at
    assert 0 <= at < size

    return tree[size + at]


class PriorityBuffer(BaseRingBuffer):
    r"""A prioritized slow ring buffer with no schema.

    Details
    -------
    We replicate https://arxiv.org/abs/1511.05952 here. Each item $j$ in the
    buffer can be drawn with probability $p_j \propto s_j^\alpha$, where $s_j$
    is the priority score $s_j$, assigned when the item was added. The batches,
    sampled from the buffer have additional field `_weight`, each value in
    which is the ratio of the item's probability $p_j$ to the probability
    of the rarest item.
    """
    def __init__(self, capacity, *, generator=None, alpha=1.):
        super().__init__(capacity=capacity, generator=generator)

        self.default, self.alpha = 1., alpha
        self._prio = torch.full((2 * capacity,), self.default, dtype=float)

    @property
    def priority(self):
        offset = len(self._prio) // 2
        return self._prio[offset:offset + len(self)]

    @property
    def min(self):
        return self._prio[1]

    def commit(self, *, _index=_NoValue, _weight=_NoValue, **kwdata):
        """Put key-value data into the buffer, evicting the oldest record if full."""
        _update(min, self._prio, self.position, self.default ** self.alpha)
        super().commit(**kwdata)

    def __setitem__(self, index, value):
        """Set the priority of an index in the buffer."""
        assert value > 0.
        self.default = max(self.default, value)
        _update(min, self._prio, index, value ** self.alpha)

    def sample_indices(self, batch_size, replacement):
        """Draw random indices into the current buffer."""
        return self.priority.multinomial(batch_size, replacement=replacement,
                                         generator=self.generator_)

    def collate(self, indices):
        # Importance weights are $\omega_j = \frac{\pi_j}{\beta_j}$, for
        #  target distribution `\pi` and sampling distribution `\beta`.
        # We assume uniform $\pi_j = \frac1N$ and prioritized $
        #    \beta_j = \frac{p_j^\alpha}{\sum_k p_k^\alpha}
        # $, and return weights normalized by `\max_j \omega_j`. After
        # simplification this yields the formula below.
        buf, batch = self.buffer, []
        for j in indices:
            batch.append({
                **buf[j], '_weight': self.min / self.priority[j], '_index': j
            })

        return torch_collate(batch)


class HeapBuffer(PriorityBuffer):
    r"""A prioritized ring buffer, that evicts data with least priority."""
    def __init__(self, capacity, *, generator=None, alpha=1., factor=5):
        super().__init__(capacity=capacity, generator=generator, alpha=alpha)
        self._id, self._heapq, self._index = count(), [], {}
        self.factor, self.n_garbage = factor, 0

    def _evict(self):
        # evict the element with the least priority, return it position
        while self._heapq:
            _, _, position = heappop(self._heapq)
            if position is not None:
                del self._index[position]  # lose the reference to the item
                break
            self.n_garbage -= 1
        else:
            raise RuntimeError('Buffer exhausted')

        # assert min(self._index, key=lambda i: self._index[i][0]) == position
        return position

    def _upsert(self, position, priority):
        # insert into or update the eviction queue and index
        if position in self._index:
            # mark the element in the eviction queue as stale
            item = self._index.pop(position)
            # `_index` and `_heapq` reference the same mutable data
            item[-1] = None
            self.n_garbage += 1
            _id = item[-2]

        else:
            _id = next(self._id)

        item = [priority, _id, position]  # list is mutable
        self._index[position] = item
        heappush(self._heapq, item)

        # update the min-prio array
        _update(min, self._prio, position, priority)
        return position

    def _compactify(self):
        # remove garbage from the queue and reindex
        _heapq, _index = [], {}
        for prio, _id, pos in self._heapq:
            if pos is not None:
                entry = [prio, _id, pos]
                _heapq.append(entry)
                _index[pos] = entry

        heapify(_heapq)
        self._heapq, self._index = _heapq, _index
        self.n_garbage = 0

    def commit(self, *, _index=_NoValue, _weight=_NoValue, **kwdata):
        """Put key-value data into the buffer, evicting the oldest record if full."""
        if len(self.buffer) < self.capacity:
            # grow buffer
            position = len(self.buffer)
            self.buffer.append(kwdata)

        else:
            position = self._evict()
            self.buffer[position] = kwdata

        self.position = self._upsert(position, self.default ** self.alpha)

        # ops are $O(\log n \rho)$ where $\rho \gg 1$ is the garbage factor.
        # but _evict is, on average, $O(\rho \log \rho n)$ because of the loop
        if self.n_garbage >= self.factor * len(self.buffer):
            self._compactify()

    def __setitem__(self, index, value):
        """Set the priority of an index in the buffer."""
        assert value > 0.
        self.default = max(self.default, value)
        self._upsert(index, value ** self.alpha)


if __name__ == '__main__':
    import tqdm
    import time
    import matplotlib.pyplot as plt

    from random import random

    self = HeapBuffer(capacity=640, alpha=1., factor=100)
    state = count()

    lengths, ticks, n_garbage = [], [], []
    for j in tqdm.tqdm(range(20000)):
        lengths.append(len(self._heapq))

        # add one
        tick = time.monotonic()
        self.commit(data=next(state), hash=random())
        ticks.append(time.monotonic() - tick)
        n_garbage.append(self.n_garbage)

        if len(self) < 32:
            continue

        # update many
        ix = self.sample_indices(32, False).tolist()

        for i in ix:
            self[i] = random()  # self[i]['hash']

    plt.plot(n_garbage)
    plt.plot(lengths)
    plt.twinx().plot(ticks, c='C1')
    plt.show()
