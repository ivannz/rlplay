import torch

from torch.utils.data.dataloader import default_collate as torch_collate

from .base import BaseRingBuffer


def _update(op, tree, at, value):
    r"""Update the binary tree up from a changed value at the leaf.

    Details
    -------
    We assume a binary `tree` with $m$ leaves and $
        l := \lfloor \log_2 m \rfloor
    $ levels. The leaves are assigned arbitrary values. The value of each inner
    node, however, is equal to the sum of its left and right child. Hence, at
    an inner level $k$ each node $j = 0, \cdots, 2^k-1$ represents a the sum of
    the slice $
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

    def commit(self, _index=None, _weight=None, **kwdata):
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
