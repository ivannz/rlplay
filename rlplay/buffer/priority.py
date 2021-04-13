import torch

from array import array
from operator import add
from random import Random

from torch.utils.data.dataloader import default_collate as torch_collate


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


class PriorityBuffer:
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
    def __init__(self, n_capacity, n_batch_size=32, *, alpha=1., random=None):
        self.random_ = random or Random()
        self.n_capacity, self.n_batch_size = n_capacity, n_batch_size
        self.buffer, self.position = [], 0

        # get the nearest greater power of two for `n_capacity`
        n_capacity = 1 << (n_capacity - 1).bit_length()
        self._sum = array('f', [0.]) * (2 * n_capacity)  # fast sums
        self._min = array('f', [float('+inf')]) * (2 * n_capacity)  # fast mins

        self.default, self.alpha = 1., alpha

    def commit(self, **kwdata):
        """Put key-value data into the buffer, evicting the oldest record if full."""
        _update(add, self._sum, self.position, self.default ** self.alpha)
        _update(min, self._min, self.position, self.default ** self.alpha)
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
        """Set the priority of an index in the buffer."""
        self.default = max(self.default, value)
        _update(add, self._sum, index, value ** self.alpha)
        _update(min, self._min, index, value ** self.alpha)

    def __iter__(self):
        return self

    def __next__(self):
        """Draw a prioritized batch of key-value data from the buffer."""
        indices = [_prefix(self._sum, self.random_.random() * self._sum[1])
                   for _ in range(self.n_batch_size)]

        # Importance weights are $\omega_j = \frac{\pi_j}{\beta_j}$, for
        #  target distributinon `\pi` and sampling distributinon `\beta`
        # We assume uniform $\pi_j = \frac1N$ and prioritized $
        #    \beta_j = \frac{p_j^\alpha}{\sum_k p_k^\alpha}
        # $, and normalize by the maximal weight `\max_j \omega_j`. After
        # simplification this yields the formula below.

        # use torch's versatile but slow default collate function
        # XXX moving it out of the loop does not seem to yield speed ups
        return torch_collate([
            {**self[j], '_index': j,
             '_weight': self._min[1] / _value(self._sum, j)
             } for j in indices
        ])
