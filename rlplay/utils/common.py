import os
import time

import torch

from functools import wraps


def batchify(fn, at):
    # at is the first dim after the batch
    @wraps(fn)
    def _decorator(main, *tensors, **kwargs):
        ld = at + main.ndim if at < 0 else at

        # *leading, feature@ld, *spatial
        output = fn(main.flatten(0, ld - 1), *tensors, **kwargs)
        return output.reshape(*main.shape[:ld], *output.shape[1:])

    return _decorator


@wraps(torch.multinomial)
def multinomial(
    input, num_samples=1, replacement=False, *, generator=None, out=None
):
    out = torch.multinomial(input.flatten(0, -2), num_samples, out=out,
                            replacement=replacement, generator=generator)
    if num_samples > 1:
        return out.reshape(input.shape[:-1] + (num_samples,))
    return out.reshape(input.shape[:-1])


def linear(t, t0=0, t1=100, v0=1., v1=0.):
    tau = min(1., max(0., (t1 - t) / (t1 - t0)))
    return v0 * tau + v1 * (1 - tau)


@torch.no_grad()
def greedy(q_value, *, epsilon=0.5):
    """epsilon-greedy strategy."""
    q_action = q_value.max(dim=-1).indices

    random = torch.randint(q_value.shape[-1], size=q_action.shape,
                           device=q_value.device)
    is_random = torch.rand_like(q_value[..., 0]).lt(epsilon)
    return torch.where(is_random, random, q_action)


def backupifexists(filename, prefix='backup'):
    if not os.path.isfile(filename):
        return None

    head, tail = os.path.split(filename)
    dttm = time.strftime('%Y%m%d-%H%M%S')

    backup = os.path.join(head, f'{prefix}__{dttm}__{tail}')
    os.rename(filename, backup)

    return backup
