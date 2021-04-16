import os
import time

import torch


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
        return filename

    head, tail = os.path.split(filename)
    dttm = time.strftime('%Y%m%d-%H%M%S')
    os.rename(filename, os.path.join(head, f'{prefix}__{dttm}__{tail}'))

    return filename
