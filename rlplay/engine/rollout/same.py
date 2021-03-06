import torch
import numpy

from plyr import suply

from ..core import prepare, startup, collect


def rollout(envs, actor, n_steps=51, *, sticky=False, device=None):
    """"""
    # always pin the runtime context if the device is 'cuda'
    device = torch.device('cpu') if device is None else device
    pinned = device.type == 'cuda'

    # initialize a buffer for one rollout fragment (optionally pinned)
    buffer = prepare(envs[0], actor, n_steps, len(envs),
                     pinned=pinned, device=device)

    # the running context tor the actor and the envs
    ctx, fragment = startup(envs, actor, buffer, pinned=pinned)
    try:
        while True:
            # collect the fragment
            collect(envs, actor, fragment, ctx, sticky=sticky, device=device)

            # move to the specified device
            batch = fragment.pyt
            if device.type == 'cuda':
                batch = suply(torch.Tensor.to, fragment.pyt,
                              device=device, non_blocking=True)

            yield batch

    finally:
        pass
