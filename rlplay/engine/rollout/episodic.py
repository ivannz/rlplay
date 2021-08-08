import torch
import numpy

from ..core import context, tensor_copy_, Fragment
from ..utils import suply, tuply, getitem, setitem


@torch.no_grad()
def episode(envs, actor, *, device=None):
    """Episodic data generator.

    Parameters
    ----------
    envs : list of gym.Env
        The stateful evaluation environments to step through.

    actor : BaseActorModule
        The actor, which steps through the batch of environments.

    device : torch.device, default=None
        The device onto which to put the input $x_t$ `obs`, $a_{t-1}$ `act`,
        $r_t$ `rew`, $d_t$ `fin`, and $h_t$ `hx` for the actor when stepping
        through the test environments.

    Yields
    ------
    state : State, shape = (1, n_envs, ...)
        The obs-act-rew-fin state with tensor data and the same semantics as
        described in `State`.

    hx : nested object with tensor data
        The present recurrent state of the actor $h_t$, which conditions
        its response along with the `state` input.

    actor : nested object with tensor data, shape = (1, n_envs, ...)
        The extra afterstate information returned by the actor's `.step`.

    next : State, shape = (1, n_envs, ...)
        The next obs-act-rew-fin state with the original true observation $x_t$
        before any reset.

    env : nested object with tensor data, shape = (1, n_envs, ...)
        The extra data received from the batch of environments' `.step` upon
        actually TAKING the actions $a_t$ in them.
    """
    # always pin the runtime context if the device is 'cuda'
    device = torch.device('cpu') if device is None else device
    pinned, on_host = device.type == 'cuda', device.type == 'cpu'

    # prepare a running context for the specified number of envs
    ctx, info_env = context(*envs, pinned=pinned)
    # `ctx` is $x_*, a_{-1}, r_0, \top, h_0$, where `r_0` is undefined

    # fast access to context's aliases
    npy, pyt, info_env_pyt = ctx.npy, ctx.pyt, info_env.pyt

    # Allocate an on-device context and recurrent state, if not on 'host'
    pyt_, info_env_pyt_ = pyt, info_env_pyt
    if not on_host:
        pyt_, info_env_pyt_ = suply(
            torch.Tensor.to, (pyt_, info_env_pyt), device=device)

    # let the actor init `hx` for us: `torch.nn.LSTM` performs two steps: inits
    #  `hx` if it is `None` and then updates it. We undo the second step here.
    _, hx, _ = actor.step(*pyt_, hx=None, virtual=True)
    hx = actor.reset(hx, at=slice(None))

    # collect the evaluation data
    current_ = suply(torch.clone, pyt_)
    while True:
        # REACT: $(t, x_t, a_{t-1}, r_t, d_t, h_t) \to a_t$ and commit $a_t$
        act_, hx_, info_actor = actor.step(*current_, hx=hx, virtual=False)

        # STEP + EMIT: `.step` through a batch of envs
        tensor_copy_(pyt.act, act_)
        for j, env in enumerate(envs):
            # get $(s_t, a_t) \to (s_{t+1}, x_{t+1}, r_{t+1}, d_{t+1})$
            act_ = suply(getitem, npy.act, index=j)
            obs_, rew_, fin_, info_ = env.step(act_)

            npy.stepno[j] += 1
            suply(setitem, info_env.npy, info_, index=j)

            # update the j-th env's '$x_{t+1}, r_{t+1}, d_{t+1}$ in `ctx`
            suply(setitem, npy.obs, obs_, index=j)
            npy.rew[j], npy.fin[j] = rew_, fin_

        # update device-resident copies of `ctx` and `info_env`
        if not on_host:
            tensor_copy_((pyt_, info_env_pyt_), (pyt, info_env_pyt))

        # response: t, state[t], h_t, actor[t], state[t+1], env[t+1]
        yield current_, hx, info_actor, pyt_, info_env_pyt_

        # reset terminated envs (see `context(...)`)
        for j, env in enumerate(envs):
            if npy.fin[j]:
                hx_ = actor.reset(hx_, j)  # h_{t+1} \to h_* at the j-th env

                # s_{t+1} \to s_*, emit x_* from s_*, reset the rest
                suply(setitem, npy.obs, env.reset(), index=j)
                npy.stepno[j], npy.rew[j] = 0, 0.

        # move the final `ctx` to its device-resident copy
        if pyt_ is not pyt:
            tensor_copy_(pyt_, pyt)

        # overwrite the current state for the next iteration
        tensor_copy_(current_, pyt_)
        hx = hx_


def rollout(envs, actor, *, batch_size=1, device=None):
    it = episode(envs, actor, device=device)
    try:
        buffer, queue, base, t = [], [], 0, 0
        beginning = numpy.zeros(len(envs), int)
        while True:
            # yield to the caller if we've prepared a batch
            if len(queue) >= batch_size:
                # TODO a better way to collate differently sized histories
                yield queue[:batch_size]

                queue = queue[batch_size:]

            # make the next step
            cur, hx, act, nxt, env = suply(torch.clone, next(it))
            buffer.append((cur, act, env))
            t += 1

            terminated = nxt.fin.nonzero(as_tuple=True)[-1]
            if len(terminated) < 1:
                continue

            # rebuild terminated trajectories
            for j in map(int, terminated):
                t0, t1 = beginning[j] - base, t - base
                nxt.fin[:, j] = False

                # extract the j-th trajectory and its terminal state
                buf_, terminal, hx_ = suply(
                    getitem, (buffer[t0:t1], nxt, hx),
                    index=(slice(None), [j]))
                state, act_, env_ = zip(*buf_)

                # react to the terminal state
                _, _, info_ = actor.step(*terminal, hx=hx_, virtual=True)

                # stack everything into [T x 1 x ...] tensors
                queue.append(Fragment(
                    tuply(torch.stack, *state, terminal),  # state t=0..T
                    tuply(torch.stack, *act_, info_),      # actor t=0..T
                    tuply(torch.stack, *env_),             # env   t=1..T
                    actor.reset(hx_, at=0),                # hx
                ))

                # make the beginning of the new trajectory (which will be
                #  actually copied from `cur` at the next iteration)
                beginning[j] = t

            # reduce the size of the buffer by slicing at the start of
            #  currently the oldest trajectory
            base_ = beginning.min()
            if base_ > base:
                # we can reduce the size by `base_ - base`
                buffer, base = buffer[base_ - base:], base_

    finally:
        it.close()
