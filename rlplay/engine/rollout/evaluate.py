# single actor double buffered sampler
import sys
import signal

import torch
import numpy
from numpy.random import SeedSequence

from copy import deepcopy
from collections import namedtuple

from ..core import context, tensor_copy_
from ..utils.plyr import suply, setitem, getitem
from ..utils import check_signature

from ..utils.multiprocessing import get_context, CloudpickleSpawner


Control = namedtuple('Control', ['alpha', 'omega', 'error'])
Endpoint = namedtuple('Endpoint', ['rx', 'tx'])


def p_evaluate(
    ss, ctrl, factory, shared, n_envs, n_steps,
    *, clone=True, close=False, device=None
):
    ctrl.omega.tx.close()
    ctrl.omega.rx.close()

    # always pin the runtime context if the device is 'cuda'
    device = torch.device('cpu') if device is None else device
    pinned, on_host = device.type == 'cuda', device.type == 'cpu'

    # disable mutlithreaded computations in the worker processes
    torch.set_num_threads(1)

    # use the reference actor is not on device
    actor = shared
    if not on_host or clone:
        # make an identical local copy
        actor = deepcopy(shared).to(device)

    # spawn a batch of environments
    env_seeds = ss.spawn(n_envs)  # spawn child seeds

    # prepare local envs and the associated local env-state runtime context
    envs = [factory(seed=seed) for seed in env_seeds]

    # prepare an aliased running context for the specified number of envs
    ctx, info_env = context(*envs, pinned=pinned)
    # `ctx` is $x_*, a_{-1}, r_0, \top, h_0$, where `r_0` is undefined

    # fast access to context's aliases
    npy, pyt = ctx.npy, ctx.pyt

    # Allocate on-device context and recurrent state, if device is not None
    pyt_ = pyt
    if not on_host:
        # XXX this also copies data in `pyt` into `pyt_`
        pyt_ = suply(torch.Tensor.to, pyt_, device=device)

    try:
        # unlike `core.evaluate`, this loop only collects the rewards, since
        #  actor's info may quickly exceed the size limit of `pipe.send`, and
        #  tensors in shared memory are not resizable.
        while True:
            # collect the evaluation data: let the actor init `hx` for us
            rewards, done, t, hx = [], False, 0, None
            while not done and t < n_steps:
                # REACT: $(x_t, a_{t-1}, r_t, d_t, h_t) \to a_t$ and commit $a_t$
                act_, hx, info_ = actor.step(*pyt_, hx=hx, virtual=False)
                tensor_copy_(pyt.act, act_)

                # STEP + EMIT: `.step` through a batch of envs
                for j, env in enumerate(envs):
                    # cease interaction with terminated envs
                    if npy.fin[j] and t > 0:
                        npy.rew[j] = 0.
                        continue

                    # get $(s_t, a_t) \to (s_{t+1}, x_{t+1}, r_{t+1}, d_{t+1})$
                    act_ = suply(getitem, npy.act, index=j)
                    obs_, rew_, fin_, info_env = env.step(act_)
                    npy.stepno[j] += 1
                    if fin_:
                        npy.stepno[j] = 0  # start a new trajectory
                        obs_ = env.reset()  # s_{t+1} \to s_*, emit x_* from s_*
                        # XXX unlike `core.collect`. do not reset `hx[j]` here

                    # update the j-th env's '$x_{t+1}, r_{t+1}, d_{t+1}$ in `ctx`
                    suply(setitem, npy.obs, obs_, index=j)
                    npy.rew[j], npy.fin[j] = rew_, fin_

                # move the updated `ctx` to its device-resident torch copy
                if pyt_ is not pyt:
                    tensor_copy_(pyt_, pyt)

                # stop only if all environments have been terminated
                done = numpy.all(npy.fin)

                rewards.append(npy.rew.copy())
                t += 1

            try:
                # block until the request and then immediately send the result
                ctrl.alpha.rx.recv()
                ctrl.alpha.tx.send(sum(rewards))

            # if the request pipe (its write endpoint) is closed, then
            #  this means that the parent process wants us to shut down.
            except EOFError:
                break

            # update parameters from the shared reference actor
            if actor is not shared:
                actor.load_state_dict(shared.state_dict(), strict=True)

    except Exception:
        from traceback import format_exc
        ctrl.error.put(format_exc())
        sys.exit(1)

    finally:
        # let the parent know that something went wrong
        ctrl.alpha.tx.close()
        ctrl.alpha.rx.close()

        # close the environments in states
        if close:
            for env in envs:
                env.close()


def evaluate(
    factory, actor, n_envs, n_steps=None,
    *, clone=True, close=False, device=None,
    start_method=None, entropy=None
):
    check_signature(factory, seed=None)

    n_steps = n_steps or float('+inf')

    # the device to put the batches onto
    device = torch.device('cpu') if device is None else device

    # get the correct multiprocessing context (torch-friendly)
    mp = get_context(start_method)

    # create a host-resident copy of the module in shared memory, which
    #  serves as a vessel for updating the actors in workers
    shared = deepcopy(actor).cpu().share_memory()

    # Create unidirectional (non-duplex) pipes, and connect them in
    # `crossover` mode between us and them (the worker). This set-up
    #  avoids rare occasions where the main process or the worker read
    #  back their own just issued messages.
    ut_rx, ut_tx = mp.Pipe(duplex=False)
    tu_rx, tu_tx = mp.Pipe(duplex=False)

    # The connection end points have the following meanigs:
    # * `ut` and `tu` stand for `us-them` and `them-us`, respectively;
    # * \alpha `ut_tx` (write) and `tu_rx` (read) ends are used by
    #     the main process to issue commands and read back responses.
    # * \omega `ut_rx` (read) and `tu_tx` (write) ends are used by
    #     the worker to receive signals and yield results, respectively;
    ctrl = Control(Endpoint(tu_rx, ut_tx), Endpoint(ut_rx, tu_tx),
                   mp.SimpleQueue())

    # prepare the seed sequence for the wroker
    ss = SeedSequence(entropy)

    # spawn a single worker subprocess
    p_worker = mp.Process(
        target=p_evaluate, daemon=False,
        args=(ss, Control(ctrl.omega, ctrl.alpha, ctrl.error),
              CloudpickleSpawner(factory), shared, n_envs, n_steps),
        kwargs=dict(clone=clone, close=close, device=device),
    )
    p_worker.start()

    # close handles unused by us: them-us tx (write), us-them rx (read)
    ctrl.omega.tx.close()
    ctrl.omega.rx.close()
    # XXX pipes and connections are commonly implemented through file
    # descriptors, which imposes a limit on their max number.

    # the code flow in the loop below and in the `p_double` is designed to
    # synchronize `flipflop` between the worker and the parent.
    emergency = False
    try:
        # instruct the worker that we are awaiting for the rewards
        while p_worker.is_alive():
            # ensure consistent update of the shared module
            shared.load_state_dict(actor.state_dict(), strict=True)

            ctrl.alpha.tx.send(None)  # essentially, this is a barrier
            yield ctrl.alpha.rx.recv()

    except (BrokenPipeError, EOFError):
        # the worker broke the pipes to indicate an emergency shutdown
        emergency = True

    finally:
        # we close our endpoint
        ctrl.alpha.tx.close()
        ctrl.alpha.rx.close()

        p_worker.join()

    if not emergency:
        return

    # handle emergency shutdown
    if not ctrl.error.empty():
        message = ctrl.error.get()

    elif p_worker.exitcode < 0:
        message = signal.Signals(-p_worker.exitcode).name

    else:
        message = f'worker terminated with exit code {p_worker.exitcode}'

    ctrl.error.close()

    raise RuntimeError(message)
