# single actor double buffered sampler
import sys
import signal

from threading import BrokenBarrierError

import torch
import numpy
from numpy.random import SeedSequence

from copy import deepcopy
from collections import namedtuple

from plyr import suply

from ..core import prepare, startup, collect

from ..utils.multiprocessing import get_context, CloudpickleSpawner
from ..utils.shared import Aliased, numpify, torchify
from ..utils import check_signature


Control = namedtuple('Control', ['reflock', 'barrier', 'error'])


def p_double(
    ss, ctrl, factory, buffers, shared,
    *, clone=True, sticky=False, close=False, device=None
):
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

    # prepare local envs and the associated local env-state runtime context
    n_envs = buffers[0].state.fin.shape[1]  # `fin` is always T x B

    env_seeds = ss.spawn(n_envs)  # spawn child seeds
    envs = [factory(seed=seed) for seed in env_seeds]

    ctx, fragment0 = startup(envs, actor, buffers[0], pinned=pinned)
    # XXX buffers are interchangeable vessels for data, whereas runtime
    # context is not synchronised to envs and the actor of this worker.

    # prepare aliased fragments, buffer-0 is already aliased
    # XXX `aliased` redundantly traverses `pyt` nested containers,
    #  but `buffers` are allways torch tensors, so we just numpify them.
    fragments = fragment0, Aliased(numpify(buffers[1]), buffers[1])

    # ctrl.barrier synchronizes `flipflop` between the worker and the parent
    flipflop = 0
    try:
        while True:
            # collect rollout with an up-to-date actor into the buffer
            collect(envs, actor, fragments[flipflop], ctx,
                    sticky=sticky, device=device)

            # indicate that the previous fragment is ready
            ctrl.barrier.wait()

            # ensure consistent parameter update from the shared reference
            if actor is not shared:
                with ctrl.reflock:
                    actor.load_state_dict(shared.state_dict(), strict=True)

            # switch to the next fragment buffer
            flipflop = 1 - flipflop

    except BrokenBarrierError:
        # our parent breaks the barrier if it wants us to shutdown
        pass

    except Exception:
        from traceback import format_exc
        ctrl.error.put(format_exc())
        sys.exit(1)

    finally:
        # let the parent know that something
        # went wrong with the current buffer by breaking the barrier.
        ctrl.barrier.abort()

        # close the environments in states
        if close:
            for env in envs:
                env.close()


def rollout(
    factory, actor, n_steps, n_envs,
    *, sticky=False, close=False, clone=True, device=None,
    start_method=None, affinity=None, entropy=None
):
    r"""UPDATE THE DOC

    Details
    -------
    We use double buffering approach with an independent actor per buffer. This
    setup allows for truly parallel collection and learning. The worker (W) and
    the learner (L) move in lockstep: W is collecting rollout into one buffer,
    while L is training on the other buffer. This tandem movement is achieved
    via barrier synchronisation.

        +--------------------+           +--------------------+
        | worker process     |  sharing  | learner process    |     sync
        +--------------------+           +--------------------+
        |                    |           | wait for the first |
        | collect(a_1, b_1)  |           | buffer to be ready |
        |                    |           |  (at the barrier)  |
        +====================+           +====================+  <- barrier
        |                    ---- b_1 -->> train(l, b_1) ---+ |
        | collect(a_2, b_2)  |           |                  | |
        |                    <<-- a_1 ---- update(l, a_1) <-+ |
        +====================+           +====================+  <- barrier
        |                    ---- b_2 -->> train(l, b_2) ---+ |
        | collect(a_1, b_1)  |           |                  | |
        |                    <<-- a_2 ---- update(l, a_2) <-+ |
        +====================+           +====================+  <- barrier
        |                    ---- b_1 -->> train(l, b_1) ---+ |
        | collect(a_2, b_2)  |           |                  | |
        |                    <<-- a_1 ---- update(l, a_1) <-+ |
        +====================+           +====================+  <- barrier
        |    ...      ...    |           |    ...      ...    |

    The dual-actor setup eliminates worker's post-collection idle time, that
    exists in a single-actor setup, which stems from the need of the learner
    to acquire a lock on that single actor's state-dict. Although state-dict
    updates via torch's shared memory storage are fast, so the idle period is
    relatively short. At the same time it takes twice as much memory, since
    we have to maintain TWO instances of the actor model.

    Single-actor setup (assuming non-negligible `train` time).

        +--------------------+           +--------------------+
        | worker process     |  sharing  | learner process    |     sync
        +--------------------+           +--------------------+
        | with a.lock:       |           | wait for the first |
        |   collect(a, b_1)  |           | buffer to be ready |
        |                    |           |  (at the barrier)  |
        +====================+           +====================+  <- barrier
        | with a.lock:       ---- b_1 -->> train(l, b_1) ---+ |
        |   collect(a, b_2)  |           |                  | |
        |                    |           | with a.lock:     | |
        | # idle time        <<--  a  ----   update(l, a) <-+ |
        +====================+           +====================+  <- barrier
        | with a.lock:       ---- b_2 -->> train(l, b_2) ---+ |
        |   collect(a, b_1)  |           |                  | |
        |                    |           | with a.lock:     | |
        | # idle time        <<--  a  ----   update(l, a) <-+ |
        +====================+           +====================+  <- barrier
        | with a.lock:       ---- b_1 -->> train(l, b_1) ---+ |
        |   collect(a, b_2)  |           |                  | |
        |                    |           | with a.lock:     | |
        | # idle time        <<--  a  ----   update(l, a) <-+ |
        +====================+           +====================+  <- barrier
        |    ...      ...    |           |    ...      ...    |

    """
    check_signature(factory, seed=None)

    # the device to put the batches onto
    device = torch.device('cpu') if device is None else device

    # the device, on which to run the worker subprocess
    if isinstance(affinity, (tuple, list)):
        affinity, *empty = affinity
        assert not empty
    affinity = torch.device('cpu') if affinity is None else affinity

    # get the correct multiprocessing context (torch-friendly)
    mp = get_context(start_method)

    # initialize a reference buffer and make its shared copies
    env = factory(seed=None)  # XXX seed=None here since used only once

    # create a host-resident copy of the module in shared memory, which
    #  serves as a vessel for updating the actors in workers
    shared = deepcopy(actor).cpu().share_memory()

    # a single one-element-batch forward pass through the copy
    batch = prepare(env, shared, n_steps, n_envs, pinned=False, device=None)

    # some environments don't like being closed or deleted, e.g. `nle`
    if close:
        env.close()

    # we use double buffered rollout, with both buffers in the shared memory
    # (shared=True always makes a copy).
    # XXX torch tensors have much simpler pickling/unpickling when sharing
    double = torchify((batch,) * 2, shared=True)

    # the sync barrier and actor update lock
    ctrl = Control(mp.Lock(), mp.Barrier(2), mp.SimpleQueue())

    # prepare the seed sequence for the wroker
    ss = SeedSequence(entropy)

    # spawn the lone worker subprocess
    p_worker = mp.Process(
        target=p_double, daemon=False,
        args=(ss, ctrl, CloudpickleSpawner(factory), double, shared),
        kwargs=dict(clone=clone, sticky=sticky, close=close, device=affinity),
    )
    p_worker.start()

    # now move the batch onto the proper device
    if device.type == 'cuda':
        batch = suply(torch.Tensor.to, batch, device=device,
                      non_blocking=True)

    # the code flow in the loop below and in the `p_double` is designed to
    # synchronize `flipflop` between the worker and the parent.
    flipflop, emergency = 0, False
    try:
        while p_worker.is_alive():
            # ensure consistent update of the shared module
            # XXX tau-moving average update?
            with ctrl.reflock:
                shared.load_state_dict(actor.state_dict(), strict=True)

            # wait for the current fragment to be ready (w.r.t `flipflop`)
            ctrl.barrier.wait()

            # yield the filled buffer and switch to the next one
            suply(torch.Tensor.copy_, batch, double[flipflop])

            yield batch
            flipflop = 1 - flipflop

    except BrokenBarrierError:
        # the worker broke the barrier to indicate an emergency shutdown
        emergency = True

    finally:
        # we break the barrier, if we wish to shut down the worker
        ctrl.barrier.abort()

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
