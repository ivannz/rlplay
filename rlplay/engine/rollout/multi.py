import torch
import numpy

from copy import deepcopy
from collections import namedtuple

from queue import Empty as QueueEmpty

from ..core import prepare, startup, collect

from ..utils.multiprocessing import get_context, CloudpickleSpawner
from ..utils.multiprocessing import start_processes
from ..utils.plyr import suply, tuply, getitem
from ..utils.shared import Aliased, numpify, torchify


Control = namedtuple('Control', ['reflock', 'empty', 'ready'])


def p_stepper(
    rk, ws, ctrl, factory, buffers, shared,
    clone=True, sticky=False, close=True, affinity=None
):
    r"""Trajectory fragment collection subprocess.

    Details
    -------
    Using queues to pass buffers (even shared) leads to resource leaks. The
    originally considered pattern:
        q2.put(collect(buf=q1.get())),
    -- is against the "passing along" guideline expressed in
    https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors

    Since torch automatically shares tensors when they're reduced to be put into
    a queue or a child precess.
    https://pytorch.org/docs/master/notes/multiprocessing.html#reuse-buffers-passed-through-a-queue

    Closing queues also does not work as a shutdown indicator.
    """
    device = affinity[rk] if affinity is not None else None

    # always pin the runtime context if the device is 'cuda'
    device = torch.device('cpu') if device is None else device
    pinned, on_host = device.type == 'cuda', device.type == 'cpu'

    # disable mutlithreaded computations in the worker processes
    torch.set_num_threads(1)

    # the very first buffer initializes the environment-state context
    ix = ctrl.empty.get()
    if ix is None:
        return

    # use the reference actor is not on device
    actor = shared
    if not on_host or clone:
        # make an identical local copy
        actor = deepcopy(shared).to(device)

    # prepare local envs and the associated local env-state runtime context
    n_envs = buffers[ix].state.fin.shape[1]  # `fin` is always T x B
    envs = [factory() for _ in range(n_envs)]  # XXX seed?
    ctx, fragment = startup(envs, actor, buffers[ix], pinned=pinned)
    # XXX buffers are interchangeable vessels for data, whereas runtime
    # context is not synchronised to envs and the actor of this worker.

    # cache of aliased fragments
    fragments = {ix: fragment}
    try:
        while ix is not None:
            # ensure consistent parameter update from the shared reference
            if actor is not shared:
                with ctrl.reflock:
                    actor.load_state_dict(shared.state_dict(), strict=True)
                    # XXX tau-moving average update?

            # XXX `aliased` redundantly traverses `pyt` nested containers, but
            # `buffers` are allways torch tensors, so we just numpify them.
            if ix not in fragments:
                fragments[ix] = Aliased(numpify(buffers[ix]), buffers[ix])

            # collect rollout with an up-to-date actor into the buffer
            collect(envs, actor, fragments[ix], ctx,
                    sticky=sticky, device=device)

            ctrl.ready.put(ix)

            # fetch the next buffer
            ix = ctrl.empty.get()

    finally:
        if close:
            for env in envs:
                env.close()


def rollout(
    factory,
    actor,
    n_steps,
    # number of actors and environments each interacts with
    n_actors=8,
    n_per_actor=2,
    # the size of the rollout buffer pool (must have spare buffers)
    n_buffers=16,
    n_per_batch=4,
    *, sticky=False, pinned=False, close=False, clone=True, device=None,
    start_method=None, timeout=10, affinity=None
):
    # the device to put the batches onto
    device = torch.device('cpu') if device is None else device

    # the device, on which to run the worker subprocess
    if not isinstance(affinity, (tuple, list)):
        affinity = (affinity,) * n_actors
    assert len(affinity) == n_actors

    # get the correct multiprocessing context (torch-friendly)
    mp = get_context(start_method)

    # initialize a reference buffer and make its shared copies
    env = factory()  # XXX seed=None here since used only once

    # create a host-resident copy of the module in shared memory, which
    #  serves as a vessel for updating the actors in workers
    shared = deepcopy(actor).cpu().share_memory()

    # a single one-element-batch forward pass through the copy
    ref = prepare(env, shared, n_steps, n_per_actor, pinned=False, device=None)

    # some environments don't like being closed or deleted, e.g. `nle`
    if close:
        env.close()

    # pre-allocate a buffer for the collated batch
    #  (non-paged physical memory for faster host-device transfers)
    # XXX `torch.cat` and `.stack` always allocate a new tensor
    stacked = suply(torch.Tensor.to,
                    tuply(torch.stack, *(ref,) * n_per_batch, dim=1),
                    device=device)

    # create slice views along dim=1 and a flattened view into the stacked data
    batch = suply(torch.flatten, stacked, start_dim=1, end_dim=2)
    batch_slice = tuple([suply(getitem, stacked, index=(slice(None), j))
                         for j in range(n_per_batch)])

    # create buffers for trajectory fragments in the shared memory
    # (shared=True always makes a copy).
    # XXX torch tensors have much simpler pickling/unpickling when sharing
    buffers = torchify((ref,) * n_buffers, shared=True)

    del ref

    # setup buffer index queues, and create a state-dict update lock, which
    #  makes `actor ->> shared` updates atomic
    # XXX randomly partial updates might inject beneficial stochasticity
    ctrl = Control(mp.Lock(), mp.Queue(), mp.Queue())
    for index, _ in enumerate(buffers):
        ctrl.empty.put(index)

    # spawn worker subprocesses (nprocs is world size)
    p_workers = start_processes(
        p_stepper, start_method=mp._name, daemon=False, nprocs=n_actors,
        # collectors' device may be other than the main device
        join=False, args=(
            n_actors, ctrl, CloudpickleSpawner(factory), buffers, shared,
            clone, sticky, close, affinity
        ))

    # fetch for ready trajectory fragments and collate them into a batch
    try:
        while True:
            for j in range(n_per_batch):
                ix = ctrl.ready.get(timeout=timeout)

                suply(torch.Tensor.copy_, batch_slice[j],
                      buffers[ix], non_blocking=True)  # XXX has fx if pinned

                ctrl.empty.put(ix)

            yield batch

            # ensure consistent update of the shared module
            # XXX tau-moving average update?
            with ctrl.reflock:
                shared.load_state_dict(actor.state_dict(), strict=True)

    except QueueEmpty:
        pass

    finally:
        # shutdown workers: we don't care about `ready` indices
        # * closing the empty queue doesn't register
        # * can only put None-s
        for _ in range(n_actors):
            ctrl.empty.put(None)

        # `start_processes` loops on `.join` until it returns True or raises
        while not p_workers.join():
            pass

        ctrl.empty.close()
        ctrl.ready.close()
