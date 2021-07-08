import torch
import numpy

from copy import deepcopy
from collections import namedtuple

from queue import Empty as QueueEmpty

from ..core import prepare, startup, collect

from ..utils.multiprocessing import get_context, CloudpickleSpawner
from ..utils.multiprocessing import start_processes
from ..utils.apply import suply, tuply
from ..utils.shared import Aliased, numpify, torchify


Control = namedtuple('Control', ['reflock', 'empty', 'ready'])


def p_stepper(
    rk, ws, ctrl, factory, buffers, shared,
    sticky=False, close=True, device=None
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

    # always pin the runtime context if the device is 'cuda'
    device = torch.device('cpu') if device is None else device
    pinned = device.type == 'cuda'

    # disable mutlithreaded computations in the worker processes
    torch.set_num_threads(1)

    # the very first buffer initializes the environment-state context
    ix = ctrl.empty.get()
    if ix is None:
        return

    # make an identical local copy of the reference actor on cpu
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
            with ctrl.reflock:
                actor.load_state_dict(shared.state_dict(), strict=True)
                # XXX tau-moving average update?

            # XXX `aliased` redundantly traverses `pyt` nested containers, but
            # `buffers` are allways torch tensors, so we just numpify them.
            if ix not in fragments:
                fragments[ix] = Aliased(numpify(buffers[ix]),
                                        buffers[ix])

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
    *, sticky=False, pinned=False, close=False, device=None,
    start_method=None, timeout=10,
):
    # get the correct multiprocessing context (torch-friendly)
    mp = get_context(start_method)

    # initialize a reference buffer and make its shared copies
    env = factory()  # XXX seed=None here since used only once

    # create a host-resident copy of the module in shared memory, which
    #  serves as a vessel for updating the actors in workers
    shared = deepcopy(actor).cpu().share_memory()

    # a single one-element-batch forward pass through the copy
    ref = prepare(env, shared, n_steps, n_per_actor, device=None)

    # some environments don't like being closed or deleted, e.g. `nle`
    if close:
        env.close()

    # pre-allocate a buffer in the pinned memory for collated micro-batches
    #  (non-paged physical memory for faster host-device transfers)
    # XXX `torch.cat` with `out=None` always makes allocates a new tensor.
    batch_buffer = torchify(
        tuply(torch.cat, *(ref,) * n_per_batch, dim=1),  # batch dim!
        pinned=pinned, shared=False)

    # create buffers for trajectory fragments in the shared memory
    # (shared=True always makes a copy).
    # XXX torch tensors have much simpler pickling/unpickling when sharing
    buffers = torchify((ref,) * n_buffers, shared=True)

    # we define a local function for moving data around
    def collate_and_move(out, *tensors, _device=device):
        """Concatenate tensors along dim=1 into `out` and move to device."""
        # XXX `out` could be a pinned tensor from a special nested container
        torch.cat(tensors, dim=1, out=out)
        return out.to(_device, non_blocking=True)

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
            sticky, True, None
        ))

    # fetch for ready trajectory fragments and collate them into a batch
    try:
        while True:
            # collate ready structured rollout fragments along batch dim=1
            indices = [ctrl.ready.get(timeout=timeout)
                       for _ in range(n_per_batch)]

            # we collate the buffers before releasing the buffers
            ready = (buffers[j] for j in indices)
            batch = suply(collate_and_move, batch_buffer, *ready)

            # repurpose buffers for later batches
            for j in indices:
                ctrl.empty.put(j)

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
