import torch
import numpy

from copy import deepcopy
from torch import multiprocessing
from collections import namedtuple

from .collect import prepare, startup, collect

from ..utils.schema.base import unsafe_apply
from ..utils.schema.shared import Aliased, numpify, torchify


Control = namedtuple('Control', ['reflock', 'empty', 'ready'])


def p_stepper(
    rk, ws, ctrl, factory, buffers, ref,
    *, sticky=False, close=True, device=None
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
    actor = deepcopy(ref).to(device)

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
                actor.load_state_dict(ref.state_dict(), strict=True)
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


class RolloutSampler:
    def __init__(
        self, factory, module, n_steps=20,
        # number of actors and environments each interacts with
        n_actors=8, n_per_actor=2,
        # the size of the rollout buffer pool (must have spare buffers)
        n_buffers=16, n_per_batch=4,
        *,
        sticky=False, pinned=False, close=False, device=None,
        start_method='spawn'
    ):
        """DOC"""
        self.factory, self.close = factory, close
        self.sticky, self.pinned = sticky, pinned
        self.n_actors, self.device = n_actors, device
        self.n_per_batch = n_per_batch
        self.start_method = start_method
        # self.n_steps, self.n_per_actor, self.n_buffers

        # create a host-resident copy of the module in shared memory, which
        #  serves as a vessel for updating the actors in workers
        self.module = deepcopy(module).cpu().share_memory()

        # initialize a reference buffer and make its shared copies
        env = factory()  # XXX seed=None here since used only once

        # create torch tensor buffers for trajectory fragments in the shared
        #  memory using the module copy as a mockup (a single one-element batch
        #  forward pass).
        # XXX torch tensors have much simpler pickling/unpickling when sharing
        #  device = next(self.module.parameters()).device
        ref = prepare(env, self.module, n_steps, n_per_actor, device=None)
        self.buffers = [torchify(ref, shared=True) for _ in range(n_buffers)]

        # some environments don't like being closed or deleted, e.g. `nle`
        if self.close:
            env.close()

        # pre-allocate a nested container for batches in pinned memory
        self.batch_buffer = torchify(
            unsafe_apply(*(ref,) * self.n_per_batch,
                         fn=lambda *x: torch.cat(x, dim=1)),
            pinned=pinned)

    def p_stepper(self, rank):
        # collectors' device may be other than the main device
        return p_stepper(
            rank, self.n_actors, self.ctrl, self.factory, self.buffers,
            self.module, sticky=self.sticky, close=True, device=None,
        )

    def update_from(self, src, *, dst=None):
        # locking makes parameter updates atomic
        # XXX randomly partial updates might inject beneficial stochasticity
        dst = self.module if dst is None else dst
        with self.ctrl.reflock:
            return dst.load_state_dict(src.state_dict(), strict=True)

    def collate_to_(self, out, *tensors):
        """Concatenate tensors along dim=1 into `out` and move to device."""
        # XXX `out` could be a pinned tensor from a special nested container
        torch.cat(tensors, dim=1, out=out)
        return out.to(self.device, non_blocking=True)

    def __iter__(self):
        """DOC"""
        # get the correct multiprocessing context (torch-friendly)
        mp = multiprocessing.get_context(self.start_method)

        # setup buffer index queues, and create a state-dict update lock
        # (more reliable than passing tensors around)
        self.ctrl = Control(mp.Lock(), mp.SimpleQueue(), mp.SimpleQueue())
        for index, _ in enumerate(self.buffers):
            self.ctrl.empty.put(index)

        # spawn worker subprocesses (nprocs is world size)
        p_workers = multiprocessing.start_processes(
            self.p_stepper, daemon=False, join=False,
            nprocs=self.n_actors, start_method=self.start_method,
        )

        # fetch for ready trajectory fragments and collate them into a batch
        try:
            while True:
                # collate ready structured rollout fragments along batch dim=1
                indices = [self.ctrl.ready.get() for _ in range(self.n_per_batch)]

                buffers = (self.buffers[j] for j in indices)
                unsafe_apply(self.batch_buffer, *buffers, fn=self.collate_to_)

                # repurpose buffers for later batches
                for j in indices:
                    self.ctrl.empty.put(j)

                yield self.batch_buffer

        finally:
            # shutdown workers: we don't care about `ready` indices
            # * closing the empty queue doesn't register
            # * can only put None-s
            for _ in range(self.n_actors):
                self.ctrl.empty.put(None)

            p_workers.join()
