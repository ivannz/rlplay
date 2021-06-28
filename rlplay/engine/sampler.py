import torch
import numpy
import time
import gym
# import nle

from torch import multiprocessing

from rlplay.engine.mp import RolloutSampler

from rlplay.engine.collect import BaseActorModule
from rlplay.engine.collect import prepare, startup, collect
from rlplay.utils.schema.base import unsafe_apply
from rlplay.utils.schema.shared import Aliased, aliased
from rlplay.utils.schema.shared import numpify, torchify

from rlplay.engine.returns import np_compute_returns
from rlplay.utils.common import multinomial
from rlplay.engine.vec import CloudpickleSpawner

from copy import deepcopy


class protoRolloutSampler:
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

        # create trajectory fragment buffers of torch tensors in the shared
        #  memory using the learner as a mockup (a single one-element batch
        #  forward pass).
        # XXX torch tensors have much simpler pickling/unpickling when sharing
        #  device = next(self.module.parameters()).device
        ref = prepare(env, self.module, n_steps, n_per_actor, device=None)
        self.buffers = [torchify(ref, shared=True) for _ in range(n_buffers)]

        # some environments don't like being closed or deleted, e.g. `nle`
        if self.close:
            env.close()

    def update_from(self, src, *, dst=None):
        # locking makes updates consistent, but it might be possible that
        #  stochastic incomplete partial updates inject extra stochastcity
        dst = self.module if dst is None else dst
        with self._update_lock:
            return dst.load_state_dict(src.state_dict(), strict=True)

    def _collate(self, *tensors, dim=1):
        """concatenate along dim=1"""
        return torch.cat(tensors, dim).to(self.device, non_blocking=True)

    def __iter__(self):
        """DOC"""
        # get the correct multiprocessing context (torch-friendly)
        mp = multiprocessing.get_context(self.start_method)

        # setup buffer index queues (more reliable than passing tensors around)
        self.q_empty, self.q_ready = mp.SimpleQueue(), mp.SimpleQueue()
        for index, _ in enumerate(self.buffers):
            self.q_empty.put(index)

        # create a state-dict update lock
        self._update_lock = mp.Lock()

        # spawn worker subprocesses (nprocs is world size)
        p_workers = multiprocessing.start_processes(
            self.collect, nprocs=self.n_actors, daemon=False, join=False,
            # collectors' device may be other than the main device
            args=(self.n_actors, self.sticky, False, None),  # pinned, device
            start_method=self.start_method
        )
        try:
            while True:
                # collate ready structured rollout fragments along batch dim=1
                indices = [self.q_ready.get() for _ in range(self.n_per_batch)]
                batch = unsafe_apply(*(self.buffers[j] for j in indices),
                                     fn=self._collate)

                # repurpose buffers for later batches
                for j in indices:
                    self.q_empty.put(j)

                yield batch

        finally:
            # shutdown workers: we don't care about `ready` indices
            # * closing the empty queue doesn't work register
            # * can only put None-s
            for _ in range(self.n_actors):
                self.q_empty.put(None)

            p_workers.join()

    def collect(self, rk, ws, sticky=False, pinned=False, device=None):
        r"""
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

        # disable mutlithreaded computations in the worker processes
        torch.set_num_threads(1)

        # the very first buffer initializes the environment-state context
        ix = self.q_empty.get()
        if ix is None:
            return

        # make an identical local copy of the reference actor on cpu
        actor = deepcopy(self.module).to(device)

        # prepare local envs and the associated local env-state context
        n_envs = self.buffers[ix].state.fin.shape[1]  # `fin` is always T x B
        envs = [self.factory() for _ in range(n_envs)]  # XXX seed?
        ctx, fragment = startup(envs, actor, self.buffers[ix], pinned=pinned)

        # cache of aliased fragments
        fragments = {ix: fragment}
        try:
            while ix is not None:
                # XXX `aliased` redundantly traverses `pyt` nested containers,
                # so we just numpify it, since buffers are ALL torch tensors.
                if ix not in fragments:
                    fragments[ix] = Aliased(numpify(self.buffers[ix]),
                                            self.buffers[ix])

                # collect rollout with an up-to-date actor into the buffer
                # XXX buffers are interchangeable, ctx is not
                self.update_from(src=self.module, dst=actor)
                collect(envs, actor, fragments[ix], ctx,
                        sticky=sticky, device=device)

                self.q_ready.put(ix)

                # fetch the next buffer
                ix = self.q_empty.get()

        finally:
            if self.close:
                for env in envs:
                    env.close()