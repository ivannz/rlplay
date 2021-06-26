import torch
import numpy
import time
import gym
# import nle

import torch.multiprocessing as mp
# from collections import namedtuple

from rlplay.engine.collect import prepare, startup, collect
from rlplay.engine.collect import BaseActorModule
from rlplay.engine.returns import np_compute_returns

from rlplay.utils.schema.base import unsafe_apply
from rlplay.utils.schema.shared import aliased, torchify
from rlplay.utils.schema.shared import Aliased, numpify
from rlplay.utils.common import multinomial


from rlplay.engine.vec import CloudpickleSpawner
from copy import deepcopy

# Endpoint = namedtuple('Endpoint', ['rx', 'tx'])


class Actor(BaseActorModule):
    # We should not store references to instantiated env,
    # since it is about to be communicated to child processes
    def __init__(self, observation, action):
        super().__init__()

        # XXX for now we just store the spaces
        self.observation_space, self.action_space = observation, action

        n_h_dim = 8
        self.emb_obs = torch.nn.Embedding(observation.n, n_h_dim)
        self.core = torch.nn.GRU(n_h_dim, n_h_dim, 1)
        self.policy = torch.nn.Linear(n_h_dim, action.n)
        self.baseline = torch.nn.Linear(n_h_dim, 1)
        self.register_buffer('_counter', torch.zeros(1))

    def forward(self, obs, act=None, rew=None, fin=None, *, hx=None):
        # Everything is  [T x B x ...]
        n_steps, n_env, *_ = obs.shape
        input = self.emb_obs(obs)

        # `fin` indicates which of the `act` and `rew` are invalid
        out, hx = self.core(input, hx)
        logits = self.policy(out).log_softmax(dim=-1)
        value = self.baseline(out)[..., 0]

        if self.training:
            actions = multinomial(logits.exp())
        else:
            actions = logits.argmax(dim=-1)

        # rnn states, value estimates, and other info
        return actions, hx, dict(value=value, logits=logits)


class SimpleActor(BaseActorModule):
    def __init__(self, observation, action):
        super().__init__()

        # XXX for now we just store the spaces
        self.observation_space, self.action_space = observation, action
        self.emb_obs = torch.nn.Embedding(observation.n, 1 + action.n)

        self.register_buffer('_counter', torch.zeros(1))

    def forward(self, obs, act=None, rew=None, fin=None, *, hx=None):
        # Everything is  [T x B x ...]
        n_steps, n_env, *_ = obs.shape
        out = self.emb_obs(obs)

        policy, value = out[..., 1:], out[..., 0]
        logits = policy.log_softmax(dim=-1)

        if self.training:
            actions = multinomial(logits.exp())
        else:
            actions = logits.argmax(dim=-1)

        # rnn states, value estimates, and other info
        return actions, (), dict(value=value, logits=logits)


class RandomActor(BaseActorModule):
    # We should not store references to instantiated env,
    # since it is about to be communicated to child processes
    def __init__(self, observation, action):
        super().__init__()

        # XXX for now we just store the spaces
        self.observation_space, self.action_space = observation, action

        self.register_buffer('_counter', torch.zeros(1))

    def forward(self, obs, act=None, rew=None, fin=None, *, hx=None):
        # Everything is [T x B x ...]
        n_steps, n_envs, *_ = fin.shape

        # `fin` indicates which of the `act` and `rew` are invalid
        logits = torch.randn(n_steps, n_envs).log_softmax(dim=-1)
        value = torch.randn(n_steps, n_envs).fill_(float(self._counter))

        actions = torch.tensor([self.action_space.sample()
                                for _ in range(n_envs)]*n_steps)

        # rnn states, value estimates, and other info
        return actions, (), dict(value=value, logits=logits)


def worker_queue(
    i, q_empty, q_ready,
    factory, reference, buffers,
    sticky=False, pin_memory=False
):
    r"""
    Details
    -------
    Using queues to pass buffers (even shared) leads to resource leaks. The
    originally considered pattern:
        q2.put(collect(buf=q1.get())),
    -- is against the "passing along" guideline expressed in
    https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
    https://pytorch.org/docs/master/notes/multiprocessing.html#reuse-buffers-passed-through-a-queue
    Since torch automatically shares tensors when they're reduce to be put into
    a queue or a child precess.

    Closing queues also does not work as a shutdown indicator.
    """
    # disable mutlithreaded computations in the child processes
    import torch
    torch.set_num_threads(1)

    # make an identical local copy of the reference actor
    actor = deepcopy(reference).cpu()  # run on cpu

    # the first buffer is used for initializing the environment-state context
    index = q_empty.get()
    if index is None:
        return True

    # prepare our own envs and the running state associated with them
    n_steps, n_envs = buffers[index].state.fin.shape  # T x B bool tensor
    envs = [factory() for _ in range(n_envs)]
    state, fragment = startup(envs, actor, buffers[index], pinned=pin_memory)
    while True:
        # update the local actor's parameters from the reference module
        actor.load_state_dict(reference.state_dict())

        # collect and send back
        collect(envs, actor, fragment, state, sticky=sticky)
        q_ready.put(index)

        # get the next buffer or the termination signal
        index = q_empty.get()
        if index is None:
            break

        # XXX `aliased` makes a redundant traversal of `pyt` nested container
        fragment = Aliased(numpify(buffers[index]), buffers[index])

    # close the environments in states
    for env in envs:
        pass  # env.close()

    return True


def collector_queue(
    factory, n_steps, n_envs, n_actors=8, n_buffers=20, n_buffers_per_batch=4, *, device=None
):
    # we need to set more buffers than workers
    env = factory()
    sticky, pin_memory = False, False

    # initialize queues buffers
    q_empty, q_ready, buffers = mp.Queue(), mp.Queue(), []

    # create trajectory fragment buffers of torch tensors in the shared memory
    #  using the learner as a mockup (a single one-element batch forward pass).
    # XXX torch tensors have much simpler pickling/unpickling when sharing
    for index in range(n_buffers):
        buffers.append(prepare(env, learner, n_steps, n_envs,
                               shared=True, device=device))
        q_empty.put(index)

    # env.close()  # ex. `nle` does not like being closed or deleted
    # del env

    # create a host-resident reference actor in shared memory, which serves
    #  as a vessel for updating the actors in workers.
    reference = deepcopy(learner).cpu().share_memory()
    # XXX we need to figure out what exactly do we do with actors

    # spawn workers
    p_workers = mp.spawn(worker_queue, args=(
        q_empty, q_ready, factory, reference, buffers, sticky, pin_memory
    ), nprocs=n_actors, daemon=False, join=False)

    # step through
    step = 0
    while True:
        filled = [q_ready.get() for _ in range(n_buffers_per_batch)]
        batch = unsafe_apply(
            *(buffers[index] for index in filled),
            fn=lambda *x: torch.cat(x, dim=1, out=None)
                               .to(device, non_blocking=True))
        for index in filled:
            q_empty.put(index)

        # train
        try:
            yield batch

        except GeneratorExit:
            # send shutdown signals to workers
            # * closing the empty queue doesn't work register
            # * can only put None-s
            for _ in range(n_actors):
                q_empty.put(None)

            p_workers.join()
            raise

        reference.load_state_dict(learner.state_dict())

        step += 1

    return


def train_one(j, module, batch, optim):
    # batch = aliased(batch)

    actions, hx, info = module(batch.state.obs, batch.state.act,
                               batch.state.rew, batch.state.fin,
                               hx=batch.hx)

    # crew = np_compute_returns(
    #     batch.npy.state.rew, batch.npy.state.fin,
    #     gamma=1.0, bootstrap=batch.npy.bootstrap)

    # time.sleep(0.25)
    module._counter.add_(1)
    return j < 120  # return True

    # logits, values = info['logits'], info['value']

    # # log_prob = logits.index_select(-1, actions)

    # negent = torch.kl_div(torch.tensor(0.), logits.exp()).sum(dim=-1).mean()

    # ell2 = 0.

    # optim.zero_grad()
    # (1e-1 * ell2 + 1e-2 * negent).backward()
    # optim.step()

    # # pass

    # return j < 1200  # return True


if __name__ == '__main__':
    from rlplay.zoo.env import NarrowPath
    # from gym_discomaze import RandomDiscoMaze
    from gym_discomaze.ext import RandomDiscoMazeWithPosition

    # in the main process
    # the fragment specs: the number of simultaneous environments
    #  and the number of steps of one fragment.
    B, T = 2, 80

    # a pickleable environment factory
    def factory():
        return NarrowPath()
        # return gym.make('NetHackScore-v0')
        # return gym.make('CartPole-v0').unwrapped
        # return gym.make('BreakoutNoFrameskip-v4').unwrapped
        # return gym.make('SpaceInvadersNoFrameskip-v4').unwrapped
        # return RandomDiscoMaze(10, 10, field=(2, 2))
        return RandomDiscoMazeWithPosition(10, 10, field=(2, 2))

    # initialize a sample environment and our learner model instance
    env = factory()

    learner = Actor(env.observation_space, env.action_space)
    learner.train()
    device_ = torch.device('cpu')  # torch.device('cuda:0')
    learner.to(device=device_)

    # prepare the optimizer for the learner
    optim = None  # torch.optim.Adam(learner.parameters(), lr=1e-1)

    collector_queue(CloudpickleSpawner(factory), T, B, 8, 24, 4)

    batch = unsafe_apply(*buffers, fn=lambda *x: torch.cat(x, dim=1))
    print(batch)
    print(unsafe_apply(batch, fn=lambda x: (x.shape, x.is_shared())))
    print(unsafe_apply(batch, fn=lambda x: x.numel() * x.element_size()))
    print(learner._counter)


    exit(0)

    p_collectors = []
    for _ in range(8):
        p = mp.Process(target=collector_double, args=(
            CloudpickleSpawner(factory), T, B
        ), daemon=False)
        p.start()
        p_collectors.append(p)

    for p in p_collectors:
        p.join()
