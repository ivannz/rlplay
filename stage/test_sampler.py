import torch
import numpy

from collections import namedtuple

from rlplay.engine.collect import prepare, startup, collect
from rlplay.engine.collect import BaseActorModule
from rlplay.engine.returns import np_compute_returns

from rlplay.utils.schema.base import unsafe_apply, apply_single
from rlplay.utils.schema.shared import aliased
from rlplay.utils.common import multinomial

import time
import gym
import nle
from rlplay.engine.vec import CloudpickleSpawner

import torch.multiprocessing as mp
from threading import BrokenBarrierError

from copy import deepcopy

from functools import wraps


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


def update_actor(learner, actor):
    return actor.load_state_dict(learner.state_dict())


@wraps(update_actor)
def locked_update_actor(learner, actor):
    # disable rollout collection during actor updates
    with actor.state_dict_lock_:
        return update_actor(learner, actor)


@wraps(collect)
def locked_collect(envs, actor, fragment, state, *, sticky=False):
    # disable actor updates during a rollout collection
    with actor.state_dict_lock_:
        return collect(envs, actor, fragment, state, sticky=sticky)


WorkerContext = namedtuple('WorkerContext', ['envs', 'actor', 'fragment', 'state'])


def worker_context(factory, actor, buffer, *, pin_memory=False):
    # assert not actor.training

    # initialize the environments
    n_steps, n_envs = buffer.state.fin.shape
    envs = [factory() for _ in range(n_envs)]

    # prepare the running state and the output buffer
    state, fragment = startup(envs, actor, buffer, pinned=pin_memory)

    return WorkerContext(envs, actor, fragment, state)


def worker_double(
    barrier, factory, actors, buffers,
    *, pin_memory=False, locking=False, sticky=False
):
    import torch
    torch.set_num_threads(1)
    assert len(buffers) == 2

    # initialize the environments
    context = [worker_context(factory, actor, buffer, pin_memory=pin_memory)
               for actor, buffer in zip(actors, buffers)]

    # the bool return value of `collect` controls the loop
    j, flipflop = 0, 0
    fn = locked_collect if locking else collect
    while fn(*context[flipflop], sticky=sticky):
        # switch to the next fragment buffer
        j, flipflop = j + 1, 1 - flipflop

        # indicate that the previous fragment is ready
        try:
            barrier.wait()

        except BrokenBarrierError:
            # our parent breaks the barrier if it wants us to shutdown
            break

    # this branch is visited only if the while clause evaluates to False, and
    # never when the loop is broken out of.
    else:
        # `collect` aborted the while-loop, let the parent know that something
        # went wrong with the current buffer by breaking the barrier.
        barrier.abort()

    # close the environments in states
    for ctx in context:
        for env in ctx.envs:
            env.close()

    return True


CollectorContext = namedtuple('CollectorContext', ['actor', 'buffer'])


def new_actor(learner, *, update_lock=True):
    # make a deepcopy of the learner (obv. synchronizes weights)
    actor = deepcopy(learner)

    # set evaluation mode and move to shared memory (in-place)
    actor.share_memory()  # .eval()

    # the actor has a shared lock, which mutexes state dict updates
    if update_lock:
        actor.state_dict_lock_ = mp.Lock()
    # XXX not needed if we use actor duplication

    return actor


def collector_contexts(env, learner, n_steps, n_envs, *, double=True):
    # the actors are stale copies of the learner (parameter-wise)
    actor = new_actor(learner, update_lock=not double)
    actor_alt = new_actor(learner, update_lock=not double) if double else actor
    actors = actor, actor_alt

    # create aliased trajectory fragment buffers in the shared memory
    buffers = [aliased(prepare(env, actor, n_steps, n_envs, shared=True))
               for actor in actors]

    # bundle actor and aliased fragment together for faster switching
    return [CollectorContext(a, f) for a, f in zip(actors, buffers)]


def collector_double(factory, n_steps, n_envs):
    r"""

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
    import torch

    # disable mutlithreaded computations in the child processes
    torch.set_num_threads(1)

    sticky, pin_memory, double = False, False, True

    # initialize a sample environment and our learner model instance
    env = factory()
    learner = SimpleActor(env.observation_space, env.action_space)

    # create two contexts for double buffering
    contexts = collector_contexts(env, learner, n_steps, n_envs, double=double)
    del env

    # prepare the optimizer for the learner
    learner.train()
    device_ = torch.device('cpu')  # torch.device('cuda:0')
    learner.to(device=device_)
    optim = torch.optim.Adam(learner.parameters(), lr=1e-1)

    # print(unsafe_apply(fragment, fn=lambda x: x.shape))
    # we use `.pyt`, since torch has simpler pickling/unpickling for sharing
    actors, fragments_pyt = zip(*((ctx.actor, ctx.buffer.pyt)
                                  for ctx in contexts))

    # the double buffering barrier for syncing interleaving iterations
    barrier = mp.Barrier(2)
    p_worker = mp.Process(
        target=worker_double, group=None, name=None, daemon=False,
        args=(barrier, factory, actors, fragments_pyt),
        kwargs=dict(sticky=sticky, pin_memory=pin_memory, locking=not double),
    )
    p_worker.start()

    # use proper update mechanism
    update = locked_update_actor if not double else update_actor

    # we use double buffering: the code flow is designed and the barriers set
    # up in such a way, that flipflop becomes synchronized between this and the
    # worker processes.
    terminate, j, flipflop = False, 0, 0
    while not terminate:
        # wait for the current fragment to be ready (w.r.t `flipflop`)
        try:
            barrier.wait()

        except BrokenBarrierError:
            # the worker broke the barrier to indicate emergency shutdown
            break

        # get the current fragment and the actor used to collect it
        ctx = contexts[flipflop]
        terminate = not do_stuff(j, learner, optim, *ctx, device=device_)

        # updating the shared actor should not be followed by heavy
        #  computations or any io-bound tasks (aside from itself)
        update(learner, ctx.actor)

        # switch to the next buffer
        j, flipflop = j + 1, 1 - flipflop

    else:
        # we break the barrier, if we wish to shut down the worker
        barrier.abort()

    p_worker.join()

    catted = unsafe_apply(*fragments_pyt, fn=lambda a, b: torch.cat([a, b], dim=1))
    print(catted)
    print(all(unsafe_apply(catted, fn=lambda x: (yield x.is_shared()))))
    print(unsafe_apply(catted, fn=lambda x: x.shape))
    print(learner._counter)


def do_stuff(j, module, optim, actor, fragment, *, device=None):
    # do something with the fragment
    pyt = apply_single(fragment.pyt, fn=lambda t: t.to(device, non_blocking=True))
    # XXX indeed a non-aliased device-resident copy

    actions, hx, info = module(pyt.state.obs, pyt.state.act,
                               pyt.state.rew, pyt.state.fin, hx=pyt.hx)

    logits, values = info['logits'], info['value']

    crew = np_compute_returns(
        fragment.npy.state.rew, fragment.npy.state.fin,
        gamma=1.0, bootstrap=fragment.npy.bootstrap)

    # log_prob = logits.index_select(-1, actions)

    negent = torch.kl_div(torch.tensor(0.), logits.exp()).sum(dim=-1).mean()

    ell2 = 0.

    optim.zero_grad()
    (1e-1 * ell2 + 1e-2 * negent).backward()
    optim.step()

    # time.sleep(0.05)
    # pass

    return j < 1200  # return True


if __name__ == '__main__':
    import tqdm
    import torch
    import numpy

    import gym, nle
    from rlplay.zoo.env import NarrowPath
    # from gym_discomaze import RandomDiscoMaze
    from gym_discomaze.ext import RandomDiscoMazeWithPosition
    import torch.multiprocessing as mp

    # in the main process
    # the fragment specs: the number of simultaneous environments
    #  and the number of steps of one fragment.
    B, T = 4, 80

    # a pickleable environment factory
    def factory():
        return NarrowPath()
        # return gym.make("NetHackScore-v0")
        # return gym.make('CartPole-v0').unwrapped
        # return gym.make('BreakoutNoFrameskip-v4').unwrapped
        # return gym.make('SpaceInvadersNoFrameskip-v4').unwrapped
        # return RandomDiscoMaze(10, 10, field=(2, 2))
        return RandomDiscoMazeWithPosition(10, 10, field=(2, 2))

    collector_double(CloudpickleSpawner(factory), T, B)

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

    # # initialize a sample environment for our main actor
    # env = factory()
    # actor = Actor(env).share_memory()

    # # create an aliased shared buffer for trajectory fragments
    # fragment = aliased(prepare(env, actor, T, B, shared=True))

    # # the sample env is no longer needed, unless by the actor (unlikely
    # #  because instantiated env would have to be communicated to child
    # #  processes).
    # del env

    # # in a child process we spawn a bunch of environments
    # # * receive actor and fragment
    # envs = [factory() for _ in range(B)]
    # collector = Stepper(actor, envs, n_steps=T,
    #                     sticky=False, pin_memory=False)

    # collector.set_buffer(fragment.pyt)

    # it, npy = iter(collector), fragment.npy
    # for j, _ in zip(tqdm.trange(1000), it):
    #     crew = np_compute_returns(npy.state.rew, npy.state.fin,
    #                               gamma=1.0, bootstrap=npy.bootstrap)

    #     # crew = np_compute_gae(npy.state.rew, npy.state.fin, npy.actor['value'],
    #     #                       gamma=1.0, C=0.5, bootstrap=npy.bootstrap)
    #     pass

    # print(fragment.npy)
    # print()
    # print(crew)

    # exit()
