import torch
import numpy

from rlplay.engine.collect import prepare, startup, collect
from rlplay.engine.collect import BaseActorModule
from rlplay.engine.returns import np_compute_returns

from rlplay.utils.schema.base import unsafe_apply
from rlplay.utils.schema.shared import aliased
from rlplay.utils.common import multinomial

import time
import gym
import nle
from rlplay.utils.vec import CloudpickleSpawner

from threading import BrokenBarrierError


class Actor(BaseActorModule):
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


def worker_double(factory, actors, buffers, sticky, pin_memory, lockstep):
    # * receive `factory`, `actors`, `buffers`, `sticky`, and `pin_memory`
    assert len(buffers) == 2

    # initialize the environments
    context = []
    for buffer, actor in zip(buffers, actors):
        n_steps, n_envs = buffer.state.fin.shape
        envs = [factory() for _ in range(n_envs)]

        # prepare the running state and the output buffer
        state, fragment = startup(envs, actor, buffer, pinned=pin_memory)
        context.append((envs, actor, fragment, state))

    # do some preparations
    j, flipflop = 0, 0

    # the bool return value of `collect` controls the loop
    while collect(*context[flipflop], sticky=sticky):
        # switch to the next fragment buffer
        j, flipflop = j + 1, 1 - flipflop

        # indicate that the previous fragment is ready
        try:
            lockstep.wait()

        except BrokenBarrierError:
            # our parent breaks the barrier if it wants us to shutdown
            break

    else:  # branch here only if the while loop was NOT broken out of.
        # `collect` aborted the while-loop, let the parent know something went
        # wrong by breaking the barrier.
        lockstep.abort()

    # close the environments in states
    pass


def collector_double(factory, n_steps, n_envs):
    # * receive `factory`, `n_steps`, and `n_envs`
    import torch.multiprocessing as mp

    sticky, pin_memory = False, False

    # initialize a sample environment and our main actor
    env = factory()

    # the actor should not store references to instantiated env, since it is
    # about to be communicated to child processes.
    actors = [SimpleActor(env.observation_space, env.action_space)
              for _ in range(2)]

    learner = SimpleActor(env.observation_space, env.action_space)

    # create an aliased shared buffer for trajectory fragments
    buffers = []
    for actor in actors:
        # prepare the trajectory fragment buffer in the shared memory
        buffer = prepare(env, actor, n_steps, n_envs // 2, shared=True)
        actor.load_state_dict(learner.state_dict())

        # bundle actor and aliased fragment together for faster switching
        buffers.append((actor.share_memory(), aliased(buffer)))

    del env, buffer

    # prepare the optimizer for the learner
    optim = torch.optim.Adam(learner.parameters(), lr=1e-1)

    # print(unsafe_apply(fragment, fn=lambda x: x.shape))
    # we use `.pyt`, since torch has simpler pickling/unpickling for sharing
    actors, fragments_pyt = zip(*((a, f.pyt) for a, f in buffers))

    # spawn the worker process
    lockstep = mp.Barrier(2)
    p_worker = mp.Process(target=worker_double, args=(
        factory, actors, fragments_pyt, sticky, pin_memory, lockstep
    ), daemon=False)
    p_worker.start()

    # we use double buffering
    terminate, j, flipflop = False, 0, 0
    while not terminate:
        # wait for the current fragment to be ready (w.r.t `flipflop`)
        try:
            lockstep.wait()

        except BrokenBarrierError:
            # the worker broke the barrier to indicate emergency shutdown
            break

        # get the current fragment and the actor used to collect it
        terminate = not do_stuff(j, learner, optim, *buffers[flipflop])

        # switch to the next buffer
        j, flipflop = j + 1, 1 - flipflop

    else:
        # we break the barrier, if we wish to shut down the worker
        lockstep.abort()

    p_worker.join()

    catted = unsafe_apply(*fragments_pyt, fn=lambda a, b: torch.cat([a, b], dim=1))
    print(catted)
    print(all(unsafe_apply(catted, fn=lambda x: (yield x.is_shared()))))
    print(unsafe_apply(catted, fn=lambda x: x.shape))


def do_stuff(j, module, optim, actor, fragment):
    # do something with the fragment
    pyt = fragment.pyt

    module.train()

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

    # update the shared actor
    actor.load_state_dict(module.state_dict())

    # time.sleep(0.25)
    pass

    # print(j, crew, fragment.npy.bootstrap)
    print(j, float(negent))

    return j < 30  # return True


if __name__ == '__main__':
    import tqdm
    import torch
    import numpy

    import gym, nle
    from rlplay.zoo.env import NarrowPath
    # from gym_discomaze import RandomDiscoMaze
    from gym_discomaze.ext import RandomDiscoMazeWithPosition

    # in the main process
    # the fragment specs: the number of simultaneous environments
    #  and the number of steps of one fragment.
    B, T = 8, 80

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
