import torch
import numpy
import time
import gym
# import nle

from rlplay.engine.sampler import RolloutSampler

from rlplay.engine.collect import BaseActorModule
from rlplay.utils.schema.base import unsafe_apply

from rlplay.engine.returns import np_compute_returns
from rlplay.utils.common import multinomial
from rlplay.engine.vec import CloudpickleSpawner


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

    rollout = RolloutSampler(
        CloudpickleSpawner(factory), learner,
        n_steps=T, n_actors=2, n_per_actor=B, n_buffers=16, n_per_batch=4,
        sticky=True,
        pinned=False,
        close=False,
        device=device_,
    )

    import tqdm
    for j, batch in enumerate(tqdm.tqdm(rollout)):
        # batch = aliased(batch)
        actions, hx, info = learner(batch.state.obs, batch.state.act,
                                    batch.state.rew, batch.state.fin,
                                    hx=batch.hx)

        # crew = np_compute_returns(
        #     batch.npy.state.rew, batch.npy.state.fin,
        #     gamma=1.0, bootstrap=batch.npy.bootstrap)

        # logits, values = info['logits'], info['value']

        # log_prob = logits.index_select(-1, actions)

        # negent = torch.kl_div(torch.tensor(0.), logits.exp()).sum(dim=-1).mean()

        # ell2 = 0.

        # time.sleep(0.25)
        learner._counter.add_(1)

        # optim.zero_grad()
        # (1e-1 * ell2 + 1e-2 * negent).backward()
        # optim.step()

        rollout.update_from(learner)
        # rollout.update(learner)

        if j > 120:
            break

    print(batch, actions, hx, info, learner._counter)
    exit(0)

    # collector_queue(, T, B, 8, 24, 4)

    # batch = unsafe_apply(*buffers, fn=lambda *x: torch.cat(x, dim=1))
    # print(batch)
    # print(unsafe_apply(batch, fn=lambda x: (x.shape, x.is_shared())))
    # print(unsafe_apply(batch, fn=lambda x: x.numel() * x.element_size()))

    # p_collectors = []
    # for _ in range(8):
    #     p = mp.Process(target=collector_double, args=(
    #         CloudpickleSpawner(factory), T, B
    #     ), daemon=False)
    #     p.start()
    #     p_collectors.append(p)

    # for p in p_collectors:
    #     p.join()
