import numpy as np

from gym import Env
from gym.spaces import Discrete, Box

from sys import maxsize
from random import Random


class ChainMDP(Env):
    """Simple chain MDP.

    Details
    -------
    The state are connected in a chain, both endpoints of which are linked to
    the common absorbing state. The agent receives a nonzero reward if it gets
    into the absorbing state through either endpoints: if it is the left end,
    then it receives a reward of 0.7, if the right -- a reward of 1.0. Ending
    up in the absorbing state means the end of the game, and any further action
    would yield reward zero.

    The state space is a binary vector, in which the states are one-hot coded,
    and the discrete action space has two options.

    For a chain with 3 intermediate states, the correct Q-values for a discount
    rate of `0.5` are:
        [[0.7, 0.25], [0.35, 0.50], [0.25, 1.00], [0., 0.]]
    """
    def __init__(self, *, n_states=3, p=1., random=None):
        self.random_ = random or Random()
        assert isinstance(self.random_, Random)

        self.n_states, self.p = n_states, p

        self.action_space = Discrete(2)  # left, right
        self.observation_space = Box(
            low=0., high=1., dtype=np.float32, shape=(n_states+1,))

        self.reset()

    def observation(self):
        return np.eye(1, self.n_states+1, self.state, dtype=np.float32)[0]

    def reset(self):
        # `self.n_states-1` state is absorbing
        self.state = self.random_.randrange(self.n_states+1)
        return self.observation()

    def step(self, action):
        if self.state == self.n_states:
            return self.observation(), 0., True, {}

        if self.random_.random() < self.p:
            self.state += -1 if action == 0 else +1

        reward = 0.
        done = not 0 <= self.state < self.n_states
        if done:
            reward = 1. if self.state >= self.n_states else 0.7
            self.state = self.n_states

        return self.observation(), reward, done, {}

    def seed(self, seed=None):
        if seed is None:
            seed = Random().randrange(maxsize)

        self.random_.seed(seed)
        return [seed]

    def close(self):
        pass
