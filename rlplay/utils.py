import numpy as np

from gym import ObservationWrapper
from gym.spaces import Box


class ImageToTensor(ObservationWrapper):
    """Convert rgb-image observations to scaled channel-first tensors the on-the-fly."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=self.observation(env.observation_space.low),
            high=self.observation(env.observation_space.high),
            dtype=np.float32)

    def observation(self, observation):
        observation = observation.transpose(2, 0, 1)
        return np.divide(observation, 255, dtype=np.float32)


def linear(t, t0=0, t1=100, v0=1., v1=0.):
    tau = min(1., max(0., (t1 - t) / (t1 - t0)))
    return v0 * tau + v1 * (1 - tau)
