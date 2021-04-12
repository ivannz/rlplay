import torch
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


@torch.no_grad()
def greedy(q_value, *, epsilon=0.5):
    """epsilon-greedy strategy."""
    q_action = q_value.max(dim=-1).indices

    random = torch.randint(q_value.shape[-1], size=q_action.shape,
                           device=q_value.device)
    is_random = torch.rand_like(q_value[..., 0]).lt(epsilon)
    return torch.where(is_random, random, q_action)
