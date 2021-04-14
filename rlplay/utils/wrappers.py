import numpy as np

from gym import ObservationWrapper, Wrapper
from gym.spaces import Box

from cv2 import cvtColor, resize
from cv2 import COLOR_RGB2GRAY, INTER_AREA


class ToTensor(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0., high=1., dtype=np.float32,
                                     shape=env.observation_space.shape)

    def observation(self, observation):
        return np.divide(observation, 255, dtype=np.float32)


class ChannelFirst(ObservationWrapper):
    """Convert rgb-image observations to scaled channel-first tensors the on-the-fly."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=self.observation(env.observation_space.low),
            high=self.observation(env.observation_space.high),
            dtype=env.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class AtariObservation(ObservationWrapper):
    """Convert image observation space from RGB to grayscale and downsample."""
    def __init__(self, env, shape=(84, 84)):
        shape = (shape, shape) if isinstance(shape, int) else shape
        assert all(x > 0 for x in shape), f'Invalid shape {shape}'
        # check if the input env's states are rgb pixels [h x w x 3]
        space = env.observation_space
        assert isinstance(space, Box)
        assert len(space.shape) == 3 and space.shape[-1] == 3

        # override the observation space
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=shape,
                                     dtype=np.uint8)

    def observation(self, observation):
        # convert to grayscale before decimating, to avoid artifacts
        dsize = self.observation_space.shape[::-1]
        return resize(cvtColor(observation, COLOR_RGB2GRAY),
                      dsize, interpolation=INTER_AREA)


class ObservationQueue(Wrapper):
    """Simple queue of the most recent observations."""
    def __init__(self, env, n_size=4):
        space = env.observation_space
        assert isinstance(space, Box)

        # override the observation space
        super().__init__(env)
        self.observation_space = Box(low=np.stack([space.low] * n_size),
                                     high=np.stack([space.high] * n_size),
                                     shape=(n_size, *space.shape),
                                     dtype=space.dtype)
        self.reset()

    def reset(self, **kwargs):
        space = self.observation_space

        self.buffer = np.empty(space.shape, dtype=space.dtype)
        self.buffer[:] = self.env.reset(**kwargs)

        return self.buffer[:]  # make sure to return a copy

    def step(self, action):
        # evict the oldest frame
        # XXX `np.roll` is not the most efficient solution
        self.buffer = np.roll(self.buffer, shift=-1, axis=0)
        self.buffer[-1], reward, done, info = self.env.step(action)

        # make sure to return a copy
        return self.buffer[:], reward, done, info


class FrameSkip(Wrapper):
    """Convert image observation space from RGB to grayscale and downsample."""
    def __init__(self, env, n_frames=4, *, kind='max', n_pool=2):
        assert isinstance(env.observation_space, Box)
        assert kind in ('max', 'average', 'last')

        super().__init__(env)
        self.n_frames, self.n_pool, self.kind = n_frames, n_pool, kind
        if kind == 'max':
            self.op = lambda x: np.max(x, axis=0)

        elif self.kind == 'average':
            self.op = lambda x: np.mean(x, axis=0)

        elif self.kind == 'last':
            self.op = lambda x: np.take(x, -1, axis=0)

    def step(self, action):
        # stick to the same action for the specified number of steps
        results = []
        for _ in range(self.n_frames):
            obs, reward, done, info = self.env.step(action)
            results.append((obs, reward, done, info))
            if done:
                break

        obs, reward, done, info = zip(*results)

        # aggregate appropriately
        obs = self.op(obs[-self.n_pool:])
        return obs, sum(reward), any(done), info
