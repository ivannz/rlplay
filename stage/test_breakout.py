import os
import gym
import time
import tqdm
import torch

import torch.nn.functional as F
import numpy as np

from rlplay.utils import ToTensor
from rlplay.utils import AtariObservation, ObservationQueue, FrameSkip
from rlplay.utils import RandomNullopsOnReset, TerminateOnLostLive

from rlplay.utils import get_instance

from rlplay.utils import greedy

from rlplay.utils.plotting import Conv2DViewer

import rlplay.utils.integration.gym  # noqa: F401; apply patches to gym

import matplotlib.pyplot as plt

from matplotlib.ticker import EngFormatter
from rlplay.utils.plotting import MultiViewer

from rlplay.buffer import SimpleBuffer
from rlplay.utils.schema import astype


def expand(min, max, rtol=5e-2, atol=1e-1):
    min = min - abs(min) * rtol - atol
    max = max + abs(max) * rtol + atol
    return min, max


def show_q_values(vals, labels=None, title=None, *, ax=None):
    ax = ax or plt.gca()

    if title is not None:
        ax.set_title(title)
    ax.yaxis.set_major_formatter(EngFormatter(unit=''))

    indices = list(range(len(vals)))
    bars = ax.bar(indices, vals.tolist())
    ax.set_ylim(expand(float(vals.min()), float(vals.max())))

    if labels is not None:
        ax.set_xticks(indices)
        ax.set_xticklabels(labels)

    return bars


# infinite post training rollout
@torch.no_grad()
def rollout(env, module, *, replay=None, conv=None, fig=None):
    # globals: device, epsilon, action_names

    module.eval()

    # allocate a on-device double buffer for observations
    state_ = torch.empty(2, 1, *env.observation_space.shape,
                         dtype=torch.float32, device=device)

    # reset the environment and move the observation onto the device
    obs, done, flip, n_step = env.reset(), False, 0, 0
    state_[flip].copy_(torch.from_numpy(obs))

    # the rollout loop
    history, f_start_time = [], time.monotonic()
    while not done:
        if not env.render(mode='human'):
            return False

        # illustrate the intermediate outputs
        with conv:
            q_values = module(state_[flip]).squeeze(0)

        # act on the q-values and copy the next obs onto the device
        action = int(greedy(q_values, epsilon=epsilon))

        next_obs, reward, done, info = env.step(action)
        state_[1 - flip].copy_(torch.from_numpy(next_obs))

        # collect partial experience
        if replay is not None:
            replay.commit(state=obs, action=action, next_state=next_obs)

        # track history
        history.append((reward, action))
        obs, flip, n_step = next_obs, 1 - flip, n_step + 1

        # do not draw with matplotlib if the viewer is closed
        if fig is not None and conv.viewers['q_values'].isopen:
            ax, = fig.axes
            ax.clear()

            # visualize q-values
            title = f'step {n_step} Q_avg {float(q_values.mean()):.1e}'
            bars = show_q_values(q_values, title=title,
                                 ax=ax, labels=action_names)
            bars.patches[action].set_facecolor('C1')

            conv.viewers.imshow('q_values', fig)

    # end-of-episode statistics
    rewards, actions = zip(*history)
    f_episode_time = time.monotonic() - f_start_time

    n_counts = torch.bincount(
        torch.tensor(actions, dtype=torch.long),
        minlength=env.action_space.n)
    n_actions = dict(zip(action_names, n_counts.tolist()))

    print(f'{n_step} in {f_episode_time:.1f} sec. :: {sum(rewards)}', n_actions)

    return True


# an instance of atari Breakout-v4
env = gym.make('BreakoutNoFrameskip-v4')
env = RandomNullopsOnReset(env, max_nullops=30)
# env = TerminateOnLostLive(env)  # messes up the randomness of ALE
env = AtariObservation(env, shape=(84, 84))
env = ToTensor(env)
env = FrameSkip(env, n_frames=4, kind='max')
env = ObservationQueue(env, n_size=4)


# the test-run hyperparameters
device = torch.device('cpu')
epsilon = 0.05

# create a default q-net just in case
model = {
    'cls': "<class 'rlplay.zoo.models.breakout.BreakoutQNetToo'>",
    'batch_norm': True,
}
q_net = get_instance(env.action_space.n, **model).to(device)

# open a checkpoint
checkpoint = '/Users/ivannazarov/Github/repos_with_rl/rlplay/gud/dandy-monkey-115.pt'
# checkpoint = os.path.join(os.path.abspath('./runs'), 'ckpt', 'latest.pt')
if os.path.isfile(checkpoint):
    ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))

    spec = ckpt.get('model', model)
    q_net = get_instance(env.action_space.n, **spec).to(device)
    print(q_net.load_state_dict(ckpt['q_net']))

else:
    raise OSError(f'Cannot locate `{checkpoint}`.')

print(repr(q_net))

# retrieve proper action labels
if hasattr(env.unwrapped, 'get_action_meanings'):
    action_names = env.unwrapped.get_action_meanings()

else:
    action_names = [f'act_{i}' for i in range(env.action_space.n)]
print(action_names)


# create a 2x pixel zoomed image viewer
viewer = MultiViewer(vsync=True, scale=(2, 2))
viewer.open(q_values='Current Q-values')

# intermediate output tracker and a canvas for q-vaues
conv = Conv2DViewer(q_net, tap=torch.nn.Identity, viewer=viewer)
fig, _ = plt.subplots(1, 1, figsize=(4, 3), dpi=60)

# a simple ring buffer for online transitions
replay = SimpleBuffer(30_000)

# schema for `astype` calls
# schema = dict(state=torch.float32, action=torch.long,
#               next_state=torch.float32)
with env:
    while rollout(env, q_net, replay=replay, fig=fig, conv=conv):
        # use the replay buffer
        # for batch in replay.sample(32):
        #     batch = astype(batch, schema=schema, device=device)
        pass

plt.close(fig)
viewer.close()
