import os
import gym
import time
import torch


from rlplay.utils import ToTensor
from rlplay.utils import AtariObservation, ObservationQueue, FrameSkip
from rlplay.utils import RandomNullopsOnReset, TerminateOnLostLive

from rlplay.zoo.models import BreakoutQNet

from rlplay.utils import greedy

from rlplay.utils.plotting import Conv2DViewer

import rlplay.utils.integration.gym  # noqa: F401; apply patches to gym

import matplotlib.pyplot as plt

from matplotlib.ticker import EngFormatter
from rlplay.utils.plotting import ImageViewer


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


path_ckpt = os.path.join(os.path.abspath('./runs'), 'ckpt')
device = torch.device('cpu')

# an instance of atari Breakout-v4
env = gym.make('BreakoutNoFrameskip-v4')
env = RandomNullopsOnReset(env, max_nullops=30)
# env = TerminateOnLostLive(env)  # messes up the randomness of ALE
env = AtariObservation(env, shape=(84, 84))
env = ToTensor(env)
env = FrameSkip(env, n_frames=4, kind='max')
env = ObservationQueue(env, n_size=4)
print(env.unwrapped.get_action_meanings())

q_net = BreakoutQNet(env.action_space.n).to(device)
print(repr(q_net))

checkpoint = os.path.join(path_ckpt, 'latest.pt')
if os.path.isfile(checkpoint):
    ckpt = torch.load(checkpoint, map_location=torch.device('cpu'))
    q_net.load_state_dict(ckpt['q_net'])

state_ = torch.empty(1, *env.observation_space.shape,
                     dtype=torch.float32, device=device)


# infinite post training rollout
@torch.no_grad()
def rollout(module, viewer=None):
    module.eval()

    obs = env.reset()
    done, totrew, n_step = False, 0., 0
    while not done:
        if not env.render(mode='human'):
            return False
        time.sleep(0.01)

        state_[0].copy_(torch.from_numpy(obs))
        with viewer:
            q_values = module(state_).squeeze(0)

        action = int(greedy(q_values, epsilon=0.05))

        # plot q-values
        ax.clear()
        bars = show_q_values(q_values, ax=ax, title=f'step {n_step}',
                             labels=env.unwrapped.get_action_meanings())
        bars.patches[action].set_facecolor('C1')

        fig.tight_layout()
        hist.imshow(fig)

        obs, reward, done, info = env.step(action)

        totrew += reward
        n_step += 1

    if totrew > 0:
        print(f'{n_step} :: {totrew}')

    return True


viewer = Conv2DViewer(q_net, tap=torch.nn.Identity, pixel=(5, 5))
hist = ImageViewer(vsync=True)
with env:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=60)
    while rollout(q_net, viewer):
        pass
    plt.close(fig)

hist.close()
