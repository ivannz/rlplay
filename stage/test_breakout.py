import os
import gym
import time
import torch

from rlplay.utils import ToTensor
from rlplay.utils import AtariObservation, ObservationQueue, FrameSkip
from rlplay.zoo.toy import BreakoutQNet

from rlplay.utils import greedy

from rlplay.utils.plotting import Conv2DViewer


path_ckpt = os.path.join(os.path.abspath('./runs'), 'checkpoints')
device = torch.device('cpu')

# an instance of atari Breakout-v4
env = gym.make('BreakoutNoFrameskip-v4')
# env = TerminateOnLostLive(env)  # messes up the randomness of ALE
env = AtariObservation(env, shape=(84, 84))
env = ToTensor(env)
env = FrameSkip(env, n_frames=4, kind='max')
env = ObservationQueue(env, n_size=4)

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
            q_value = module(state_).squeeze(0)

        # print(q_value.cpu().numpy())
        action = int(greedy(q_value, epsilon=0.1))
        obs, reward, done, info = env.step(action)

        totrew += reward
        n_step += 1

    if totrew > 0:
        print(f'{n_step} :: {totrew}')

    return True


viewer = Conv2DViewer(q_net, tap=torch.nn.Identity, pixel=(5, 5))
with env:
    while rollout(q_net, viewer):
        pass
