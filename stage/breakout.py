import os
import time
import copy
import tqdm
import wandb

import gym
import torch

from torch.nn.utils import clip_grad_norm_

from rlplay.algo import dqn
from rlplay.buffer import SimpleBuffer, to_device, ensure
from rlplay.utils import linear, greedy

from rlplay.utils import ToTensor
from rlplay.utils import AtariObservation, ObservationQueue, FrameSkip
from rlplay.zoo.toy import BreakoutQNet

from functools import partial
from tempfile import mkdtemp

from gym.wrappers import Monitor


class ResetOnLostLive(gym.Wrapper):
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.lives = self.unwrapped.ale.lives()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.env.unwrapped.ale.lives() < self.lives:
            self.reset()
            done = True

        return obs, reward, done, info


@torch.no_grad()
def update_target_model(*, src, dst):
    dst.load_state_dict(src.state_dict())


# wandb mode
wandb_mode = 'online'  # 'disabled'

# setup folders for the run
root = os.path.abspath('./runs')
os.makedirs(root, exist_ok=True)

path_ckpt = os.path.join(root, 'checkpoints')
os.makedirs(path_ckpt, exist_ok=True)

path_wandb = os.path.join(root, 'wandb')
os.makedirs(path_wandb, exist_ok=True)

# hyperparamters
config = dict(
    seed=None,  # 897_458_056
    gamma=0.99,
    n_batch_size=32,
    n_transitions=50_0+00,
    n_steps_total=10_000_0+00,
    n_freeze_frequency=10_0+00,
    lr=2e-3,  # 25e-5,
    epsilon=dict(
        t0=0,
        t1=1_000_0+00,
        v0=1e-0,
        v1=1e-1,
    )
)

# the device
device = torch.device('cpu')

# wandb is gud frmaewrk u shud uze plaeaze
with wandb.init(
         tags=['d-dqn', 'test', 'breakout'],
         config=config, monitor_gym=True,
         mode=wandb_mode, dir=path_wandb) as experiment:

    config = experiment.config

    # an instance of atari Breakout-v4
    env = ResetOnLostLive(gym.make('BreakoutNoFrameskip-v4'))
    env = ToTensor(AtariObservation(env, shape=(84, 84)))
    env = ObservationQueue(FrameSkip(env, n_frames=4, kind='max'), n_size=4)

    # wandb-friendly monitor
    env = Monitor(env, directory=mkdtemp(prefix='_mon', dir=root), force=True)

    # set env seed
    if config['seed'] is not None:
        env.seed(config['seed'])

    # dtype schema for iterating over the buffer
    schema = dict(state=torch.float, action=torch.long, reward=torch.float,
                  state_next=torch.float, done=torch.bool, info=None)
    replay = SimpleBuffer(config['n_transitions'], config['n_batch_size'])

    # on-device storage for state
    state_ = torch.empty(1, *env.observation_space.shape,
                         dtype=torch.float32, device=device)

    # q_net, target_q_net
    q_net = BreakoutQNet(env.action_space.n).to(device)
    if os.path.isfile(os.path.join(path_ckpt, 'latest.pt')):
        checkpoint = torch.load(os.path.join(path_ckpt, 'latest.pt'))
        q_net.load_state_dict(checkpoint['q_net'])

        # rename and keep the backup
        dttm = time.strftime('%Y%m%d-%H%M%S')
        os.rename(os.path.join(path_ckpt, 'latest.pt'),
                  os.path.join(path_ckpt, f'backup__{dttm}.pt'))

    target_q_net = copy.deepcopy(q_net).to(device)
    update_target_model(src=q_net, dst=target_q_net)

    # the optimizer and schedulers
    optim = torch.optim.Adam(q_net.parameters(), lr=config['lr'])
    epsilon_schedule = partial(linear, **config['epsilon'])

    # request immediate env reset
    done, n_episodes, n_qnet_updates = True, 0, 0
    n_episode_start, f_episode_reward, n_freeze_countdown = 0, 0., 0

    # count burn-in as well
    n_total_steps = config['n_steps_total'] + config['n_transitions']
    for n_step in tqdm.tqdm(range(n_total_steps)):
        # handle episodic interaction
        if done:
            # log end of episode
            experiment.log({
                'n_episodes': n_episodes,
                'n_duration': n_step - n_episode_start,
                'f_episode_reward': f_episode_reward,
            }, step=n_step)

            # begin a new episode
            n_episodes += 1
            n_episode_start, f_episode_reward = n_step, 0.
            state, done = env.reset(), False
            env.render('human')

        # epsilon-greedy tracks sgd updates
        epsilon_ = epsilon_schedule(n_qnet_updates)

        # sample one step according to the current exploration policy
        f_step_start = time.monotonic()
        with torch.no_grad():
            state_.copy_(torch.from_numpy(state))  # copy_ also broadcasts
            action = int(
                greedy(q_net(state_), epsilon=epsilon_).squeeze(0))

        # get the environment's response and store it
        state_next, reward, done, info = env.step(action)
        env.render('human')

        # XXX `info` may have different fields depending on the env
        replay.commit(state=state, action=action, reward=reward,
                      state_next=state_next, done=done)

        # next state
        state = state_next
        f_episode_reward += reward

        experiment.log({
            'f_step_time': time.monotonic() - f_step_start,
            'f_epsilon': epsilon_,
            'f_reward': reward,
        }, step=n_step)

        if len(replay) < config['n_transitions']:
            continue

        # train for one batch only if the buffer has enough data.
        batch = ensure(next(iter(replay)), schema=schema)
        loss, info = dqn.loss(to_device(batch, device=device), gamma=config['gamma'],
                              module=q_net, target=target_q_net)

        # sgd step
        optim.zero_grad()
        loss.backward()
        f_grad_norm = clip_grad_norm_(q_net.parameters(), max_norm=1.0)
        optim.step()
        n_qnet_updates += 1

        # freeze the target q-net every once in a while (number of sgd updates)
        if n_freeze_countdown <= 0:
            n_freeze_countdown = config['n_freeze_frequency']
            update_target_model(src=q_net, dst=target_q_net)
        n_freeze_countdown -= 1

        # record metrics
        experiment.log({
            'loss': float(loss),
            'td_error': float(abs(info['td_error']).mean()),
            'b_target_freeze': n_freeze_countdown == config['n_freeze_frequency'],
            'n_qnet_updates': n_qnet_updates,
            'f_grad_norm': f_grad_norm,
        }, step=n_step)

        # save the current models
        torch.save({
            'q_net': q_net.state_dict(),
        }, os.path.join(path_ckpt, 'latest.pt'))
# end with


# infinite rollout
@torch.no_grad()
def rollout():
    obs = env.reset()
    done, totrew, n_step = False, 0., 0
    while not done:
        if not env.render(mode='human'):
            return False
        time.sleep(0.01)

        state_[0].copy_(torch.from_numpy(obs))
        # action = int(q_net(state_).max(dim=-1).indices[0])
        action = int(greedy(q_net(state_), epsilon=0.01).squeeze(0))
        obs, reward, done, info = env.step(action)

        totrew += reward
        n_step += 1

    if totrew > 0:
        print(f'{n_step} :: {totrew}')

    return True


with env:
    while rollout():
        pass
