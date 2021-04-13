import os
import time
import copy
import tqdm
import wandb

import gym
import torch

from torch.nn.utils import clip_grad_norm_

from rlplay.algo import dqn
from rlplay.buffer import SimpleBuffer, PriorityBuffer
from rlplay.buffer import to_device, ensure
from rlplay.utils import linear, greedy

from rlplay.utils import ToTensor
from rlplay.utils import AtariObservation, ObservationQueue, FrameSkip
from rlplay.zoo.toy import BreakoutQNet

from functools import partial
from tempfile import mkdtemp

from gym.wrappers import Monitor


class TerminateOnLostLive(gym.Wrapper):
    @property
    def ale(self):
        return self.env.unwrapped.ale

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # detect loss-of-life
        current = self.ale.lives()
        done = done or current < self.lives
        self.lives = current

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.lives = self.ale.lives()
        return obs


@torch.no_grad()
def update_target_model(*, src, dst):
    dst.load_state_dict(src.state_dict())


# wandb mode
wandb_mode = 'online'  # 'disabled'
n_checkpoint_frequency = 250
render = True
# `monitor_gym` makes wnadb patch gym's ImageRecorder with a `wandb.log({})`,
#  without `commit=False`, which messes with the logging in the main loop.
monitor_gym = False

# setup folders for the run
root = os.path.abspath('./runs')
os.makedirs(root, exist_ok=True)

path_ckpt = os.path.join(root, 'checkpoints')
os.makedirs(path_ckpt, exist_ok=True)

# hyperparamters
config = dict(
    seed=None,  # 897_458_056
    gamma=0.99,
    n_batch_size=32,
    n_transitions=50_00+0,  # 500 is a really small replay buffer
    n_steps_total=10_000_0+00,
    n_freeze_frequency=10_0+00,
    lr=2e-3,  # 25e-5,
    epsilon=dict(
        t0=0,
        t1=3_000_0+00,  # t1=1_000_0+00,
        v0=1e-0,
        v1=1e-1,
    ),
    beta=dict(
        t0=0,
        t1=5_000_0+00,
        v0=4e-1,
        v1=1e-0,
    ),
    replay__alpha=0.6,
)

# the device
device = torch.device('cpu')

# wandb is gud frmaewrk u shud uze plaeaze
with wandb.init(
         tags=['d-dqn', 'test', 'breakout'],
         config=config, monitor_gym=monitor_gym,
         mode=wandb_mode, dir=root) as experiment:

    config = experiment.config

    # an instance of atari Breakout-v4
    env = gym.make('BreakoutNoFrameskip-v4')
    # env = TerminateOnLostLive(env)  # messes up the randomness of ALE
    env = AtariObservation(env, shape=(84, 84))
    env = ToTensor(env)
    env = FrameSkip(env, n_frames=4, kind='max')
    env = ObservationQueue(env, n_size=4)

    # wandb-friendly monitor
    if monitor_gym:
        env = Monitor(env, force=True,
                      directory=mkdtemp(prefix='mon_', dir=root))

    # set env seed
    if config['seed'] is not None:
        env.seed(config['seed'])

    # dtype schema for iterating over the buffer
    schema = dict(state=torch.float, action=torch.long, reward=torch.float,
                  state_next=torch.float, done=torch.bool, info=None)

    # replay = SimpleBuffer(config['n_transitions'], config['n_batch_size'])
    replay = PriorityBuffer(config['n_transitions'], config['n_batch_size'],
                            alpha=config['replay__alpha'])

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
    beta_schedule = partial(linear, **config['beta'])

    # request immediate env reset
    done, n_episodes, n_qnet_updates = True, 0, 0
    n_episode_start, f_episode_reward = 0, 0.
    n_checkpoint_countdown, n_freeze_countdown = 0, 0

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
            }, step=n_step, commit=False)
            # assert experiment.step == n_step

            # begin a new episode
            n_episodes += 1
            n_episode_start, f_episode_reward = n_step, 0.
            state, done = env.reset(), False
            if render:
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
        if render:
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
        }, step=n_step, commit=False)

        if len(replay) < config['n_transitions']:
            continue

        # train for one batch only if the buffer has enough data.
        batch = next(iter(replay))
        batch = to_device(ensure(batch, schema=schema), device=device)

        # beta scheduling for loss weights
        weight = batch.get('_weight')
        if weight is not None:
            beta = beta_schedule(n_qnet_updates)
            weight = weight.to(state_).pow_(beta)

        loss, info = dqn.loss(batch, gamma=config['gamma'],
                              module=q_net, target=target_q_net,
                              weights=weight)

        # reassign priority
        priority = abs(info['td_error']).cpu().squeeze_().add_(1e-6)
        for j, p in zip(batch['_index'], priority):
            replay[j] = p

        # sgd step
        optim.zero_grad()
        loss.backward()
        f_grad_norm = float(clip_grad_norm_(q_net.parameters(), max_norm=1.0))
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
            'n_qnet_updates': n_qnet_updates,
            'f_grad_norm': f_grad_norm,
        }, step=n_step, commit=False)

        # from time to time save the current Q-net
        if n_checkpoint_countdown <= 0:
            n_checkpoint_countdown = n_checkpoint_frequency
            torch.save({
                'q_net': q_net.state_dict(),
            }, os.path.join(path_ckpt, 'latest.pt'))
        n_checkpoint_countdown -= 1
# end with


# infinite post training rollout
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
    while render and rollout():
        pass
