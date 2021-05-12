import os
import time
import copy
import tqdm
import wandb

import gym
import torch

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from rlplay.algo import dqn
# from rlplay.buffer import SimpleBuffer, PriorityBuffer
from rlplay.utils import linear, greedy
from rlplay.utils import backupifexists
from rlplay.utils.schema import astype

from rlplay.utils import ToTensor
from rlplay.utils import AtariObservation, ObservationQueue, FrameSkip
from rlplay.utils import RandomNullopsOnReset, TerminateOnLostLife
# from rlplay.zoo.models import BreakoutQNet

from rlplay.utils import get_instance  # yep, this one, again -_-

import rlplay.utils.integration.gym  # noqa: F401; apply patches to gym

from rlplay.utils.plotting import Conv2DViewer, DummyConv2DViewer

from functools import partial
from tempfile import mkdtemp

from gym.wrappers import Monitor


@torch.no_grad()
def update_target_model(*, src, dst):
    dst.load_state_dict(src.state_dict())


# wandb mode
wandb_mode = 'online'  # 'disabled'
tags = ['d-dqn', 'test', 'breakout']

render = True
# `monitor_gym` makes wnadb patch gym's ImageRecorder with a `wandb.log({})`,
#  without `commit=False`, which messes with the logging in the main loop.
monitor_gym = False

# setup folders for the run
root = os.path.abspath('./runs')
os.makedirs(root, exist_ok=True)

path_ckpt = os.path.join(root, 'ckpt')
os.makedirs(path_ckpt, exist_ok=True)

latest_ckpt = os.path.join(path_ckpt, 'latest.pt')

# hyperparamters: a step is one interaction with environment
config = dict(
    seed=None,  # 897_458_056
    double=True,
    gamma=0.99,
    lr=25e-5,
    epsilon=dict(         # epsilon-greedy exploration policy
        t0=0,             # linear decay from (t=0, v=1.) to (t=100k, v=0.1)
        t1=1_000_00+0,    # maybe we should use cyclic epsilon schedule?
        v0=1e-0,
        v1=1e-1,
    ),
    n_transitions=50_0+00,       # num. burn-in steps to prepopulate the buffer
    n_steps_total=50_000_0+00,   # total number of steps (excl. burn-in)
    n_update_frequency=1,        # number of steps between parameter updates
    n_batches_per_update=1,      # number of sgd training batches in an update
    n_batch_size=32,             # batch size in a single update
    n_freeze_frequency=10_000,   # updates between target q-net freezes
                                 # 10k upd * 1 steps/upd = The rarer the better!
                                 # increasing from 100 to 10k was RIGHT!
    n_checkpoint_frequency=250,  # updates between online q-net checkpoints
    beta=dict(            # importance weight annealing
        t0=0,             # beta = 0 -- prio. sampling (biased)
        t1=25_000_0+00,   # beta = 1 -- random sampling (IS -> uniform)
        v0=4e-1,
        v1=1e-0,
    ),
    # replay=dict(        # simple ring buffer
    #     cls="<class 'rlplay.buffer.simple.SimpleBuffer'>",
    #     capacity=1_000_00+0,
    # ),
    replay=dict(          # prioritized ring buffer
        cls="<class 'rlplay.buffer.priority.PriorityBuffer'>",
        capacity=1_000_00+0,     # smaller buffer may contain less 'garbage',
                                 # but also means less likely to retain rare
                                 # but promising states. Also, maybe as the
                                 # number of steps grows, we should increase
                                 # the capacity? 100k buffer seems okay.
        alpha=0.6,
    ),
    max_nullops=30,              # random number of no-op at the start of an
                                 #  episode. Appears to affect the prng of ALE.
    terminate_on_loss_of_life=False,  # whether to restart of loss-of-life
    observation_shape=(84, 84),
    n_frame_stack=4,
    n_frame_skip=4,
    model=dict(
        cls="<class 'rlplay.zoo.models.breakout.BreakoutQNet'>",
        batch_norm=False,
    ),
    clip_grad_norm=0.0,
    clip_rewards=0.,
    clip_td_error=0.,
    loss_fn='smooth_l1_loss',  # the loss function to use in DQN
)

# the device
device = torch.device('cpu')
clsViewer = Conv2DViewer if render else DummyConv2DViewer

# wandb is gud frmaewrk u shud uze plaeaze
with wandb.init(
         tags=tags, config=config, monitor_gym=monitor_gym,
         mode=wandb_mode, dir=root) as experiment:

    config = experiment.config

    # get the loss function for DQN
    # XXX should we use? "<class 'torch.nn.functional.mse_loss'>"
    if config['loss_fn'] == 'smooth_l1_loss':
        loss_fn = F.smooth_l1_loss

    elif config['loss_fn'] == 'mse_loss':
        loss_fn = F.mse_loss

    else:
        raise ValueError(f'Unrecognized loss function {config["loss_fn"]}.')

    # dtype schema for iterating over the buffer
    schema = dict(state=torch.float, action=torch.long, reward=torch.float,
                  state_next=torch.float, done=torch.bool)

    # create a dedicated generator for experience buffers
    g_cpu = torch.Generator(torch.device('cpu'))
    replay = get_instance(**config['replay'], generator=g_cpu)

    # an instance of atari Breakout-v4
    env = gym.make('BreakoutNoFrameskip-v4')
    env = RandomNullopsOnReset(env, max_nullops=config['max_nullops'])
    if config['terminate_on_loss_of_life']:
        env = TerminateOnLostLife(env)
    env = AtariObservation(env, shape=config['observation_shape'])
    env = ToTensor(env)
    env = FrameSkip(env, n_frames=config['n_frame_skip'], kind='max')
    env = ObservationQueue(env, n_size=config['n_frame_stack'])

    # on-device storage for state (one-element batch)
    state_ = torch.empty(1, *env.observation_space.shape,
                         dtype=torch.float32, device=device)

    # q_net, target_q_net
    if os.path.isfile(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=torch.device('cpu'))
        spec = ckpt.get('model', config['model'])

        q_net = get_instance(env.action_space.n, **spec)
        q_net.load_state_dict(ckpt['q_net'])

        # rename the latest and keep the backup, unless we do not train
        if config['n_steps_total'] > 0:
            backupifexists(latest_ckpt, prefix='backup')
    else:
        q_net = get_instance(env.action_space.n, **config['model'])

    # copy the target network (twice, for safety)
    q_net.to(device)
    target_q_net = copy.deepcopy(q_net).to(device)
    update_target_model(src=q_net, dst=target_q_net)

    # the optimizer and schedulers
    optim = torch.optim.Adam(q_net.parameters(), lr=config['lr'])

    # epsilon-greedy exploration schedule
    epsilon_schedule = partial(linear, **config['epsilon'])

    # priority sampled estimator's bias schedule
    beta_schedule = partial(linear, **config['beta'])

    # various statistics
    n_episodes, n_qnet_updates = 0, 0
    n_episode_start, f_episode_reward = 0, 0.
    n_checkpoint_countdown, n_freeze_countdown = 0, 0
    n_update_countdown = 0

    # keep track of action counts per episode (for debug)
    l_episode_actions = []
    action_names = env.unwrapped.get_action_meanings()

    # set env seed
    if config['seed'] is not None:
        env.seed(config['seed'])

    # request immediate env reset
    done = True

    # intermediate output viewer: use custom identity taps, instead of Conv2d
    viewer = clsViewer(q_net, tap=torch.nn.Identity)
    viewer.toggle(False)  # inactive by default: won't collect intermediates

    # wandb-friendly monitor: does not affect the parameters of the wrapped env
    if monitor_gym:
        env = Monitor(env, directory=mkdtemp(prefix='mon_', dir=root),
                      force=True)

    # count the burn-in as well
    n_total_steps = config['n_steps_total'] + config['n_transitions']
    for n_step in tqdm.tqdm(range(n_total_steps)):
        q_net.eval()

        # handle episodic interaction
        if done:
            # log end of episode
            n_counts = torch.bincount(
                torch.tensor(l_episode_actions, dtype=torch.long),
                minlength=env.action_space.n)

            # wandb reacts to nested dicts and to slashes,
            #  nested dicts are flattened with `dot`
            experiment.log({
                'n_episodes': n_episodes,
                'n_duration': n_step - n_episode_start,
                'f_episode_reward': f_episode_reward,
                'n_action': dict(zip(action_names, n_counts.tolist())),
            }, step=n_step, commit=False)
            # assert experiment.step == n_step

            # begin a new episode
            n_episodes += 1
            n_episode_start, f_episode_reward = n_step, 0.
            l_episode_actions = []
            state, done = env.reset(), False
            if render:
                env.render('human')

        # epsilon-greedy tracks sgd updates
        epsilon_ = epsilon_schedule(n_qnet_updates)

        # sample one step according to the current exploration policy
        f_step_start = time.monotonic()
        with torch.no_grad(), viewer:
            state_.copy_(torch.from_numpy(state))  # copy_ also broadcasts
            q_values = q_net(state_)
            action = int(greedy(q_values, epsilon=epsilon_).squeeze(0))

        # get the environment's response and store it
        state_next, reward, done, info = env.step(action)
        if render:
            env.render('human')

        # XXX `info` may have different fields depending on the env
        replay.commit(state=state, action=action, reward=reward,
                      state_next=state_next, done=done)

        l_episode_actions.append(action)

        # next state
        state = state_next
        f_episode_reward += reward

        experiment.log({
            'f_step_time': time.monotonic() - f_step_start,
            'f_epsilon': epsilon_,
            'f_reward': reward,
            'f_avg_q_value': float(q_values.mean()),
        }, step=n_step, commit=False)

        if len(replay) < config['n_transitions']:
            continue

        # train with a preset frequency
        n_update_countdown -= 1
        if n_update_countdown > 0:
            continue
        n_update_countdown = config['n_update_frequency']

        # train for one batch only if the buffer has enough data.
        q_net.train()
        target_q_net.eval()  # deactivate dropout and batch norms in the target

        losses = []
        for _ in range(config['n_batches_per_update']):
            batch = replay.draw(config['n_batch_size'], replacement=True)
            batch = astype(batch, schema=schema, device=device)

            if config['clip_rewards'] > 0.:
                batch['reward'].clamp_(-config['clip_rewards'],
                                       config['clip_rewards'])

            # beta scheduling for loss weights (related to prioritized replay)
            weight = batch.get('_weight')
            if weight is not None:
                beta = beta_schedule(n_qnet_updates)
                weight = weight.to(state_).pow_(beta)

            loss, info = dqn.loss(batch, gamma=config['gamma'],
                                  module=q_net, target=target_q_net,
                                  double=config['double'], weights=weight,
                                  loss=loss_fn)

            # sgd step
            optim.zero_grad()
            loss.backward()

            f_grad_norm = float('nan')
            if config['clip_grad_norm'] > 0.:
                f_grad_norm = clip_grad_norm_(
                    q_net.parameters(), max_norm=config['clip_grad_norm'])

            optim.step()

            # reassign priority (related to prioritized replay)
            priority = abs(info['td_error']).cpu().squeeze(-1)
            if config['clip_td_error'] > 0.:
                priority.clamp_(max=config['clip_td_error'])

            for j, p in zip(batch['_index'].tolist(), priority.tolist()):
                replay[j] = p + 1e-6

            losses.append((
                float(loss),
                float(abs(info['td_error']).mean()),
                float(f_grad_norm),
            ))

        n_qnet_updates += 1

        # freeze the target q-net every once in a while (number of sgd updates)
        if n_freeze_countdown <= 0:
            n_freeze_countdown = config['n_freeze_frequency']
            update_target_model(src=q_net, dst=target_q_net)
        n_freeze_countdown -= 1

        # record metrics
        loss, td_error, f_grad_norm = zip(*losses)
        experiment.log({
            'loss': loss[-1],
            'td_error': td_error[-1],
            'n_qnet_updates': n_qnet_updates,
            'f_grad_norm': f_grad_norm[-1],
        }, step=n_step, commit=False)

        # save the current Q-net occasionally or at the end of the run
        if n_checkpoint_countdown <= 0 or (n_step + 1 >= n_total_steps):
            # backupifexists(latest_ckpt, prefix=f'ckpt__{n_step}')
            torch.save({
                'q_net': q_net.state_dict(),
                'model': config['model'],  # save model specs
            }, latest_ckpt)
            n_checkpoint_countdown = config['n_checkpoint_frequency']

        n_checkpoint_countdown -= 1

    viewer.close()

    # use the experiment's name if available
    if experiment.name is not None:
        torch.save({
            'q_net': q_net.state_dict(),
            'model': config['model'],  # save model specs
        }, os.path.join(path_ckpt, experiment.name + '.pt'))
# end with


# infinite post training rollout
@torch.no_grad()
def rollout(module, viewer=None):
    obs = env.reset()
    done, totrew, n_step = False, 0., 0

    module.eval()
    while not done:
        if not env.render(mode='human'):
            return False
        time.sleep(0.01)

        state_[0].copy_(torch.from_numpy(obs))
        with viewer:
            action = int(greedy(module(state_).squeeze(0), epsilon=0.1))
        obs, reward, done, info = env.step(action)

        totrew += reward
        n_step += 1

    if totrew > 0:
        print(f'{n_step} :: {totrew}')

    return True


viewer = clsViewer(q_net, tap=torch.nn.Identity, pixel=(5, 5))
with env:
    while render and rollout(q_net, viewer):
        pass
