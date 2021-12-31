# RLPlay

This is by no means a comprehensive implementation of the overwhelming multitude of RL algorithms. For this you might want to check out awesome libraries such as [stable-baselines 3](https://github.com/DLR-RM/stable-baselines3) and [rlpyt](https://github.com/astooke/rlpyt/) (i was unaware of this particular one when naming this repo).


## Motivation

The reason for this take on RL if twofold: 
1. it is easier for me to study the algos and their modifications by building them from ground up
2. although they are really great, i didn't like the pre-packaged object-oriented one-liner approach to algorithms in the repos mentioned above
3. rollout data collection is merely a mechanical method to acquire data from the actor-environment interactions. It is not a part of a learning algorithm, nor is it a part of the actor.

My idea was to shift all focus away from the collection process to the algorithm, the actor and the method implementations within a standard pytorch training loop. I envisioned something like the snippet below:

```python
import gym
import torch

from rlplay.engine.rollout.same import rollout

actor = Actor()  # minimal compatibility requirements
optim = torch.optim.SGD(actor.parameters())

# infinite rollout generator with each 120 steps long fragment coming
#  from a batch of eight environments
feed = rollout([gym.make(...) for _ in range(8)], actor, n_steps=120)

for j, fragment in zip(range(100), feed):  # limit rollout to 100 fragments

    # here we compute the loss for an A2C or D-DQN algorithm for a fragment
    loss, info = rl_algorithm_loss(fragment, actor)

    # do the optimization
    optim.zero_grad()
    loss.backward()
    optim.step()
```

For this it was necessary to implement a lean collection core with minimally necessary interface requirements for rollout trajectory collection. For example, the actor must provide special `.step` and `.reset` methods and have external runtime/recurrent state, like `hx` in `torch.nn.LSTM` (see the extensive documentation in `rlplay.engine.core.BaseActorModule`):

```python
from rlplay.engine import BaseActorModule


class Actor(BaseActorModule):
    def reset(self, hx, at):
        ...
        return hx_update

    def step(self, stepno, obs, act, rew, fin, /, *, hx, virtual):
        ...
        return actions_to_take, hx_update, auxiliary_info
```


## Installation

Although the rollout collection engine core API is stable, overall this research project is currently in unstable pre-release stage. Thus it has not been published at PyPI, so no `pip install rlplay` (atm, or ever). If you want to install the package, please use:

```bash
# just install
pip install git+https://github.com/ivannz/rlplay.git

# clone and install
git clone https://github.com/ivannz/rlplay.git
cd rlplay

pip install -e .
```
