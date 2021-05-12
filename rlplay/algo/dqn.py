import torch

from torch.nn.functional import smooth_l1_loss


@torch.enable_grad()
def loss(batch, *, module, target, gamma=0.95, double=True, weights=None):
    r"""Compute the Double-DQN loss.

    Parameters
    ----------
    batch : dict of tensor
        The batch of state-action-reward transitions $
            s_t, a_t, r_{t+1}, s_{t+1}, T_{t+1}
        $ where $
            s_{t+1}, r_{t+1} \sim p(s, r \mid s_t, a_t)
        $ and $T_{t+1}$ indicates whether $s_{t+1}$ is terminal.

    module : torch.nn.Module
        The current Q-network.

    target : torch.nn.Module or None
        The target Q-network. If `None` then the current network is used and
        in this case `double` is forced to `False`.
        No backprop takes place via the target network.

    gamma : float, default=0.95
        The discount factor in the reward stream present value. Must be in
        `[0, 1)`.

    double : bool, default=True
        Whether to use Double DQN or not. See details.

    weights : torch.Tensor or None
        The weight associated with each transition in the batch. Assumed uniform
        if `None`.

    Details
    -------
    In Q-learning the action value function minimizes the TD-error
    $$
        r_{t+1} + \gamma v^*(s_{t+1}) 1_{\neg T_{t+1}}
            - Q(s_t, a_t; \theta)
        \,, $$
    w.r.t. Q-network parameters $\theta$. If classic Q-learning there is no
    target network and the next state optimal value function is bootstrapped
    using the current Q-network (`module`):
    $$
        v^*(s_{t+1})
            \approx \max_a Q(s_{t+1}, a; \theta)
        \,. $$
    The DQN method, proposed by
        [Minh et al. (2013)](https://arxiv.org/abs/1312.5602),
    uses a secondary Q-network to estimate the value of the next state:
    $$
        v^*(s_{t+1})
            \approx \max_a Q(s_{t+1}, a; \theta^-)
        \,, $$
    where $\theta^-$ are frozen parameters of the Q-network (`target`). The
    Double DQN algorithm of
        [van Hasselt et al. (2015)](https://arxiv.org/abs/1509.06461)
    unravels the $\max$ operator as $
        \max_k u_k
            \equiv u_{\arg \max_k u_k}
    $ and replaces the outer $u$ with the Q-values of the target net, while
    the inner $u$ (inside the $\arg\max$) is computed with the Q-values of the
    current Q-network. Specifically, the Double DQN value estimate is
    $$
        v^*(s_{t+1})
            \approx Q(s_{t+1}, \hat{a}_{t+1}; \theta^-)
        \,, $$
    for $
        \hat{a}_{t+1}
            = \arg \max_a Q(s_{t+1}, a; \theta)
    $ being the action taken by the current Q-network $\theta$ at $s_{t+1}$.

    Replay Buffer
    -------------
    `replay` is a finite generator of structured batches, each being a dict
    with the following schema (these are the required fields):
         'state' : Tensor [batch x *space] of `float32`
        'action' : Tensor [batch x *space] of `long`
        'reward' : Tensor [batch] of `float32`
    'state_next' : Tensor [batch x *space] of `float32`
          'done' : Tensor [batch] of `bool`
          'info' : dict

    `weights` is expected to be None or a Tensor of shape [batch] of `float32`
    """

    if target is None:
        # use Q(\cdot; \theta) instead of Q(\cdot; \theta^-)
        target, double = module, False

    # get Q(s_t, a_t; \theta)
    q_replay = module(batch['state']).gather(-1, batch['action'].unsqueeze(-1))

    # make sure no grads are computed for the target Q-value
    with torch.no_grad():
        # get Q(s_{t+1}, \cdot; \theta^-)
        q_target = target(batch['state_next'])
        if double:
            # get \hat{a} = \arg \max_a Q(s_{t+1}, a; \theta)
            hat_a = module(batch['state_next']).max(dim=-1).indices

            # get \hat{q} = Q(s_{t+1}, \hat{a}; \theta^-)
            q_value = q_target.gather(-1, hat_a.unsqueeze(-1))

        else:
            # get \max_a Q(s_{t+1}, a; \theta^-)
            q_value = q_target.max(dim=-1, keepdim=True).values

        # mask terminal states and get $r_t + \gamma \hat{q}_t 1_{T_t}$
        q_value.masked_fill_(batch['done'].unsqueeze(-1), 0.)
        q_value.mul_(gamma).add_(batch['reward'].unsqueeze(-1))  # inplace ops

        # compute the Temp. Diff. error (δ) for experience prioritization
        td_error = q_replay - q_value
    # end with

    # the (weighted) td-error loss
    if weights is None:
        loss = smooth_l1_loss(q_replay, q_value, reduction='mean')
        return loss, {'td_error': td_error}

    values = smooth_l1_loss(q_replay, q_value, reduction='none')
    return weights.mul(values).mean(), {'td_error': td_error}
