import torch

from torch.nn.functional import smooth_l1_loss


@torch.enable_grad()
def loss(batch, *, module, target, gamma=0.95, double=True, weights=None):
    r"""Compute the Double-DQN loss.

    Details
    -------
    The DQN-target is given by $
        r_t + \gamma \max_a Q(s_{t+1}, a; \theta^-)
    $ where $\theta^-$ are frozen parameters of the Q-network (`target`).
    Now, the D-DQN algorithm of
        [van Hasselt et al. (2015)](https://arxiv.org/abs/1509.06461)
    unravels the $\max$ operator as $
        \max_k v_k
            \equiv v_{\arg \max_k v_k}
    $ and replaces the outer $v$ with the Q-values of the target net, while
    the inner $v$ (inside the $\arg \max$) is given by the Q-values of the
    under the current policy (`module`). Specifically, the Double DQN-target
    is $
        r_t + \gamma Q(s_{t+1}, \hat{a}_{t+1}; \theta^-)
    $ for $
        \hat{a}_{t+1} = \arg \max_a Q(s_{t+1}, a; \theta)
    $ being the action taken by the current `\theta` at $s_{t+1}$.

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

    # get Q(s_t, a_t; \theta)
    q_replay = module(batch['state']).gather(-1, batch['action'].unsqueeze(-1))
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
