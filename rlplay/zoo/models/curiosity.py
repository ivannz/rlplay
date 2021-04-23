import torch
import torch.nn.functional as F

from torch.nn import Sequential
from torch.nn import Flatten, Identity
from torch.nn import Conv2d, Linear
from torch.nn import ELU, ReLU


class IntrinsicCuriosityModule(torch.nn.Module):
    """The Intrinsic Curiosity Module of
        [Pathak et al. (2017)](http://proceedings.mlr.press/v70/pathak17a.html)
    as described in the paper.
    """
    def __init__(self, n_actions=4, n_channels=4):
        super().__init__()

        self.phi = Sequential(
            # f_32 k_3 s_2 p_1
            Conv2d(n_channels, 32, 3, stride=2, padding=1, bias=True),
            ELU(),
            Conv2d(32, 32, 3, stride=2, padding=1, bias=True),
            ELU(),
            Conv2d(32, 32, 3, stride=2, padding=1, bias=True),
            ELU(),
            Conv2d(32, 32, 3, stride=2, padding=1, bias=True),
            ELU(),
            Identity(),  # tap
            Flatten(1, -1),
        )

        self.gee = torch.nn.Sequential(
            Linear(2*32*3*3, 256, bias=True),
            ReLU(),
            Linear(256, n_actions, bias=True),
        )

        self.eff = torch.nn.Sequential(
            Linear(32*3*3 + n_actions, 256, bias=True),
            ReLU(),
            Linear(256, 32*3*3, bias=True),
        )

        self.n_actions, self.n_emb_dim = n_actions, 32*3*3

    def inv_dyn(self, s_t, s_tp1):
        # join along batch dim, compute, then split
        batched = torch.cat([s_t, s_tp1], dim=0)
        h_t, h_tp1 = torch.chunk(self.phi(batched), 2, dim=0)  # undo cat

        # inverse dynamics inference
        logits = self.gee(torch.cat([h_t, h_tp1], dim=1))  # feature cat
        return h_t, h_tp1, logits

    def fwd_dyn(self, s_t, a_t, *, h_t=None):
        # compute state embedding if one was not provided
        if isinstance(s_t, torch.Tensor):
            h_t = self.phi(s_t)  # `s_t` overrides `h_t`

        # forward dynamics model
        hat_h_tp1 = self.eff(torch.cat([
            h_t, F.one_hot(a_t, self.n_actions)
        ], dim=1))

        return h_t, hat_h_tp1

    def forward(self, s_t, s_tp1, a_t, *, beta=0.2):
        # get the inverse and forward dynamics outputs
        h_t, h_tp1, logits = self.inv_dyn(s_t, s_tp1)
        h_t, hat_h_tp1 = self.fwd_dyn(None, a_t, h_t=h_t)

        # loss terms: cross entropy and mse
        ell_inv = F.cross_entropy(logits, a_t, reduction='mean')
        ell_fwd = F.mse_loss(hat_h_tp1, h_tp1, reduction='mean')

        # beta becomes dimensionless when scaled by the output dims
        loss = beta * ell_fwd * self.n_emb_dim + (1 - beta) * ell_inv

        # also return the probs (for tracking)
        return loss, {
            'fwd': F.mse_loss(hat_h_tp1.detach(), h_tp1.detach(),
                              reduction='none').sum(1),
            'inv': F.softmax(logits.detach(), dim=1),
        }
