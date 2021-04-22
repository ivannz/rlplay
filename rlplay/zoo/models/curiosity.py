import torch
import torch.nn.functional as F

from torch.nn import Sequential
from torch.nn import Flatten
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

    def forward(self, s_t, s_tp1, a_t, *, beta=0.2):
        # stack along batch dim, pass, then split back
        # XXX make two independent passes is `phi` has batchnorms
        batched = torch.cat([s_t, s_tp1], dim=0)  # batch cat
        h_t, h_tp1 = torch.chunk(self.phi(batched), 2, dim=0)  # undo

        # inverse dynamics inference
        logits = self.gee(torch.cat([h_t, h_tp1], dim=1))  # feature cat

        # forward dynamics
        e_t = F.one_hot(a_t, self.n_actions)  # XXX `nn.Embedding` maybe?
        output = self.eff(torch.cat([h_t, e_t], dim=1))

        # loss terms: cross entropy and mse
        ell_inv = F.cross_entropy(logits, a_t, reduction='mean')

        ell_fwd = F.mse_loss(output, h_tp1,
                             reduction='mean') * self.n_emb_dim

        # also return the probs (for tracking)
        loss = beta * ell_inv + (1 - beta) * ell_fwd

        return loss, F.softmax(logits.detach(), dim=-1)
