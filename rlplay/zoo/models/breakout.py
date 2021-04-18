import torch
from collections import OrderedDict

from torch.nn import Module, Sequential
from torch.nn import Linear, Identity, Flatten

from torch.nn import ReLU
from torch.nn import Conv2d

from torch.nn import BatchNorm1d, BatchNorm2d, Dropout

from .base import Duelling


class BreakoutQNet(Module):
    def __new__(cls, n_outputs, *, batch_norm=True):
        layers = [
            ('conv_0', Conv2d(4, 32, 8, 4, bias=not batch_norm)),
            ('bnorm0', BatchNorm2d(32, affine=True)) if batch_norm else None,
            ('relu_0', ReLU()),
            ('tap_0', Identity()),  # nop tap for viewer

            # ('drop_1', Dropout(p=0.5)),
            ('conv_1', Conv2d(32, 64, 4, 2, bias=not batch_norm)),
            ('bnorm1', BatchNorm2d(64, affine=True)) if batch_norm else None,
            ('relu_1', ReLU()),
            ('tap_1', Identity()),  # nop tap for viewer

            # ('drop_2', Dropout(p=0.2)),
            ('conv_2', Conv2d(64, 64, 3, 1, bias=not batch_norm)),
            ('bnorm2', BatchNorm2d(64, affine=True)) if batch_norm else None,
            ('relu_2', ReLU()),
            ('tap_2', Identity()),  # nop tap for viewer

            ('flat_3', Flatten(1, -1)),
            # ('drop_3', Dropout(p=0.5)),
            ('dense3', Linear(64*7*7, 512, bias=not batch_norm)),
            ('bnorm3', BatchNorm1d(512, affine=True)) if batch_norm else None,
            ('relu_3', ReLU()),

            # ('drop_4', Dropout(p=0.2)),
            ('dense4', Linear(512, n_outputs, bias=True)),
        ]

        # filter out `None`s and build a sequential network
        return Sequential(OrderedDict(list(filter(None, layers))))


class DuellingBreakoutQNet(torch.nn.Module):
    def __new__(cls, n_outputs):
        return torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(64*7*7, 512, bias=True),
            torch.nn.Dropout(p=0.2),
            torch.nn.ReLU(),
            Duelling(
                value=torch.nn.Sequential(
                    torch.nn.Linear(512, 256, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 1, bias=True),
                ),
                advantage=torch.nn.Sequential(
                    torch.nn.Linear(512, 256, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, n_outputs, bias=True)
                ),
            ),
        )
