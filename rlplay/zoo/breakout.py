import torch
from collections import OrderedDict

from torch.nn import Module, Sequential
from torch.nn import Linear, Identity, Flatten

from torch.nn import ReLU
from torch.nn import Conv2d

from torch.nn import BatchNorm1d, BatchNorm2d, Dropout

from .base import Duelling


class BreakoutQNet(Module):
    def __new__(cls, n_outputs):
        return Sequential(OrderedDict([
            ('conv_0', Conv2d(4, 32, 8, 4, bias=False)),
            ('bnorm0', BatchNorm2d(32, affine=True)),
            ('relu_0', ReLU()),
            ('tap_0', Identity()),  # nop tap for viewer

            # ('drop_1', Dropout(p=0.5)),
            ('conv_1', Conv2d(32, 64, 4, 2, bias=False)),
            ('bnorm1', BatchNorm2d(64, affine=True)),
            ('relu_1', ReLU()),
            ('tap_1', Identity()),  # nop tap for viewer

            # ('drop_2', Dropout(p=0.2)),
            ('conv_2', Conv2d(64, 64, 3, 1, bias=False)),
            ('bnorm2', BatchNorm2d(64, affine=True)),
            ('relu_2', ReLU()),
            ('tap_2', Identity()),  # nop tap for viewer

            ('flat_3', Flatten(1, -1)),
            # ('drop_3', Dropout(p=0.5)),
            ('dense3', Linear(64*7*7, 512, bias=False)),
            ('bnorm3', BatchNorm1d(512, affine=True)),
            ('relu_3', ReLU()),

            # ('drop_4', Dropout(p=0.2)),
            ('dense4', Linear(512, n_outputs, bias=True)),
        ]))


class BreakoutQNet_43984a7(torch.nn.Module):
    def __new__(cls, n_outputs):
        return torch.nn.Sequential(OrderedDict([
            ('conv_0', torch.nn.Conv2d(4, 32, 8, 4, bias=False)),
            ('bnorm0', torch.nn.BatchNorm2d(32, affine=True)),
            ('relu_0', torch.nn.ReLU()),
            ('tap_0', torch.nn.Identity()),  # nop tap for viewer

            # ('drop_1', torch.nn.Dropout(p=0.5)),
            ('conv_1', torch.nn.Conv2d(32, 64, 4, 2, bias=False)),
            ('bnorm1', torch.nn.BatchNorm2d(64, affine=True)),
            ('relu_1', torch.nn.ReLU()),
            ('tap_1', torch.nn.Identity()),  # nop tap for viewer

            # ('drop_2', torch.nn.Dropout(p=0.2)),
            ('conv_2', torch.nn.Conv2d(64, 64, 3, 1, bias=False)),
            ('bnorm2', torch.nn.BatchNorm2d(64, affine=True)),
            ('relu_2', torch.nn.ReLU()),
            ('tap_2', torch.nn.Identity()),  # nop tap for viewer

            ('flat_3', torch.nn.Flatten(1, -1)),
            ('drop_3', torch.nn.Dropout(p=0.5)),
            ('dense3', torch.nn.Linear(64*7*7, 512, bias=False)),
            # ('bnorm3', torch.nn.BatchNorm1d(512, affine=True)),
            ('relu_3', torch.nn.ReLU()),

            ('drop_4', torch.nn.Dropout(p=0.2)),
            ('dense4', torch.nn.Linear(512, n_outputs, bias=True)),
        ]))


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
