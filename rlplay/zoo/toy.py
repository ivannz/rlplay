import torch

from collections import OrderedDict
from .base import Duelling


class DuellingQNet(torch.nn.Module):
    def __new__(cls, shape, n_outputs):
        return torch.nn.Sequential(
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(torch.Size(shape).numel(), 256, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256, bias=True),
            torch.nn.ReLU(),
            Duelling(
                value=torch.nn.Sequential(
                    torch.nn.Linear(256, 128, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 1, bias=True),
                ),
                advantage=torch.nn.Sequential(
                    torch.nn.Linear(256, 128, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, n_outputs, bias=True)
                ),
            ),
        )


class ToyQNet(torch.nn.Module):
    def __new__(cls, shape, n_outputs):
        return torch.nn.Sequential(
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(torch.Size(shape).numel(), 32, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(32, n_outputs, bias=True)
        )


class BreakoutQNet(torch.nn.Module):
    def __new__(cls, n_outputs):
        return torch.nn.Sequential(OrderedDict([
            ('conv_0', torch.nn.Conv2d(4, 32, 8, 4, bias=True)),
            ('relu_0', torch.nn.ReLU()),
            ('bnorm0', torch.nn.BatchNorm2d(32, affine=False)),
            # ('drop_1', torch.nn.Dropout(p=0.5)),
            ('conv_1', torch.nn.Conv2d(32, 64, 4, 2, bias=True)),
            ('relu_1', torch.nn.ReLU()),
            ('bnorm1', torch.nn.BatchNorm2d(64, affine=False)),
            # ('drop_2', torch.nn.Dropout(p=0.2)),
            ('conv_2', torch.nn.Conv2d(64, 64, 3, 1, bias=True)),
            ('relu_2', torch.nn.ReLU()),
            ('bnorm2', torch.nn.BatchNorm2d(64, affine=False)),
            # ('drop_3', torch.nn.Dropout(p=0.5)),
            ('flat_3', torch.nn.Flatten(1, -1)),
            ('dense3', torch.nn.Linear(64*7*7, 512, bias=True)),
            ('relu_3', torch.nn.ReLU()),
            ('bnorm3', torch.nn.BatchNorm1d(512, affine=False)),
            # ('drop_4', torch.nn.Dropout(p=0.2)),
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
