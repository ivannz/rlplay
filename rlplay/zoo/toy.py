import torch

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
        return torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(64*7*7, 512, bias=True),
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
