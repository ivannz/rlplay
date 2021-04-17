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
