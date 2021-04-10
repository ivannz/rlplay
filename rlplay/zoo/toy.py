import torch

from .base import Duelling


class QNet(torch.nn.Module):
    def __new__(cls, shape, n_outputs):
        return torch.nn.Sequential(
            torch.nn.Flatten(1, -1),
            torch.nn.Linear(torch.Size(shape).numel(), 512, bias=True),
            torch.nn.ReLU(),
            # torch.nn.Linear(512, 512, bias=True),
            # torch.nn.ReLU(),
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
