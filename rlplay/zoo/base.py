import torch


class Duelling(torch.nn.Module):
    def __init__(self, *, value, advantage):
        super().__init__()
        self.value, self.advantage = value, advantage

    def forward(self, input):
        a, v = self.advantage(input), self.value(input)
        return a + (v - a.mean(dim=-1, keepdim=True))
