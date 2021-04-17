import torch
from torch.nn import Conv2d, Module


class Duelling(torch.nn.Module):
    """Duelling q-net head from
        [Wang et al. (2016)](http://arxiv.org/abs/1511.06581)
    """
    def __init__(self, *, value, advantage):
        super().__init__()
        self.value, self.advantage = value, advantage

    def forward(self, input):
        a, v = self.advantage(input), self.value(input)
        return a + (v - a.mean(dim=-1, keepdim=True))


class PixelAttnConv(Module):
    """Pixel attention from
        [Zhao et al. (2021)](https://arxiv.org/abs/2010.01073) fig. 3
    """
    def __init__(self, n_channels, kernel=3):
        super().__init__()
        self.k1 = Conv2d(n_channels, n_channels, 1, stride=1, bias=True)
        self.k2 = Conv2d(n_channels, n_channels, kernel, stride=1,
                         padding=(kernel - 1)//2, bias=True)
        self.k3 = Conv2d(n_channels, n_channels, kernel, stride=1,
                         padding=(kernel - 1)//2, bias=True)

    def forward(self, input):
        attn = self.k1(input).sigmoid()
        return self.k3(attn.mul(self.k2(input)))
