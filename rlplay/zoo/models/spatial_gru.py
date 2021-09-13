import torch

from torch import Tensor
from typing import Union, Tuple, Optional
from torch.nn.modules.conv import _pair

from torch.nn.modules.module import Module
from torch.nn.modules.conv import Conv2d


class GRUConv2dCell(Module):
    """a single GRU cell with spatially local hidden state dependencies.

    Details
    -------
    We don't just flatten spatial data and feed it to the GRU -- we want hidden
    state locality as well!

    Upon a very cursory inspection the most related work is
        [Shi et al. (2015)](https://arxiv.org/abs/1506.04214)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        hidden_kernel: Union[int, Tuple[int, int]],
        **kwargs,
        # XXX star-args are passed directly to Conv2d for the `input`
    ):
        # make sure the kenrel sizes are odd, and get the padding
        # XXX the only place where the `2d-ness` is hardcoded here!
        n_kernel = _pair(hidden_kernel)
        assert all(k & 1 for k in n_kernel)

        n_pad = [k >> 1 for k in n_kernel]  # stride == 1

        super().__init__()

        # input to reset and update gates, and the candidate state
        self.x_hrz = Conv2d(
            in_channels,
            3 * out_channels,
            *args,
            **kwargs,
        )

        # hidden state to reset and update gates
        self.h_rz = Conv2d(
            out_channels,
            2 * out_channels,
            n_kernel,
            stride=1,
            bias=False,
            padding=n_pad,
        )

        # hidden state to the candidate
        self.h_h = Conv2d(
            out_channels,
            out_channels,
            n_kernel,
            stride=1,
            bias=False,
            padding=n_pad,
        )

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # declare and init
        n_channels: int = self.h_h.out_channels

        # try to batch as many convs as possible
        rzh: Tensor = self.x_hrz(input)  # XXX adds the bias term
        if hx is None:
            # XXX unbound from the multilayer hx along dim=0
            hx = torch.zeros(
                len(rzh),
                n_channels,
                # XXX jit.script cannot statically infer the next shape :(
                *rzh.shape[2:],
                dtype=rzh.dtype,
                device=rzh.device,
            )
            # XXX can we make this more exposed to the user for more convenient
            #  init, perhaps or a learnable zero-th hidden state?

        # rz: Tensor, r: Tensor, z: Tensor, h: Tensor

        # r_t = \sigma(b_r + W_{xr} x_t + U_{hr} h_{t-1}), z_t -- ditto
        rz, hat = rzh.split([2 * n_channels, n_channels], dim=1)
        z, r = torch.sigmoid(rz + self.h_rz(hx)).chunk(2, dim=1)

        # \hat{h} = \tanh(b_h + W_{xh} x_t + U_{hh} (r_t \cdot h_{t-1}))
        # h_t = (1 - z_t) \cdot h_{t-1} + z_t \cdot \hat{h}
        hx = hx.lerp(torch.tanh(hat + self.h_h(r * hx)), z)
        # `u.lerp(v, t)` is the same as `(1 - t) * u + t * v`, but saves
        # some memory. It is an alias for `torch.lerp(u, v, t)`.
        # See LerpKernel.cpp#L49-53.

        # XXX can `.cat` the input and hx (channel-wise) for r-z and then
        # `.cat` the `r * hx` with the input to get the candidate `\hat{h}`.
        # This will take two convolutions altogether, but cannot be used for
        # non-unit seq-dim though, and does not support different input and
        # hidden kernel sizes.

        return hx
