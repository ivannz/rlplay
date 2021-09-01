import torch
import torch.nn.functional as F

from math import sqrt, ceil


def check_dims(first, second=None, *, kind=(float, int), positive=True):
    """"""
    if isinstance(first, kind):
        dims = first, (first if second is None else second)

    elif isinstance(first, (list, tuple)) and len(first) == 2:
        dims = first

    else:
        raise TypeError(f'Expected a number or a pair. Got `{first}`.')

    assert all(isinstance(p, kind) and p >= 0 for p in dims)
    if positive and any(p <= 0 for p in dims):
        raise ValueError(f'`{dims}` must be strictly positive.')

    return dims


def make_grid(tensor, *, aspect=(16, 9), pixel=(1, 1), pad=(0, 0), normalize=True):
    """Another version of `torchvision.utils.make_grid`"""

    # the tensor is either grayscale or rgb
    assert tensor.dim() in (3, 4)

    # add grayscale dim
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(3)

    # validate aspec and pixel scaling
    aspect, pixel = check_dims(aspect, 1), check_dims(pixel, kind=int)
    pad = check_dims(pad, kind=int, positive=False)

    # compute the range of values and optionally pad each image
    lo, hi, r = float(tensor.min()), float(tensor.max()), 1e-1
    lo, hi = lo + r * abs(lo), hi - r * abs(hi)

    # pad the input tensors, to make geometry eqn simpler
    ph, pw = pad
    if ph > 0 or pw > 0:
        padding = [ph, ph, pw, pw] + [(0, 0)] * (tensor.dim() - 3)
        tensor = F.pad(tensor, pad=padding, value=hi)

    # add fake pixel-size dims
    n_images, height, width, *color = tensor.shape
    tensor = tensor.reshape(n_images, height, width, 1, 1, *color)

    # compute the geometry of the canvas: aspect = width : height
    (aw, ah), (pw, ph) = aspect, pixel
    ratio = (pw * width * ah) / (ph * height * aw)
    n_row = int(ceil(sqrt(ratio * n_images)))
    n_col = (n_images + n_row - 1) // n_row

    # create grayscale checkerboard on the host ('cpu')
    canvas = torch.full((n_row * height, n_col * width, ph, pw, *color),
                        lo, dtype=tensor.dtype)
    canvas[0::2, 1::2] = canvas[1::2, 0::2] = hi
    # `canvas` is [R x H] x [C x W] x ph x pw x k

    # put the tensor onto the canvas: resahpe, copy, then undo reshape
    canvas = canvas.reshape(n_row, height, n_col, width, -1)
    # `canvas` is R x H x C x W x [ph x pw x k]

    canvas = canvas.swapaxes(1, 2)
    # `canvas` is R x C x H x W x [ph x pw x k]

    canvas = canvas.reshape(n_row * n_col, height, width, ph, pw, *color)
    # `canvas` is [R x C] x H x W x ph x pw x k

    # copy_ transparently moves from a device to host
    canvas[:n_images].copy_(tensor)

    canvas = canvas.reshape(n_row, n_col, height, width, ph, pw, *color)
    # `canvas` is R x C x H x W x ph x pw x k

    canvas = canvas.permute(0, 2, 4, 1, 3, 5, 6)
    # `canvas` is R x H x ph x C x W x pw x k

    canvas = canvas.reshape(n_row * height * ph, n_col * width * pw, *color)
    # `canvas` is [R x H x ph] x [C x W x pw] x k

    # eliminate the last (color) dim if it is one (aka grayscale)
    canvas.squeeze_(-1)

    # normalize to [0, 1] and store the grid
    if normalize:
        canvas.sub_(canvas.min())
        canvas.div_(canvas.max())

    return canvas
