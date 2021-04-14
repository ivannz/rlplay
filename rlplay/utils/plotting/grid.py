import torch

from math import sqrt, ceil


def make_grid(tensor, *, aspect=(16, 9), pixel=(1, 1), normalize=True):
    """Another version of `torchvision.utils.make_grid`"""

    # validate args
    if isinstance(aspect, float):
        aspect = aspect, 1
    assert all(isinstance(a, (float, int)) and a > 0 for a in aspect)

    if isinstance(pixel, int):
        pixel = pixel, pixel
    assert all(isinstance(p, int) and p > 0 for p in pixel)

    # the tensor is either grayscale or rgb
    assert tensor.dim() in (3, 4)
    n_images, height, width, *color = tensor.shape
    color = (1,) if not color else color

    # compute the geometry of the canvas: aspect = width : height
    (aw, ah), (pw, ph) = aspect, pixel
    ratio = (pw * width * ah) / (ph * height * aw)
    n_row = int(ceil(sqrt(ratio * n_images)))
    n_col = (n_images + n_row - 1) // n_row

    # create grayscale checkerboard on the host ('cpu')
    lo, hi, r = float(tensor.min()), float(tensor.max()), 1e-1
    canvas = torch.full((n_row * height, n_col * width, ph, pw, *color),
                        lo + r * abs(lo), dtype=tensor.dtype)
    canvas[0::2, 1::2] = canvas[1::2, 0::2] = hi - r * abs(hi)
    # `canvas` is [R x H] x [C x W] x ph x pw x k

    # put the tensor onto the canvas: resahpe, copy, then undo reshape
    canvas = canvas.reshape(n_row, height, n_col, width, -1)
    # `canvas` is R x H x C x W x [ph x pw x k]

    canvas = canvas.swapaxes(1, 2)
    # `canvas` is R x C x H x W x [ph x pw x k]

    canvas = canvas.reshape(n_row * n_col, height, width, -1)
    # `canvas` is [R x C] x H x W x [ph x pw x k]

    # copy_ transparently moves from a device to host
    canvas[:n_images].copy_(tensor.unsqueeze(3))  # fake pixel dim

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
