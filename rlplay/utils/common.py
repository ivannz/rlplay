import torch

from math import sqrt, ceil


def linear(t, t0=0, t1=100, v0=1., v1=0.):
    tau = min(1., max(0., (t1 - t) / (t1 - t0)))
    return v0 * tau + v1 * (1 - tau)


@torch.no_grad()
def greedy(q_value, *, epsilon=0.5):
    """epsilon-greedy strategy."""
    q_action = q_value.max(dim=-1).indices

    random = torch.randint(q_value.shape[-1], size=q_action.shape,
                           device=q_value.device)
    is_random = torch.rand_like(q_value[..., 0]).lt(epsilon)
    return torch.where(is_random, random, q_action)


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

    # compute the geometry of the canvas
    (aw, ah), (pw, ph) = aspect, pixel
    ratio = (pw * width * ah) / (ph * height * aw)
    n_row = int(ceil(sqrt(ratio * n_images)))
    n_col = (n_images + n_row - 1) // n_row

    # create grayscale checkerboard
    lo, hi, r = tensor.min(), tensor.max(), 1e-1
    canvas = torch.full((n_row * height, n_col * width, ph, pw, *color),
                        lo + r * abs(lo), dtype=tensor.dtype)
    canvas[0::2, 1::2] = canvas[1::2, 0::2] = hi - r * abs(hi)

    # put the tensor onto the canvas: resahpe, copy, then undo reshape
    canvas = canvas.reshape(n_row, height, n_col, width, -1).swapaxes(1, 2)
    canvas = canvas.reshape(-1, height, width, ph * pw)

    canvas[:n_images] = tensor.unsqueeze(3)  # fake pixel dim

    canvas = canvas.reshape(n_row, n_col, height, width, ph, pw, *color)
    canvas = canvas.permute(0, 2, 4, 1, 3, 5, 6)
    canvas = canvas.reshape(n_row * height * ph, n_col * width * pw, *color)

    # eliminate the color dim if grayscale
    canvas.squeeze_(-1)

    # normalize to [0, 1] and store the grid
    if normalize:
        canvas.sub_(canvas.min())
        canvas.div_(canvas.max())

    return canvas
