import torch
import numpy as np

from .grid import make_grid
from .imshow import MultiViewer

from ..module import BaseModuleHook


class Conv2DViewer(BaseModuleHook):
    """Monitor the intermediate outputs of a module.

    Parameters
    ----------
    module : torch.nn.Module
        The module, intermediate outputs of which to monitor. The viewer
        attaches itself to the structure of the module using torch's API,
        which notifies the former of the new intermediate tensors and which
        submodules produced it.

    activation : callable, or dict of callable
        The extra nonlinearity applied to the monitored intermediate outputs.

    normalize : bool, default=True
        Specify whether the feature/channels of the tracked tensors are to
        be normalized individually or altogether.

    tap : torch.nn.Module, an str, or a sequence
        The layers, the outputs of which are tapped by the viewer.

    aspect : pair of numbers
        The width-height aspect ratio.

    viewer : None, or MultiViewer
        The multi-image viewer with which to display the monitored outputs.
        A new one is created if `None`.
    """
    def __init__(self, module, activation=None, *, normalize=True,
                 tap=torch.nn.Conv2d, aspect=(16, 9), viewer=None):
        super().__init__()

        # create own multi-image viewer instance
        if viewer is None:
            viewer = MultiViewer()
        assert isinstance(viewer, MultiViewer)

        self.feature_maps, self.viewers = {}, viewer
        self.aspect, self.normalize = aspect, normalize

        # immediately register self with the module
        if not isinstance(tap, tuple):
            tap = tap,

        *types, = map(lambda T: T if isinstance(T, type) else type(T), tap)
        if all(issubclass(t, torch.nn.Module) for t in types):
            # filter by layer class
            for name, mod in module.named_modules():
                if isinstance(mod, tap):
                    self.register(name, mod)

        elif all(issubclass(t, str) for t in types):
            # use string identifiers: match prefix
            for name, mod in module.named_modules():
                if name.startswith(tap):
                    self.register(name, mod)

        else:
            raise TypeError('`tap` tuple must be either all prefix strings'
                            f' or all `Module` subclasses. Got `{types}`.')

        # sort out the activations
        if callable(activation):
            activation = dict.fromkeys(self.labels, activation)

        elif activation is None:
            activation = {}

        assert isinstance(activation, dict)
        assert all(callable(act) for act in activation.values())

        # ensure mod-keyed dict, in case of string keys in `activation` dict
        mod_activation = {}
        label_to_module = dict(zip(self.labels.values(), self.labels.keys()))
        for mod, act in activation.items():
            if not isinstance(mod, torch.nn.Module):
                mod = label_to_module[mod]
            mod_activation[mod] = act

        assert mod_activation.keys() <= self.labels.keys()
        self.activation = mod_activation

    def exit(self):
        super().exit()
        self.feature_maps.clear()
        self.close()

    @torch.no_grad()
    def on_forward(self, label, mod, inputs, output):
        """Make a grid of grayscale 2d conv outputs."""
        assert isinstance(output, torch.Tensor)
        batch, channels, *spatial = output.shape
        assert batch == 1 and len(spatial) == 2

        # apply the specified activation
        activation = self.activation.get(mod)
        if callable(activation):
            output = activation(output)

        # flatten the channel and batch dims and apply normalization
        tensor = output.cpu().reshape(-1, *spatial)
        if self.normalize:
            # flatten and 0-1 normalize each image independently
            flat = tensor.flatten(1, -1).clone()
            flat = flat.sub_(flat.min(dim=-1, keepdim=True).values)
            upper = flat.max(dim=-1, keepdim=True).values
            flat = flat.div_(upper.clamp_(min=1e-4))

            # rebuild images
            tensor = flat.reshape(-1, *spatial)

        # make a grid of images with a checkerboard placeholder
        self.feature_maps[label] = make_grid(
            tensor, aspect=self.aspect,
            normalize=not self.normalize).numpy()

    def __iter__(self):
        yield from self.feature_maps.items()

    @property
    def isopen(self):
        """Check if any viewers are open."""
        return self.viewers.isopen

    def close(self):
        """Close all viewers."""
        self.viewers.close()

    def draw(self, *, vsync=False):
        """Draw the most recently collected outputs."""
        for label, image in self:
            if label not in self.viewers:
                # create a viewer with a proper caption
                self.viewers.get(label, f'Feature maps of `{label}`')

            # update our own viewer only
            self.viewers.imshow(label, (image * 255).astype(np.uint8))

    def __exit__(self, exc_type, exc_value, traceback):
        """Draw on exit."""
        super().__exit__(exc_type, exc_value, traceback)
        self.draw()


if __name__ == '__main__':
    """A simple illustration of how the Conv2dViewer can be used."""

    from collections import OrderedDict

    module = torch.nn.Sequential(OrderedDict([
        ('block_1', torch.nn.Sequential(OrderedDict([
            ('conv_1', torch.nn.Conv2d(32, 64, 5, 2)),
            ('relu_1', torch.nn.ReLU()),
        ]))),
        ('conv_2', torch.nn.Conv2d(64, 128, 3, 1)),
        ('relu_2', torch.nn.ReLU()),
    ]))

    print('Tapping layers with names strating with `conv_` or `block_`')
    viewer = Conv2DViewer(module, tap=('conv_', 'block_'))

    # do not track the intermediates outside `with` contexts
    viewer.toggle(False)

    single = torch.randn(1, 32, 28, 28)
    while True:
        with viewer:  # track this forward pass
            module(single)  # open the viewers on the first in-context call

        # the forward pass of this batch is not tracked:
        #  the viewer is hard coded to accept single element batches only.
        module(torch.randn(3, 32, 28, 28))
        if not viewer.isopen:
            break

    print('Tapping layers of type `torch.nn.Conv2d`')
    viewer = Conv2DViewer(module, tap=torch.nn.Conv2d)
    while True:
        with viewer:
            module(torch.randn(1, 32, 28, 28))

        if not viewer.isopen:
            break
