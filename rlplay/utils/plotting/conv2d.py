import torch
import numpy as np

from .grid import make_grid
from .imshow import ImageViewer
from ..module import BaseModuleHook


class Conv2DViewer(BaseModuleHook):
    def __init__(self, module, activation=None, *, normalize='independent',
                 tap=torch.nn.Conv2d, aspect=(16, 9), pixel=(1, 1)):
        assert normalize in ('full', 'independent', 'none')
        super().__init__()

        self.feature_maps, self.normalize = {}, normalize
        self.aspect, self.pixel = aspect, pixel

        # immediately register self with the module
        for name, mod in module.named_modules():
            if isinstance(mod, tap):
                self.register(name, mod)

        self.viewers = {}

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
        if self.normalize != 'independent':
            normalize = self.normalize == 'full'
        else:
            normalize = False

            # flatten and 0-1 normalize each image independently
            flat = tensor.flatten(1, -1).clone()
            flat = flat.sub_(flat.min(dim=-1, keepdim=True).values)
            upper = flat.max(dim=-1, keepdim=True).values
            flat = flat.div_(upper.clamp_(min=1e-4))

            # rebuild images
            tensor = flat.reshape(-1, *spatial)

        # make a grid of images with a checkerboard placeholder
        self.feature_maps[label] = make_grid(
            tensor, aspect=self.aspect, pixel=self.pixel,
            normalize=normalize).numpy()

    def __iter__(self):
        yield from self.feature_maps.items()

    @property
    def isopen(self):
        """Check if any viewers are open."""
        return any(v.isopen for v in self.viewers.values())

    def close(self):
        """Close all viewers."""
        for viewer in self.viewers.values():
            viewer.close()
        self.viewers.clear()

    def draw(self):
        """Draw the most recently collected outputs."""
        # create as many Viewers are there are convolutional layers
        for label, image in self:
            if label not in self.viewers:
                self.viewers[label] = ImageViewer(
                    caption=f'Feature maps of `{label}`')

            # update the image data
            self.viewers[label].imshow((image * 255).astype(np.uint8))

    def __exit__(self, exc_type, exc_value, traceback):
        """Draw on exit."""
        super().__exit__(exc_type, exc_value, traceback)
        self.draw()


if __name__ == '__main__':
    pass
