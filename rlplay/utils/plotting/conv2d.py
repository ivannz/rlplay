import torch
import numpy as np

from .grid import make_grid
from .imshow import ImageViewer
from ..module import BaseModuleHook


class Conv2DViewer(BaseModuleHook):
    def __init__(self, module, *, normalize=True,
                 aspect=(16, 9), pixel=(1, 1)):
        super().__init__()

        self.feature_maps, self.normalize = {}, normalize
        self.aspect, self.pixel = aspect, pixel

        # immediately register self with the module
        for name, mod in module.named_modules():
            if isinstance(mod, (torch.nn.Conv2d,)):
                self.register(name, mod)

        self.viewers = {}

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

        # flatten the channel and batch dims and make grid
        self.feature_maps[label] = make_grid(
            output.cpu().reshape(-1, *spatial), aspect=self.aspect,
            pixel=self.pixel, normalize=self.normalize).numpy()

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


if __name__ == '__main__':
    pass
