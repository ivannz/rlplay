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

    def clear(self):
        if self.isopen:
            self.close()
        self.feature_maps.clear()
        super().clear()

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

    def close(self):
        self.viewers.clear()

    @property
    def isopen(self):
        return any(v.isopen for v in self.viewers.values())

    def draw(self):
        # create as many Viewers are there are convolutional layers
        for label, image in self:
            if label not in self.viewers:
                viewer = ImageViewer(caption=f'Feature maps of `{label}`')
                viewer.imshow((image * 255).astype(np.uint8))
                self.viewers[label] = viewer

        self.render()

    def render(self):
        for label, viewer in self.viewers.items():
            viewer.render()


if __name__ == '__main__':
    pass
