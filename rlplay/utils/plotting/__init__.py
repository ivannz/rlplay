from .grid import make_grid

from .dummy import Conv2DViewer as DummyConv2DViewer
from .dummy import ImageViewer as DummyImageViewer

try:
    # pyglet fails to acquire a display and raises
    from .imshow import ImageViewer
    from .conv2d import Conv2DViewer

# pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"
except Exception:
    Conv2DViewer, ImageViewer = DummyConv2DViewer, DummyImageViewer
