
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


class ImageViewer:
    def __init__(self, *args, **kwargs):
        self._was_closed = False

    def close(self):
        self._was_closed = True

    def on_close(self):
        self.close()

    @property
    def isopen(self):
        return not self._was_closed

    def __bool__(self):
        return self.isopen

    def __getattr__(self, name):
        if not name.startswith('on_'):
            raise AttributeError(name)

        return lambda *a, **k: None

    def imshow(self, data, *, keepdims=True):
        if isinstance(data, Figure):
            canvas = FigureCanvasAgg(data)
            canvas.draw()
            data = np.array(canvas.buffer_rgba())

        self.data = data
        return self.render()

    def render(self):
        return self.isopen

    def read_buffer(self, format='rgb'):
        return self.data.copy()


class MultiViewer:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def isopen(self):
        return not self._was_closed

    def open(self, *captionless, **captioned):
        self._was_closed = False

    def on_key_press(self, symbol, modifiers, *, wr_viewer):
        pass

    def _cycle(self, wr_viewer):
        pass

    def close(self, *which):
        self._was_closed = True

    def imshow(self, **content):
        return self.isopen

    def refresh(self, *which):
        return self.isopen

    def __bool__(self):
        return self.isopen

    def __iter__(self):
        return []

    def __contains__(self, label):
        return True

    def __getitem__(self, label):
        return None

    def __setitem__(self, label, content):
        pass

    def __delitem__(self, label):
        pass


class Conv2DViewer:
    def __init__(self, *args, **kwargs):
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    def toggle(self, mode=None):
        if mode is None:
            mode = not self._enabled

        old_mode = self._enabled
        self._enabled = bool(mode)
        return old_mode

    def register(self, label, module):
        pass

    def exit(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.draw()

    def __del__(self):
        self.exit()

    def __iter__(self):
        return []

    @property
    def isopen(self):
        return not self._was_closed

    def close(self):
        self._was_closed = True

    def draw(self):
        self._was_closed = False
