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

    def __getattr__(self, name):
        if not name.startswith('on_'):
            raise AttributeError(name)

        return lambda *a, **k: None

    def imshow(self, *args, **kwargs):
        pass

    def render(self):
        pass


class Conv2DViewer:
    def __init__(self, *args, **kwargs):
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    def register(self, label, module):
        pass

    def toggle(self, mode=None):
        if mode is None:
            mode = not self._enabled

        old_mode = self._enabled
        self._enabled = bool(mode)
        return old_mode

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.draw()

    def exit(self):
        self.close()

    def __iter__(self):
        return []

    @property
    def isopen(self):
        return not self._was_closed

    def close(self):
        self._was_closed = True

    def draw(self):
        self._was_closed = False
