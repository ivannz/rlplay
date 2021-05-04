"""Apply patches to gym's rendering utilities that address handling of GUI
events and open GL context initialization. Also enhances gym's ImageViewer
with zoom in/out controls and proper window GUI handling.
"""
import importlib

from ..plotting.imshow import get_display, ImageViewer


class SimpleImageViewer(object):
    def __init__(self, display=None, maxwidth=500):
        self.window = None
        self.display = get_display(display)
        self.maxwidth = maxwidth

    @property
    def isopen(self):
        try:
            return self.window.isopen

        except AttributeError:
            return False

    def imshow(self, arr):
        # create new viewer window with the specified size
        if self.window is None:
            self.window = ImageViewer(display=self.display)
            height, width = arr.shape[:2]
            if width > self.maxwidth:
                self.window.texture = None
                scale = self.maxwidth / width
                self.window.set_size(int(scale * width), int(scale * height))

            self.window.dispatch_events()

        # the same message as in the original implementation
        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"

        # render unless closed
        if self.isopen:
            self.window.imshow(arr, keepdims=True)

    def close(self):
        if self.isopen:
            self.window.close()

    def __del__(self):
        self.close()


def patch_Viewer_render():
    # import the lib into the global namespace, and ensure proper gl context
    rendering = importlib.import_module('gym.envs.classic_control.rendering')
    rendering.Viewer.unpatched_render = rendering.Viewer.render

    def render(self, return_rgb_array=False):
        # pyglet docs for `Window` state that the proper OpenGL context must
        # be activated with `.switch_to` before rendering. This includes calls
        # to `.clear`, the docs of which require the current gl context.
        self.window.switch_to()  # ensure the context

        return self.unpatched_render(return_rgb_array=return_rgb_array)

    rendering.Viewer.render = render


def patch_Viewer_on_close():
    # import the lib into the global namespace, and patch on-close handlers
    rendering = importlib.import_module('gym.envs.classic_control.rendering')

    # we reference the original method, since render's __init__ overwrties it
    rendering.Viewer.pyglet_on_close = staticmethod(
        rendering.pyglet.window.Window.on_close)

    rendering.Viewer.unpatched_on_close = \
        rendering.Viewer.window_closed_by_user

    def window_closed_by_user(self):
        """Patched handler that correctly handles GUI close event."""
        self.unpatched_on_close()
        self.pyglet_on_close(self.window)
        self.window.close()

    rendering.Viewer.window_closed_by_user = window_closed_by_user


def patch_SimpleImageViewer():
    # import the lib into the global namespace, and replace the ImageViewer
    rendering = importlib.import_module('gym.envs.classic_control.rendering')
    rendering.SimpleImageViewer = SimpleImageViewer


patch_Viewer_render()
patch_Viewer_on_close()
patch_SimpleImageViewer()
