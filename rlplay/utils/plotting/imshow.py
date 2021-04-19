import sys

import pyglet
import numpy as np

from random import randint

from pyglet import gl
from pyglet.image import ImageData
from pyglet.window import Window, key, mouse

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .grid import check_dims


def get_display(display):
    if display is None:
        return pyglet.canvas.get_display()

    elif isinstance(display, pyglet.canvas.Display):
        return display

    elif isinstance(display, str):
        return pyglet.canvas.Display(display)

    raise RuntimeError('`display` must be a Display object, an str, like ":0",'
                       f' or None for a default display. Got `{display}`.')


class ImageViewer(Window):
    """A window displaying an image or matplotlib figure.

    Parameters
    ----------
    caption : str, or None
        The caption of the viewer window. Defaults to the current script
        filename if None.

    scale : float, or tuple, None
        Pixel scaling applied to the image data. A tuple `(W, H)` of positive
        numbers corresponds to WxH pixels, a single number `S` results in equal
        scaling and `SxS` pixels. `None` defaults to `1x1` pixels.

    display : str, or None
        Uses the primary display if `None`, otherwise puts the viewer on the
        specified display. `pyglet` only supports multiple displays on Linux.

    vsync : bool, default=False
        Set the vertical retrace synchronisation to remove flicker caused by
        the video display not keeping up with buffer flips, and displaying
        partially stale picture. Setting `vsync=True` results in flicker-free
        animation, but the rendering becomes blocking, as the window drawing
        routines wait for the video device to refresh. Setting it to False
        introduces tearing artifacts, but allows for accurate profiling and
        real-time response.
    """

    def __init__(self, caption=None, *, scale=None, display=None, vsync=False):
        self.scale = check_dims(scale or (1, 1), kind=(int, float))

        super().__init__(caption=caption, resizable=True, vsync=vsync,
                         display=get_display(display))

        # randomly displace the window
        pos_y = randint(0, max(self.screen.height - self.height, 0))
        pos_x = randint(0, max(self.screen.width - self.width, 0))
        self.set_location(self.screen.x + pos_x, self.screen.y + pos_y)

        # XXX maybe use FPSDisplay from `pyglet.window`

    def close(self):
        """Close the viewer window."""
        # avoid `Python is likely shutting down` ImportError raised by cocoa
        if not sys.is_finalizing():
            super().close()

    def on_close(self):
        """User pressed the `X` button on the window to close it."""
        super().on_close()
        self.close()  # super().__del__() also invokes this method

    @property
    def isopen(self):
        """Flag indicating if a viewer window is open."""
        return not self._was_closed  # self.close() sets this flag

    def on_key_press(self, symbol, modifiers):
        """User pressed `SPACE` to fit viewer to the current image's dims."""
        if hasattr(self, 'texture') and symbol == key.SPACE:
            sw, sh = self.scale
            self.set_size(self.texture.width * sw, self.texture.height * sh)
            return

        super().on_key_press(symbol, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Move the window around when the user drags it with a mouse."""
        if buttons & mouse.LEFT:
            x, y = self.get_location()
            self.set_location(x + dx, y - dy)
            return

        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_draw(self):
        """Redraw the current image."""
        # the pyglet.window docs specify:
        # > The window will already have the GL context, so there is no need to
        # > call `.switch_to`. The window’s `.flip` method will be called after
        # > this event, so your event handler should not.
        self.clear()

        if hasattr(self, 'texture'):
            # do not interpolate when resizing textures
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D,
                gl.GL_TEXTURE_MAG_FILTER,
                gl.GL_NEAREST
            )

            # draw the image data into the window's buffer as a texture
            # blit transforms the texture data only virtually: the texture's
            # `.width` and `.height` dims still retain original array's shape.
            self.texture.blit(0, 0, width=self.width, height=self.height)

    def imshow(self, data, *, keepdims=True):
        """Show the monochrome, RGB or RGBA image data, or a matplotlib Figure.

        Parameters
        ----------
        data : numpy.ndarray, or matplotlib.figure.Figure
            The data to display in the viewer.
            * If `data` is a matplotlib Figure, that has already been composed
              elsewhere, then it gets rendered into a 3d RGBA array and
              displayed as color image.
            * If `data` is a numpy array, then it must either be a 2d or a 3d
              array for a grayscale or color image, respectively. In the color
              image case the last dim of the array must have size `3` for an
              RGB image or `4` for and RGBA image (RGB with an alpha channel).

        keepdims : bool, default=True
            Whether to preserve the current dimensions of the viewer, or resize
            it to fit the width and height of the current image.

        Details
        -------
        To prevent memory leaks it is advisable to close the figure afterwards,
        using `plt.close(fig)`. However, creating a new figure and then closing
        it appears to incur excessive overhead. One possible workaround is to
        create a figure and axes only once, and, when updating it, recreate its
        contents on the axes before and clear them after any call to `.imshow`.
        In general matplotlib is not optimized for live redrawing, but lowering
        `figsize` and `dpi` may improve fps.
        """

        # use the `agg` backend directly to create RGBA arrays from figures
        # https://matplotlib.org/stable/gallery/user_interfaces/canvasagg.html
        if isinstance(data, Figure):
            # assume all necessary plotting and artistry has been done
            canvas = FigureCanvasAgg(data)  # XXX does not seem to leak mem
            # render into a buffer and convert to numpy array
            canvas.draw()
            data = np.array(canvas.buffer_rgba())

        if not isinstance(data, np.ndarray):
            raise TypeError(f'`data` must be either a numpy array or a'
                            f' matplotlib `Figure`. Got `{type(data)}`.')

        # figure out the image format
        if not data.dtype == np.uint8:
            raise TypeError(f'`data` must be `np.unit8`. Got `{data.dtype}`.')

        height, width, *channels = data.shape
        if not channels:  # grayscale image
            format = 'I'

        elif len(channels) == 1:  # color image optionally with alpha channel
            assert channels[0] in (3, 4)
            format = 'RGBA' if channels[0] == 4 else 'RGB'

        else:
            raise TypeError(f'`data` is not an image `{data.shape}`.')

        # ensure the current window's gl context before manipulating textures
        self.switch_to()

        # convert image data to gl texture
        texture = ImageData(
            width, height, format, data.tobytes(),
            pitch=-width * len(format)).get_texture()

        # resize if instructed or on the first call to `.imshow`
        if not hasattr(self, 'texture') or not keepdims:
            sw, sh = self.scale
            self.set_size(texture.width * sw, texture.height * sh)
        self.texture = texture

        # handle events and draw the data
        return self.render()

    def render(self):
        """One pass of the event loop: respond to events and redraw."""
        # ensure the proper gl context before dispatching and drawing
        self.switch_to()

        # poll the os event queue and call the attached handlers
        self.dispatch_events()

        # manually call on-draw make it immediate
        self.on_draw()

        # swap the OpenGL front and back buffers.
        self.flip()

        # in event loops it is useful to know if the window still exists
        return self.isopen


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    # create a static image with custom pixel scaling
    unused = ImageViewer('Static Image', scale=(2, 4))
    unused.imshow(np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8))

    viewer = ImageViewer('Dynamic Image')
    while viewer.isopen:
        # change the contents, and handle window's events to make it responsive
        viewer.imshow(np.random.randint(0, 255, size=(128, 128), dtype=np.uint8))

        # just process window's events
        unused.render()

        time.sleep(0.01)

    # cannot reuse the same viewer window
    viewer = ImageViewer('Matplotlib Figure', scale=(2, 2), vsync=True)

    t0, t = 0., np.linspace(0, 1, num=10001)
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=120)
    while viewer.isopen:
        ax.set_title(f'matplotlib example {t0:.1e}', fontsize='small')
        ax.plot(t, np.sin((t0 + t) * np.pi * .2 * 35))
        ax.plot(t, np.sin((t0 + t) * np.pi * .2 * 21))
        ax.plot(t, np.sin((t0 + t) * np.pi * .2 * 7))

        viewer.imshow(fig)
        ax.clear()
        t0 += 1e-2

    plt.close(fig)
