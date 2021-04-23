import sys

import pyglet
import numpy as np

from weakref import ref

from random import randint
from functools import partial
from collections import OrderedDict

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

    resizable : bool, default=True
        Specify whether the image viewer has resizable window.

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

    def __init__(self, caption=None, *, scale=None, resizable=True,
                 display=None, vsync=False):
        self.scale = check_dims(scale or (1, 1), kind=(int, float))

        super().__init__(caption=caption, resizable=resizable,
                         vsync=vsync, display=get_display(display))

        # randomly displace the window
        pos_y = randint(0, max(self.screen.height - self.height, 0))
        pos_x = randint(0, max(self.screen.width - self.width, 0))
        self.set_location(self.screen.x + pos_x, self.screen.y + pos_y)

        # XXX maybe use FPSDisplay from `pyglet.window`
        self._init_scale = self.scale

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

    def __bool__(self):
        """Truth value testing amounts to checking if the window is open."""
        return self.isopen

    def on_key_press(self, symbol, modifiers):
        """Handle `-/_` and `+/=` to zoom in and out, or `0` to fit the viewer
        to the current image's dimensions.
        """
        # always fall back to the default handler
        if not hasattr(self, 'texture') or symbol not in {
            key._0, key.MINUS, key.PLUS, key.UNDERSCORE, key.EQUAL
        }:
            return super().on_key_press(symbol, modifiers)

        sw, sh = self._init_scale

        # reset to the scale set at initialization
        if symbol == key._0:
            self.scale = sw, sh

        # zoom out by 25% limited by the tenth of the original scaling factor
        elif symbol in (key.MINUS, key.UNDERSCORE):
            self.scale = max(sw / 10, self.scale[0] / 1.25), \
                         max(sh / 10, self.scale[1] / 1.25)

        # zoom in by 25% bounded by ten time the inital scaling
        elif symbol in (key.PLUS, key.EQUAL):
            self.scale = min(sw * 10, self.scale[0] * 1.25), \
                         min(sh * 10, self.scale[1] * 1.25)

        # fit the window to the image dims proportional to the pixel scaling
        sw, sh = self.scale
        self.set_size(self.texture.width * sw, self.texture.height * sh)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Move the window around when the user drags it with a mouse."""
        if buttons & mouse.LEFT:
            x, y = self.get_location()
            self.set_location(x + dx, y - dy)
            return

        super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)

    def on_resize(self, width, height):
        """Resize the window, with optional aspect lock."""
        if hasattr(self, 'aspect'):
            # maintain `W:H` aspect ratio based on width
            aw, ah = self.aspect
            height = int(width * ah / aw)

            # do the lower-level physical widow resizing (exhibits some janky
            #  `animated` resizing effects)
            self.set_size(width, height)

        # update the GL viewport and the dims of the orthogonal projection
        #  see `Window.on_resize` and `Window._projection` docs.
        super().on_resize(width, height)

    def on_draw(self):
        """Redraw the current image."""
        # the pyglet.window docs specify:
        # > The window will already have the GL context, so there is no need to
        # > call `.switch_to`. The window’s `.flip` method will be called after
        # > this event, so your event handler should not.
        # self.switch_to()

        # gl.glClearColor(0, 0, 0, 0)
        self.clear()

        if hasattr(self, 'texture'):
            # do not interpolate when resizing textures
            gl.glTexParameteri(
                gl.GL_TEXTURE_2D,
                gl.GL_TEXTURE_MAG_FILTER,
                gl.GL_NEAREST
            )

            # textures with alpha channels require gl blending capability
            gl.glEnable(gl.GL_BLEND)

            # Additive blending `R = S * F_s + D * F_d` for `c4f` f-RGBA
            gl.glBlendEquation(gl.GL_FUNC_ADD)

            # Factors `F_s = S_a` and `F_d = 1 - S_a` give transparency fx,
            #    >>> gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
            # but we set `F_s = 1.` and `F_d = 0.` for complete overwrite.
            # see https://learnopengl.com/Advanced-OpenGL/Blending
            #     and https://www.khronos.org/opengl/wiki/Blending
            #     and https://stackoverflow.com/a/18497511 (for opacity eqn)
            gl.glBlendFunc(gl.GL_ONE, gl.GL_ZERO)

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

    def read_buffer(self, format='rgb'):
        """Read back the recently rendered color data from the window's buffer.

        Returns
        -------
        data : unit8 array, shape=(height, width, n_channels)
            RGB or RGBA color data.

        Details
        -------
        Call to `.read_buffer()` should be preceded by `.render()` in order
        to read the latest color data. The dims of the returned color array
        correspond to the width and height of the gl color buffer used for
        rendering, and may not coincide with the dims of the original image.
        Typically, they correspond to the window dims scaled by some factor.
        """
        assert format in ('rgb', 'rgba')
        if format == 'rgb':
            format, n_channels = gl.GL_RGB, 3

        else:
            format, n_channels = gl.GL_RGBA, 4

        # ensure current gl context
        self.switch_to()

        # make sure the color data we read back is fresh
        # self.on_draw()
        # `.flip()` x2 may cause unnecessary vsync and flicker

        # allocate unit8 buffer for RGB data
        width, height = self.get_framebuffer_size()
        buf = (gl.GLubyte * (height * width * n_channels))(0)

        # read a block of pixels from the current display frame buffer
        gl.glReadBuffer(gl.GL_FRONT)  # FRONT -- display, BACK -- drawing
        gl.glReadPixels(0, 0, width, height, format, gl.GL_UNSIGNED_BYTE, buf)

        # properly reshape and reorder the data
        flat = np.frombuffer(buf, dtype=np.uint8)
        return flat.reshape(height, width, n_channels)[::-1].copy()


class MultiViewer:
    """A managed collection of image viewers.

    Parameters
    ----------
    resizable : bool, default=True
        Specify whether the viewers have resizable window.

    vsync : bool, default=False
        Set the vertical retrace synchronisation.

    display : str, or None
        Uses the primary display if `None`, otherwise puts the viewer on the
        specified display.
    """

    def __init__(self, *, scale=None, resizable=True,
                 vsync=False, display=None):
        self.resizable, self.vsync = resizable, vsync
        self.scale, self.display = scale, display
        self.viewers = OrderedDict()

    def open(self, *captionless, **captioned):
        """Open missing or reopen closed viewers."""
        captions = {**dict.fromkeys(captionless), **captioned}
        for label, caption in captions.items():
            if not isinstance(label, str):
                raise TypeError('Viewer labels must be '
                                f'str. Got `{label}`.')

            if not (caption is None or isinstance(caption, str)):
                raise TypeError('Captions must be either `None` '
                                f'or str. Got `{caption}`.')

            # open a new viewer if one doesn't exist or has been closed
            if label in self.viewers:
                if self.viewers[label].isopen:
                    continue

            viewer = ImageViewer(caption=caption,
                                 scale=self.scale, display=self.display,
                                 resizable=self.resizable, vsync=self.vsync)

            # weakref to the viewer baked into partial calls
            viewer.push_handlers(
                # `on_close` and `on_key_press` to activate the next viewer
                on_close=partial(self._cycle, wr_viewer=ref(viewer)),
                on_key_press=partial(self.on_key_press, wr_viewer=ref(viewer)),
            )

            self.viewers[label] = viewer

        return self

    def on_key_press(self, symbol, modifiers, *, wr_viewer):
        """Handle `tab` to cycle through the viewers."""
        if symbol == key.TAB:
            self._cycle(wr_viewer)

    def _cycle(self, wr_viewer):
        """Activate the next viewer if another one gets closed."""

        # the method is referenced by us and the caller
        viewer = wr_viewer()
        if viewer is None:
            return

        # find the next open and active viewer to activate
        for label, vw in self.viewers.items():
            if vw is not viewer and vw.isopen:
                # move-to-end to prevent alternating between two viewers
                self.viewers.move_to_end(label, last=True)
                return vw.activate()

    def close(self, *which):
        """Close all or only the specified viewers."""
        for label in list(which or self.viewers):
            if label in self.viewers:
                self.viewers[label].close()
                del self.viewers[label]

    def imshow(self, **content):
        """Display images on the specified viewers."""
        for label, image in content.items():
            if label not in self.viewers:
                self.open(label)

            # do not update closed viewers: check `isopen` since a viewer
            #  may exist in the dict, but still have been closed externally
            if self.viewers[label].isopen:
                self.viewers[label].imshow(image, keepdims=True)

        # refresh the left out viewers
        return self.refresh(*(self.viewers.keys() - content.keys()))

    def refresh(self, *which):
        """Refresh all open and active viewers."""
        labels = list(which or self.viewers)
        for label in labels:
            if label not in self.viewers:
                raise KeyError(f'Viewer `{label}` does not exist.')
            if self.viewers[label].isopen:
                self.viewers[label].render()

        return self.isopen

    @property
    def isopen(self):
        """Check if at least one viewer is open and active (alias)."""
        return any(viewer.isopen for viewer in self.viewers.values())

    def __bool__(self):
        """Check if at least one viewer is open and active."""
        return self.isopen

    def __getitem__(self, label):
        """Get the window of the specified viewer."""
        return self.viewers[label]


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
