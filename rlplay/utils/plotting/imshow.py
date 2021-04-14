import pyglet
import numpy as np

from random import randint

from pyglet import gl
from pyglet.image import ImageData
from pyglet.window import Window, key


class ImageViewer(Window):
    """My own version of the image viewer.
    """
    def __init__(self, caption=None):
        super().__init__(caption=caption, resizable=True, vsync=False,
                         display=pyglet.canvas.get_display())

        # randomly displace the window
        pos_y = randint(0, max(self.screen.height - self.height, 0))
        pos_x = randint(0, max(self.screen.width - self.width, 0))
        self.set_location(self.screen.x + pos_x, self.screen.y + pos_y)

        # XXX maybe use FPSDisplay from `pyglet.window`

    def on_close(self):
        super().on_close()
        self.close()  # super().__del__() also invokes this method

    @property
    def isopen(self):
        return not self._was_closed  # self.close() sets this flag

    def on_key_press(self, symbol, modifiers):
        if hasattr(self, 'texture') and symbol == key.SPACE:
            self.set_size(self.texture.width, self.texture.height)
            return

        super().on_key_press(symbol, modifiers)

    def on_draw(self, *, flip=False):
        self.switch_to()
        self.clear()

        if hasattr(self, 'texture'):
            self.texture.blit(0, 0, width=self.width, height=self.height)

        if flip:
            self.flip()

    def imshow(self, data, *, resize=False):
        # figure out hte image format
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

        # ensure current window's gl context
        self.switch_to()

        # convert image data to gl texture
        gl.glTexParameteri(
            gl.GL_TEXTURE_2D,
            gl.GL_TEXTURE_MAG_FILTER,
            gl.GL_NEAREST
        )
        texture = ImageData(
            width, height, format, data.tobytes(),
            pitch=-width * len(format)).get_texture()

        # resize if instructed or on the first call to `.imshow`
        if resize or not hasattr(self, 'texture'):
            self.set_size(texture.width, texture.height)
        self.texture = texture

        self.render()

    def render(self):
        self.switch_to()

        # a piece of event loop
        self.dispatch_events()

        # manually call on-draw make it immediate
        self.on_draw(flip=True)


if __name__ == '__main__':
    import time

    viewer = ImageViewer()
    while viewer.isopen:
        viewer.imshow(np.random.randint(0, 255, size=(128, 128), dtype=np.uint8))
        time.sleep(0.01)
