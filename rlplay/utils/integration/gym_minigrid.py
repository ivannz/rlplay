# minigrid hotfix
import importlib
from functools import wraps

from ..plotting import ImageViewer


def MiniGridEnv_render(
    self,
    mode='none',
    **kwargs,
):
    # patch the human mode rendering logic
    if mode != 'human':
        return self.unpatched_render(mode=mode, **kwargs)

    viewer_ = getattr(self, 'viewer', None)
    if not isinstance(viewer_, ImageViewer) or not viewer_.isopen:
        self.viewer = ImageViewer(type(self).__name__, scale=(2, 2))

    return self.viewer.imshow(
        self.unpatched_render(mode='none', **kwargs)
    )


def patch_MiniGridEnv_render():
    minigrid = importlib.import_module('gym_minigrid.minigrid')
    fn = minigrid.MiniGridEnv.unpatched_render = minigrid.MiniGridEnv.render

    minigrid.MiniGridEnv.render = wraps(fn)(MiniGridEnv_render)


patch_MiniGridEnv_render()
