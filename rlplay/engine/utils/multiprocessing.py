import cloudpickle
import torch.multiprocessing as multiprocessing

# import all essential multiprocessing tools that torch exports
from torch.multiprocessing import *

from torch.multiprocessing.spawn import spawn, start_processes
from torch.multiprocessing.spawn import SpawnContext, ProcessContext
from torch.multiprocessing.spawn import ProcessRaisedException, \
                                        ProcessExitedException

__all__ = [
    'get_context',
    'spawn',
    'start_processes',
    'CloudpickleSpawner',
]


__all__ += multiprocessing.__all__  # type: ignore[attr-defined]


def get_context(context):
    """Get the correct multiprocessing context (torch-friendly)"""
    from multiprocessing.context import BaseContext

    if isinstance(context, (str, bytes)) or context is None:
        context = multiprocessing.get_context(context)

    if not isinstance(context, BaseContext):
        raise TypeError(f'Unrecognized multiprocessing context `{context}`.')

    return context


class CloudpickleSpawner:
    def __init__(self, cls, *args, **kwargs):
        self.cls, self.args, self.kwargs = cls, args, kwargs

    def __call__(self):
        return self.cls(*self.args, **self.kwargs)

    def __getstate__(self):
        return cloudpickle.dumps(self.__dict__)

    def __setstate__(self, data):
        self.__dict__.update(cloudpickle.loads(data))
