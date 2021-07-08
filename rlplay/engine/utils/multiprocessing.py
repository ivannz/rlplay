import cloudpickle
import numpy

from ctypes import c_byte
from .apply import suply

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


class PickleShared:
    """Pickle for numpy arrays with storage in shared memory.

    Details
    -------
    Subclassing numpy.ndarray is a bad idea. We can override `__reduce__` to
    make it aware of the shared array's special pickling, instead of array's
    default which forces a copy (because it doesn't own the storage). However,
    the result will not 100% compatible with a true array: the reduction
    mechanism has no legitimate way of knowing offsets into the underlying
    storage, which are necessary in certain types of views or slices (rather
    difficult or hacky to compute using `__array_interface__`).

    Instead we create a special object, the purpose of which is to let the
    shared storage handle its own pickling, then get passed to a subprocess,
    and finally rebuilt as a array.
    """
    __slots__ = 'shape', 'dtype', 'buffer'

    def __init__(self, shape, dtype, buffer):
        self.shape, self.dtype, self.buffer = shape, dtype, buffer

    @classmethod
    def empty(cls, *shape, dtype, ctx=multiprocessing):
        """Create an empty array of specified dtype and shape in shared memory."""
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        # let numpy handle the input dtype specs (`dtype.itemsize` is in bytes)
        dtype = numpy.dtype(dtype)

        # Although numpy's dtype typecodes mostly coincide with codes from
        #  python's std array, we allocate `nbytes` of `c_byte` shared memory
        n_bytes = int(numpy.prod(shape)) * dtype.itemsize

        return cls(shape, dtype, ctx.RawArray(c_byte, n_bytes))

    def numpy(self):
        """Rebuild the array with storage in the shared memory."""
        return numpy.ndarray(self.shape, dtype=self.dtype, buffer=self.buffer)

    @classmethod
    def from_numpy(cls, array, *leading, copy=True, ctx=multiprocessing):
        """Create a new shared array like the one given with extra leading dims."""
        shm = cls.empty(*leading, *array.shape, dtype=array.dtype, ctx=ctx)
        if copy:
            # temporarily rebuild to copy the data
            numpy.copyto(shm.numpy(), array, casting='no')
        return shm

    def __call__(self):
        return self.numpy()

    @classmethod
    def from_structured(cls, struct, *leading, ctx=multiprocessing):
        """Recursively allocate array in shared memory."""
        def _empty_like(npy):
            if isinstance(npy, numpy.ndarray):
                return cls.from_numpy(npy, *leading, ctx=ctx)
            raise TypeError(f'Unrecognized type `{type(npy)}`')

        return suply(_empty_like, struct)

    @classmethod
    def to_structured(cls, struct):
        """Recursively rebuild the shared structured data."""
        def _build(sh):
            if isinstance(sh, cls):
                return sh.numpy()
            raise TypeError(f'Unrecognized type `{type(sh)}`')

        return suply(_build, struct)
