import torch
import numpy

import multiprocessing as mp

from ctypes import c_byte
from collections import namedtuple

from .base import apply_single


Aliased = namedtuple('Aliased', ['npy', 'pyt'])
Aliased.__doc__ = """An aliased pair of numpy-torch data.

    Attributes
    ----------
    npy : any
        A nested container of numpy arrays assumed to have idenitcal heirachy
        to that of `.pyt` and aliasing the same data as the tensors therein.

    pyt : any
        A nested container of torch tensors assumed to have idenitcal heirachy
        to that of `.npy` and aliasing the same data as the arrays therein.

    Details
    -------
    numpy arrays and torch tensors can be cross initialized is such a way as
    to allow both to alias the same underlying data storage, so that every
    update made to one is reflected in the other. This is achieved through
    the support of numpy's array protocol (like PEP 3118) by torch's cpu
    tensors.
    """


def torchify(obj, *leading, copy=False, pinned=False, shared=False):
    """Convert the values in the nested container into torch tensors.

    Parameters
    ----------
    obj: any
        The nested container (dict, list, tuple, or namedtuple) of data.

    *leading: tuple of int
        The extra leading dimensions to prepend to the shape of the resulting
        tensors residing within the nested container.

    copy: bool, default=False
        Forces a copy even in the case when one is not necessary, which
        disables torch-numpy memory aliasing altogether, if the original data
        were a numpy array.

    pinned: bool, default=False
        The underlying storage of the newly created tensor resides in pinned
        memory (non-paged) for faster host-device transfers. If the original
        value was a numpy array, then no aliasing is possible, and a copy
        is made.

        Cannot be used with `shared=True`.

    shared: bool, default=False
        Allocates the underlying storage of new tensors using torch's memory
        interprocess memory sharing logic which makes it so that all changes
        to the data are reflected between all processes. If the original value
        was a numpy array, then no aliasing is possible, and a copy is made.

        Cannot be used with `pinned=True`.

    Returns
    -------
    obj: any
        The nested container with torch tensors, optionally shared or pinned.

    Details
    -------
    `torch.as_tensor` creates new tensors from lists of numeric data, and
    lets tensors and arrays through (no copy). The resulting torch tensor and
    the original numpy array alias the same memory, hence changes to one are
    reflected in the other UNLESS a copy has been made due to pinned or shared
    flags.

    This function does not copy the original torch tensors or numpy arrays,
    UNLESS extra leading non-broadcasting dims were specified or moving to
    shared or pinned memory was requested. If a copy is made then memory
    aliasing between torch tensors and numpy arrays is IMPOSSIBLE.

    As of 1.8 torch doesn't support string data types.

    Warning
    -------
    Torch automatically detects if a tensor is being shared and hot-swaps
    correctly allocated shared storage. Hence even when torchified with
    `shared=False` the nested container's tensors will be correctly shared
    between processes. However any numpy array aliases created prior to
    sharing will still reference the swapped-out invalidated torch's storage.
    So it is advisable to preemptively torchify the data within shared memory.
    """
    if pinned and shared:
        raise ValueError('`pinned` and `shared` flags are mutually exclusive.')

    assert all(isinstance(j, int) for j in leading)

    # not is_broadcast => leading has at least one non unit dim => copy
    # pinned => copy, because it requires allocating in non-paged memory
    is_broadcast = all(j == 1 for j in leading)  # also True if not leading

    def _as_tensor(x):
        # alias numpy arrays, keep tensors intact, create new from scalars
        pyt = torch.as_tensor(x)

        # warn that some data is on device, and cannot be aliased
        assert pyt.device == torch.device('cpu')

        # do not make a copy, unless we broadcast and unpinned memory is ok
        if not copy and not (pinned or shared) and is_broadcast:
            if leading:
                pyt = pyt.reshape(leading + pyt.shape)
            return pyt

        # allocate a new cpu tensor optionally in the pinned memory (non-paged)
        # XXX uses PinnedMemoryAllocator `/aten/src/ATen/utils.cpp#L44`
        # XXX also see `torch.Storage.pin_memory`
        out = torch.empty(leading + pyt.shape, dtype=pyt.dtype,
                          pin_memory=(pinned and not shared))

        out.copy_(pyt)  # chaining is less readable (`out.copy_` returns `out`)
        if shared:
            # move the storage to torch's shared memory (preserving the data)
            # this allocates shared storage, copies data to it, replaces the old storage 
            # XXX see `/torch/csrc/generic/StorageSharing.cpp#L76-L111`:shareFilename
            # as of 1.8 we cannot allocate new tensors in shared memory
            out.share_memory_()

        return out

    return apply_single(obj, fn=_as_tensor)


def numpify(obj, *leading, copy=False, ctx=None):
    """Convert the values in the nested container into numpy arrays.

    Parameters
    ----------
    obj: any
        The nested container (dict, list, tuple, or namedtuple) of data.

    *leading: tuple of int
        The extra leading dimensions to prepend to the shape of the resulting
        arrays residing within the nested container.

    copy: bool, default=False
        Forces a copy even in the case when one is not necessary, which
        disables torch-numpy memory aliasing altogether, if the original data
        were a torch tensor.

    ctx: multiprocessing context, default=None
        If `ctx` is not None, then allocate the newly created array in the
        shared memory (managed by the ctx multiprocessing context) and copy
        the original data into it.

        torch has slightly better developed memory sharing functionality,
        hence it is recommended to use `torchify` for sharing.

    Returns
    -------
    obj: any
        The nested container with numpy arrays, optionally in shared memory.

    Details
    -------
    We use `numpy.asarray` to create new array from lists of numeric or string
    data, leaving numpy arrays intact, and aliasing torch tensors as arrays. In
    the latter case the resulting numpy array and the original torch tensor
    reference the same underlying data storage, hence changes to one will be
    reflected in the other.

    This function does not copy the original torch tensors or numpy arrays,
    UNLESS extra leading non-broadcasting dims were specified or moving to
    shared memory was requested (for sharing between processes). If a copy is
    made then memory aliasing with corresponding torch tensors is DISABLED.
    """

    assert all(isinstance(j, int) for j in leading)
    # not is_broadcast => leading has at least one non unit dim => copy
    # shared => copy, because it requires allocating in shared memory
    is_broadcast = all(j == 1 for j in leading)

    def _as_array(x):
        # alias torch tensors, keep arrays intact, create new from scalars
        npy = numpy.asarray(x)
        if not copy and ctx is None and is_broadcast:
            if leading:
                npy = npy.reshape(leading + npy.shape)
            return npy

        # allocate `nbytes` of shared memory, or use numpy's default allocator
        n_bytes, buffer = int(numpy.prod(leading)) * npy.nbytes, None
        if ctx is not None:
            buffer = ctx.RawArray(c_byte, n_bytes)

        # create an uninitialized array and copy the original data into it
        out = numpy.ndarray(leading + npy.shape, dtype=npy.dtype, buffer=buffer)
        numpy.copyto(dst=out, src=npy, casting='no')

        return out

    return apply_single(obj, fn=_as_array)


def aliased(ref, *leading, copy=False, pinned=False, shared=False):
    """Make a pair of nested containers with aliased numpy arrays and torch
    tensors from the reference data.

    Parameters
    ----------
    ref: any
        The reference nested container (dict, list, tuple, or namedtuple)
        of data.

    *leading: tuple of int
        The extra leading dimensions to prepend to the shape of the resulting
        tensors residing within the nested container.

    copy: bool, default=False
        Force a copy of the original data in `ref` during torchification even
        if one is not necessary. This ONLY prevents aliasing the data in `ref`,
        not in the returned pair.

    pinned: bool, default=False
        The storage of the newly created tensors during torchification resides
        in pinned memory (non-paged) for faster host-device transfers. If the
        original data in `ref` is a numpy array, or a non-pinned tensor, then
        no aliasing is possible, and a copy is forced. This ONLY prevents
        aliasing the data in `ref`.

        Cannot be used with `shared=True`.

    shared: bool, default=False
        During torchification allocate the underlying storage of new tensors
        using torch's interprocess memory sharing logic, making it so that all
        changes to the tensor are reflected in ALL processes. If the original
        data in `ref` is a numpy array or a non-shared tensor, then no aliasing
        is possible, and a copy is forced. This ONLY prevents aliasing the data
        in `ref`.

        Cannot be used with `pinned=True`.

    Returns
    -------
    obj: Aliased
        A named tuple with `npy` and `pyt` slots having identically structured
        nested containers with numpy arrays and torch tensors aliasing the
        same memory. Whether they also alias the data in `ref` depends on
        the parameters.

    Details
    -------
    Prioritizes torchification, since it is more versatile than numpyfication.
    The array or tensor data in `ref` is always aliased if `shared`, `pinned`,
    and `copy` are all `False`.

    If `ref` contains ONLY torch tensors, then this is equivalent to

    >>> res1 = aliased(ref)
    >>> res2 = Aliased(numpify(ref), ref)
    >>> # res1.pyt is res2.pyt
    """
    assert not (pinned and shared)

    pyt = torchify(ref, *leading, pinned=pinned, shared=shared, copy=copy)
    return Aliased(npy=numpify(pyt), pyt=pyt)


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
    def empty(cls, *shape, dtype, ctx=mp):
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
    def from_numpy(cls, array, *leading, copy=True, ctx=mp):
        """Create a new shared array like the one given with extra leading dims."""
        shm = cls.empty(*leading, *array.shape, dtype=array.dtype, ctx=ctx)
        if copy:
            # temporarily rebuild to copy the data
            numpy.copyto(shm.numpy(), array, casting='no')
        return shm

    def __call__(self):
        return self.numpy()

    @classmethod
    def from_structured(cls, struct, *leading, ctx=mp):
        """Recursively allocate array in shared memory."""
        def _empty_like(npy):
            if isinstance(npy, numpy.ndarray):
                return cls.from_numpy(npy, *leading, ctx=ctx)
            raise TypeError(f'Unrecognized type `{type(npy)}`')

        return apply_single(struct, fn=_empty_like)

    @classmethod
    def to_structured(cls, struct):
        """Recursively rebuild the shared structured data."""
        def _build(sh):
            if isinstance(sh, cls):
                return sh.numpy()
            raise TypeError(f'Unrecognized type `{type(sh)}`')

        return apply_single(struct, fn=_build)
