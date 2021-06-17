import torch
import numpy

from ctypes import c_byte
from collections import namedtuple

from .base import apply_single


Aliased = namedtuple('Aliased', ['npy', 'pyt'])


# XXX we cannot have shared and pinned at the same time!
def torchify(obj, *leading, copy=False, pinned=False, shared=False):
    """Convert the values in the nested container into torch tensors.

    Parameters
    ----------
    obj: any
        The nested container (dict, list, tuple, or namedtuple) of data.

    leading: tuple of int
        The extra leading dimensions to prepend to the shape of the resulting
        tensors residing within the nested container.

    copy: bool, default=False
        Forces a copy even in the case when one is not necessary, which
        effectively disables torch-numpy memory aliasing altogether.

    pinned: bool, default=False
        The underlying storage of the newly created tensor resides in pinned
        memory (non-paged) for faster host-device transfers. If the original
        value was a numpy ndarray, then no aliasing is possible, and a copy
        is made.

        Cannot be used with `shared=True`.

    shared: bool, default=False
        Allocates the underlying storage of new tensors using torch's memory
        interprocess memory sharing logic which makes it so that all changes
        to the data are reflected between all processes. If the original value
        was a numpy ndarray, then no aliasing is possible, and a copy is made.

        Cannot be used with `pinned=True`.

    Returns
    -------
    obj: any
        The nested container with torch tensors, optionally shared or pinned.

    Details
    -------
    `torch.as_tensor` creates new tensors from lists of numeric data, and lets
    tensors and ndarrays through (no copy). The resulting torch tensor and
    the original numpy ndarray alias the same memory, hence changes to one are
    reflected in the other UNLESS a copy has been made due to pinned or shared
    flags.

    This function does not copy the original torch tensors or numpy arrays,
    UNLESS extra leading non-broadcasting dims were specified or moving to
    shared or pinned memory was requested. If a copy is made then memory
    aliasing between torch tensors and numpy arrays is IMPOSSIBLE.

    As of 1.8 torch doesn't support string data types.
    """
    if pinned and shared:
        raise ValueError('`pinned` and `shared` flags are mutually exclusive.')

    assert all(isinstance(j, int) for j in leading)

    # not is_broadcast => leading has at least one non unit dim => copy
    # pinned => copy, because it requires allocating in non-paged memory
    is_broadcast = all(j == 1 for j in leading)  # also True if not leading

    def _as_tensor(x):
        # alias numpy ndarrays, keep tensors intact, create new from scalars
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


def numpyfy(obj, *leading, copy=False, ctx=None):
    """Convert the values in the nested container into numpy arrays.

    Parameters
    ----------
    obj: any
        The nested container (dict, list, tuple, or namedtuple) of data.

    leading: tuple of int
        The extra leading dimensions to prepend to the shape of the resulting
        arrays residing within the nested container.

    copy: bool, default=False
        Forces a copy even in the case when one is not necessary, which
        effectively disables torch-numpy memory aliasing altogether.

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
    the latter case the resulting numpy ndarray and the original torch tensor
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
        # alias torch tensors, keep ndarrays intact, create new from scalars
        npy = x.numpy() if isinstance(x, torch.Tensor) else numpy.asarray(x)
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
        numpy.copyto(out, npy, casting='no')

        return out

    return apply_single(obj, fn=_as_array)


def alias(obj, *, pinned=False, shared=False, copy=False):
    """Make nested containers with aliased numpy ndarrays and torch tensors.

    Parameters
    ----------
    obj: any
        The nested container (dict, list, tuple, or namedtuple) of data.

    Returns
    -------
    obj: Aliased
        A named tuple with npy and pyt slots having identically structured
        nested containers of aliased numpy ndarrays and torch tensors.

    Details
    -------
    Prioritizes torchification, since it is more versatile than numpyfication.
    """
    assert not (pinned and shared)

    pyt = torchify(obj, pinned=pinned, shared=shared, copy=copy)
    return Aliased(npy=numpyfy(pyt), pyt=pyt)


# XXX `torchify` blocks `.base` tracking in `numpy.ndarray`, so we won't
#  be able to obtain the real underlying buffer to see if it is shared.
# >>> data = {'foo': True, 'bar': np.arange(10), 'baz': {'a': +1., 'z': -1.}}
# >>> pyt = torchify(data, 10, shared=True)
# >>> npy = numpyfy(pyt)  # makes an alias
# >>> # set up two processes with two barriers.
# >>> apply_single(npy, fn=lambda x: x.__setitem__(slice(None), 0))
# >>> pyt_two = torchify(npy)
# >>> assert pyt_two == pyt
