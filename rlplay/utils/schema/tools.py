import torch
import numpy

from numbers import Number

from .base import apply_single, apply_pair

# from torch.utils.data._utils.collate import default_collate as torch_collate


def getitem(container, index):
    """Recursively get `element[index]` from a nested container."""
    def _getitem(elem):
        return None if elem is None else elem[index]

    return apply_single(container, fn=_getitem)


def setitem(container, index, value):
    """Recursively set `element[index] = value` in nested container.

    Details
    -------
    Skips elements or values that are `None`.
    """
    def _setitem(elem, val):
        if elem is None or val is None:
            return
        elem[index] = val

    apply_pair(container, value, fn=_setitem)


def getattribute(container, name):
    """Recursively get `element.name` from a nested container."""
    def _getattribute(elem):
        return getattr(elem, name, None)

    return apply_single(container, fn=_getattribute)


def method(container, name, *args, **kwargs):
    """Call a method on elements of the structured container, which have it.
    """

    def _call(elem):
        # return `None` if the requested method cannot be called
        attr = getattr(elem, name, None)
        if callable(attr):
            return attr(*args, **kwargs)

    return apply_single(container, fn=_call)


def gettype(container):
    """Get the data type of elements in nested structured containers."""
    def _dtype(elem):
        # types are returned as is
        if isinstance(elem, type):
            return elem

        # special elements
        elif isinstance(elem, (str, bytes, Number)):
            return type(elem)

        # tensors and ndarrays get special treatment
        elif isinstance(elem, (torch.Tensor, numpy.ndarray)):
            return elem.dtype

        raise TypeError(f'`{elem}` is not supported.')

    return apply_single(container, fn=_dtype)


def astype(container, *, schema=None, device=None):
    """Ensure the specified device-dtype schema on a structured container.

    Must be used only on structured data collated into `torch.Tensor`-s.
    """
    def _astype(elem, schema=None):
        # undefined schema means keeping `source` specs
        if schema is None:
            return elem

        # if batch is a `torch.Tensor` then we're in the base case
        if isinstance(elem, torch.Tensor):
            if isinstance(schema, torch.dtype):
                return elem.to(dtype=schema, device=device)

            elif not isinstance(schema, dict):
                raise TypeError(f'`schema` for a torch.Tensor '
                                f'must be a `dict`. Got `{schema}`.')

            # if the specs match the tensor, then no copy is made
            return elem.to(**{'device': device, **schema})

        if isinstance(elem, numpy.ndarray):
            return elem.astype(schema)

        # schema does not apply to anything else: keep it as is
        #  Since we are called **after** `torch_collate`, all atypical cases
        #  must have already been handled there.
        return elem

    return apply_pair(container, schema, fn=_astype)


def empty_like(element, *leading):
    """Empty structured storage for `element` with extra leading dimensions."""

    def _empty(el):
        if isinstance(el, torch.Tensor):
            # forced an on-host tensor storage
            return el.new_empty((*leading, *el.shape),
                                device=torch.device('cpu'))

        elif isinstance(el, numpy.ndarray):
            # a numpy ndarray is kept as is
            return numpy.empty((*leading, *el.shape), dtype=el.dtype)

        elif isinstance(el, Number):
            # numpy storage for pythonic scalar numbers
            return numpy.empty(leading, dtype=type(el))

        # use array of object as storage for strings, bytes and other stuff
        #  this does not readily befriend torch's `default_collate`
        return numpy.empty(leading, dtype=object)

    return apply_single(element, fn=_empty)
