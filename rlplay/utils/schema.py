import torch
import numpy

from numbers import Number
from collections import abc

from itertools import repeat

# from torch.utils.data._utils.collate import default_collate as torch_collate


def apply(*objects, fn):
    """Traverse nested structured containers (dict, list, or tuple) of objects.

    Details
    -------
    Inspired by the logic of `torch_collate` in traversing structured objects:
        `/torch/utils/data/_utils/collate.py#L43-L86`
        as of commit `b80c6f863f2327c712c478f67c248b94d66b65ac` (2021-04-11)

    The semantics is the following:
        * schema is a basic nested container (list-dict-tuple) of `type`
        * a object has fancy nested subcontainers of elements of type `object`
    """
    main, *others = objects
    if not others:
        # apply maintains the inital number of objects in every recursive call
        #  hence if a single object has been passed, then there was only one
        #  object in var-args to begin with. Therefore `fn` must be unary.
        return apply_single(main, fn=fn)

    # special sequence types must precede generic `Sequence` check
    # XXX `abc` says that `range` and `memoryview` are also sequences
    if isinstance(main, (str, bytes)):
        return fn(*objects)

    # delegate other types to the function (int, float, etc.)
    if not isinstance(main, (abc.Sequence, abc.Mapping)):
        return fn(*objects)

    # `objects` is Iterable[Union[Sequence, Mapping]], but not `str` or `bytes`
    # check that the current level data are objects of the same type
    if not all(o is None or isinstance(o, type(main)) for o in others):
        raise TypeError(f'`{others}` do not match type `{type(main)}`')

    # Mapping, dict, OrderedDict, etc. -> dict
    if isinstance(main, abc.Mapping):
        # make sure main has superset of keys
        keys, _empty = main.keys(), {}
        assert all(d is None or keys >= d.keys() for d in others)  # TypeError

        # recurse and rebuild the mapping as dict (main left-joins with others)
        others = [_empty if d is None else d for d in others]
        return {k: apply(main[k], *(d.get(k) for d in others), fn=fn)
                for k in keys}

    # Sequence, tuple, etc. -> tuple, MutableSequence, list, etc. -> list
    #  namedtuple -> namedtuple
    elif isinstance(main, abc.Sequence):
        # check if sequences conform
        length, _none = len(main), repeat(None)
        assert all(s is None or length == len(s) for s in others)  # TypeError

        # replace None-s with infinitely repeated None
        others = (_none if s is None else s for s in others)
        values = [apply(*row, fn=fn) for row in zip(main, *others)]

        # demote mutable (changeable) sequences to lists
        if isinstance(main, abc.MutableSequence):
            return values

        # ... and immutable to tuple, keeping in mind namedtuples
        # XXX telling a namedtuple from anything else by whether it
        #  has `._make` is as reliable as using `_fields`.
        return getattr(main, '_make', tuple)(values)


def apply_single(main, *, fn):
    """A version of `apply` optimized for use with one structured container.
    """
    # special sequence types must precede generic `Sequence` check
    if isinstance(main, (str, bytes)):
        return fn(main)

    # any Sequence is regressed to either a list or a tuple
    elif isinstance(main, abc.Sequence):
        values = [apply_single(row, fn=fn) for row in main]
        # demote mutable sequences to lists
        if isinstance(main, abc.MutableSequence):
            return values

        #  ... and immutable to tuple, keeping in mind namedtuples
        return getattr(main, '_make', tuple)(values)

    # all Mapping-s are devolved into dicts
    elif isinstance(main, abc.Mapping):
        # recurse and rebuild any mapping as dict
        return {k: apply_single(main[k], fn=fn) for k in main.keys()}

    return fn(main)


def apply_pair(main, other, *, fn):
    """`Apply` optimized for use with paired structured containers."""
    # special sequence types must precede generic `Sequence` check
    if isinstance(main, (str, bytes)):
        return fn(main, other)

    # delegate other types to the function (int, float, etc.)
    elif not isinstance(main, (abc.Sequence, abc.Mapping)):
        return fn(main, other)

    # `objects` is Iterable[Union[Sequence, Mapping]], but not `str` or `bytes`
    # check that the current level data are objects of the same type
    if not (other is None or isinstance(other, type(main))):
        raise TypeError(f'`{other}` does not match type `{type(main)}`')

    # (dict, OrderedDict, etc) -> dict
    if isinstance(main, abc.Mapping):
        # recurse and rebuild the mapping as dict (main left-joins with others)
        return {k: apply_pair(main[k], other.get(k), fn=fn)
                for k in main.keys()}

    # (tuple, namedtuple, list, etc) -> list*
    elif isinstance(main, abc.Sequence):
        # check if sequences conform
        assert len(main) == len(other)  # TypeError

        values = [apply_pair(m, o, fn=fn) for m, o in zip(main, other)]
        # demote mutable sequences to lists
        if isinstance(main, abc.MutableSequence):
            return values

        # ... and immutable to tuple, keeping in mind special namedtuples
        return getattr(main, '_make', tuple)(values)


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


def dtype(container):
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
