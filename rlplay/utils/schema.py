import torch
import numpy

from numbers import Number
from collections import abc

from itertools import repeat


def apply(*objects, fn):
    """Traverse nested containers (dict, list, or tuple) of objects.

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
    if isinstance(main, (str, bytes)):
        return fn(*objects)

    # delegate other types to the function (int, float, etc.)
    if not isinstance(main, (abc.Sequence, abc.Mapping)):
        return fn(*objects)

    # `objects` is Iterable[Union[Sequence, Mapping]], but not `str` or `bytes`
    # check that the current level data are objects of the same type
    if not all(o is None or isinstance(o, type(main)) for o in others):
        raise TypeError(f'`{others}` do not match type `{type(main)}`')

    # (dict, OrderedDict, etc) -> dict
    if isinstance(main, abc.Mapping):
        # make sure main has superset of keys
        keys, _empty = main.keys(), {}
        assert all(d is None or keys >= d.keys() for d in others)  # TypeError

        # recurse and rebuild the mapping as dict (main left-joins with others)
        others = [_empty if d is None else d for d in others]
        return {k: apply(main[k], *(d.get(k) for d in others), fn=fn)
                for k in keys}

    # (tuple, namedtuple, list, etc) -> list*
    elif isinstance(main, abc.Sequence):
        # check if sequences conform
        length, _none = len(main), repeat(None)
        assert all(s is None or length == len(s) for s in others)  # TypeError

        # replace None-s with infinitely repeated None
        others = (_none if s is None else s for s in others)
        values = [apply(*row, fn=fn) for row in zip(main, *others)]
        if isinstance(main, tuple):
            # use `._make` to preserve namedtuples
            return getattr(main, '_make', tuple)(values)

        # demote everything else to list
        return values


def apply_single(main, fn):
    """A version of `apply` optimized for use with one structured container.
    """
    # special sequence types must precede generic `Sequence` check
    if isinstance(main, (str, bytes)):
        return fn(main)

    # (tuple, namedtuple, list, etc) -> list*
    elif isinstance(main, abc.Sequence):
        values = [apply_single(row, fn=fn) for row in main]
        if isinstance(main, tuple):
            # use `._make` to preserve namedtuples
            return getattr(main, '_make', tuple)(values)

        # demote everything else to list
        return values

    # (dict, OrderedDict, etc) -> dict
    elif isinstance(main, abc.Mapping):
        return {k: apply_single(main[k], fn=fn) for k in main.keys()}

    # delegate other types to the function (int, float, etc.)
    return fn(main)


def apply_pair(main, other, fn):
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
        if isinstance(main, tuple):
            # use `._make` to preserve namedtuples
            return getattr(main, '_make', tuple)(values)

        # demote everything else to list
        return values


def dtype(obj):
    """Get the schema of data in nested containers (list, dict, tuple)."""
    def _dtype(obj):
        # types are returned as is
        if isinstance(obj, type):
            return obj

        # special objects
        elif isinstance(obj, (str, bytes, Number)):
            return type(obj)

        # tensors and ndarrays get special treatment
        elif isinstance(obj, (torch.Tensor, numpy.ndarray)):
            return obj.dtype

        raise TypeError(f'`{obj}` is not supported.')

    return apply_single(obj, fn=_dtype)


def astype(batch, *, schema=None, device=None):
    """Ensure the specified device-dtype schema on a structured container.

    Must be used only on structured data collated into `torch.Tensor`-s.
    """
    def _astype(data, schema=None):
        # undefined schema means keeping `source` specs
        if schema is None:
            return data

        # if batch is a `torch.Tensor` then we're in the base case
        if isinstance(data, torch.Tensor):
            if isinstance(schema, torch.dtype):
                return data.to(dtype=schema, device=device)

            elif not isinstance(schema, dict):
                raise TypeError(f'`schema` for a torch.Tensor '
                                f'must be a `dict`. Got `{schema}`.')

            # if the specs match the tensor, then no copy is made
            return data.to(**{'device': device, **schema})

        if isinstance(schema, numpy.ndarray):
            return data.astype(schema)

        # schema does not apply to anything else: keep it as is
        #  Since we are called **after** `torch_collate`, all atypical cases
        #  must have already been handled there.
        return data

    return apply_pair(batch, schema, fn=_astype)


def shape(obj, n_batch_dim=0):
    """Get the schema of data in nested containers (list, dict, tuple)."""
    def _shape(obj):
        # types are returned as is
        if isinstance(obj, (torch.Tensor, numpy.ndarray)):
            return obj.shape[n_batch_dim:]

    return apply_single(obj, fn=_shape)
