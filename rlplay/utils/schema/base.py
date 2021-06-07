from collections import abc
from itertools import repeat


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
        other = {} if other is None else other

        # recurse and rebuild the mapping as dict (main left-joins with others)
        return {k: apply_pair(main[k], other.get(k), fn=fn)
                for k in main.keys()}

    # (tuple, namedtuple, list, etc) -> list*
    elif isinstance(main, abc.Sequence):
        if other is None:
            other = repeat(None)
        else:
            # check if sequences conform
            assert len(main) == len(other)  # TypeError

        values = [apply_pair(m, o, fn=fn) for m, o in zip(main, other)]
        # demote mutable sequences to lists
        if isinstance(main, abc.MutableSequence):
            return values

        # ... and immutable to tuple, keeping in mind special namedtuples
        return getattr(main, '_make', tuple)(values)
