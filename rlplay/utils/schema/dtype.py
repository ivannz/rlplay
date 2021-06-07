import numpy
from collections import abc


def infer(obj, *, align=False):
    """Infer numpy's structured data type for the given nested structure.

    Details
    -------
    Recursively converts sequences to structured data types with either proper
    fields in case of namedtuples, or default opaque names (`f#`) in case of
    ordered containers (tuple, list). Similarly, mappings, e.g. dict, are
    transformed into structured dtypes with field names defined by the keys.

    Uses `numpy.asanyarray` to infer the correct basic data type and shape.

    Keeps the original container and element types in dtype's metadata to
    facilitate reconstruction.
    """

    if isinstance(obj, abc.Sequence) and not isinstance(obj, (str, bytes)):
        # detect a namedtuple by an `_asdict` attribute
        if hasattr(obj, '_asdict'):
            # in case of a namedtuple, the dtype inherits its field names
            named = obj._asdict().items()

        else:
            # other list- or tuple-like get anonymous field names ('f#')
            named = [('', v) for v in obj]

        # recurse into each value in the container
        dtype = [(name, infer(v, align=align)) for name, v in named]

    elif isinstance(obj, abc.Mapping):
        # XXX a mapping emits the same structured dtype as a namedtuple
        dtype = [(k, infer(obj[k], align=align)) for k in obj.keys()]

    else:
        # construct an elementary data with shape inferred by numpy
        inferred = numpy.asanyarray(obj)
        dtype = inferred.dtype, inferred.shape

    # save original type in metadata, to enable reassembly
    return numpy.dtype(dtype, align=align, metadata={'type': type(obj)})


def check(obj, dtype):
    """Pack the given nested structured object according to dtype.

    Details
    -------
    `dtype` is needed to make sure that flattened values in mappings
    are strictly aligned by their keys with the dtype's fields.
    """
    # simplify the object and validate
    kind = dtype.metadata['type']
    if not isinstance(obj, kind):
        raise TypeError(f'`obj` must be `{kind}`. Got `{type(obj)}`.')

    if isinstance(obj, abc.Sequence) and not isinstance(obj, (str, bytes)):
        # though namedtuples have mappable fields, they are nonetheless
        #  immutable tuples with fixed order order of elements
        assert len(obj) == len(dtype.fields)

        # sequence-likes are aligned with dtype's field order
        dtypes = [dt for dt, *_ in dtype.fields.values()]
        return tuple(map(check, obj, dtypes))

    elif isinstance(obj, abc.Mapping):
        assert obj.keys() == dtype.fields.keys()

        # make sure to order the items according to dtype
        return tuple([
            check(obj[k], dt) for k, (dt, *_) in dtype.fields.items()
        ])

    # dtype is unstructured
    if not numpy.can_cast(obj, dtype, casting='safe'):
        raise TypeError(f'`{obj}` cannot be safely cast to `{dtype}`.')

    # return the basic element as is (don't care about actual dtype)
    return obj


def rebuild(value, *, copy=False):
    """Reassemble a structured value."""
    assert value.dtype.metadata

    kind = value.dtype.metadata['type']
    if issubclass(kind, abc.Sequence) and not issubclass(kind, (str, bytes)):
        # retrieve the fields' values and wrap it as an appropriate sequence
        values = [rebuild(value[k], copy=copy) for k in value.dtype.fields]
        if issubclass(kind, abc.MutableSequence):
            return values

        return getattr(kind, '_make', tuple)(values)

    elif issubclass(kind, abc.Mapping):
        # reconstruct the fields into a dict
        return {k: rebuild(value[k], copy=copy) for k in value.dtype.fields}

    # every field of a structured type is view into the base array
    return value.copy() if copy else value
