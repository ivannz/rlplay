from .plyr import suply, tuply, setitem, getitem
from .shared import aliased, torchify, numpify

from inspect import signature


def check_signature(fn, /, *args, **kwargs):
    # make sure that the required arguments are proper identifiers
    arguments = list(map(str, args + tuple(kwargs.keys())))
    assert all(n.isidentifier() for n in arguments)

    # fetch the parameters in the signature
    parameters = signature(fn).parameters
    order = {
        p.name: j for j, p in enumerate(parameters.values())
        if p.kind != p.VAR_KEYWORD
    }

    if not order.keys() >= set(arguments):
        raise RuntimeError(f'Bad signature `{order}` in `{fn.__name__}`. '
                           f'Must include `{arguments}`.')

    # `obs`, `act`, `rew`, `fin` must be the FIRST arguments
    for j, n in enumerate(map(str, args)):
        if order[n] != j:
            raise RuntimeError(f'Argument `{n}` is at the wrong '
                               f'position `{j} != {order[n]}`.')
