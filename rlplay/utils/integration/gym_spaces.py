import importlib

from collections.abc import Mapping, Sequence


def keys(self):
    "D.keys() -> a set-like object providing a view on D's keys"
    return self.spaces.keys()


def items(self):
    "D.items() -> a set-like object providing a view on D's items"
    return self.spaces.items()


def values(self):
    "D.values() -> an object providing a view on D's values"
    return self.spaces.values()


def get(self, key):
    "D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."
    return self.spaces.get(key)


def __contains__(self, key):
    return key in self.spaces


def __len__(self):
    return len(self.spaces)


def __iter__(self):
    return iter(self.spaces)


def patch_spaces_dict_Dict():
    # Mapping has the following inheritance structure and requirements:
    #  Mapping: `__getitem__`
    #    Collection:
    #      Sized: `__len__`
    #      Iterable: `__iter__`
    #      Container: `__contains__`
    # `abc.Mapping` expects `__iter__` to sweep over the keys of the mapping,
    #  and provides generic implementations of `.keys`, `.values`, `.items`
    #  and `.get` based on these methods.

    spaces_dict = importlib.import_module('gym.spaces.dict')

    # patch the missing methods into the class prototype
    spaces_dict.Dict.__len__ = __len__
    spaces_dict.Dict.__contains__ = __contains__
    spaces_dict.Dict.keys = keys
    spaces_dict.Dict.items = items
    spaces_dict.Dict.values = values

    # replace the __iter with a more direct one
    spaces_dict.Dict.__iter__ = __iter__

    # declare Mapping as a virtual base class of Dict
    Mapping.register(spaces_dict.Dict)


def patch_spaces_tuple_Tuple():
    # Sequence: `__getitem__`
    #   Reversible:  `__reversed__`*
    #   Collection:
    #     Sized: `__len__`
    #     Iterable: `__iter__`*
    #     Container: `__contains__`*
    # * [optional] Based on `__len__` and `__getitem__` (which should raise
    #   IndexError), Sequence provides generic slow-ish implementations of
    #   `__iter__`, `__contains__` and `__reversed__`.

    spaces_tuple = importlib.import_module('gym.spaces.tuple')

    # patch the missing methods into the class prototype
    spaces_tuple.Tuple.__len__ = __len__
    spaces_tuple.Tuple.__iter__ = __iter__

    # declare Sequence as a virtual base class of Tuple
    Sequence.register(spaces_tuple.Tuple)


patch_spaces_dict_Dict()
patch_spaces_tuple_Tuple()
