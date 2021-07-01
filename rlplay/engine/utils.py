import cloudpickle

import torch.multiprocessing as multiprocessing


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
