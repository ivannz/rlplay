from .schema import schema


class BaseModuleHook:
    """Base class for module-level hooks."""
    def __init__(self):
        self.hooks, self.n_inputs, self.n_inputs, self.labels = {}, {}, {}, {}

    def register(self, label, module):
        assert module not in self.hooks

        # the input-ouput hook
        self.labels[module] = label
        self.n_inputs[module], self.n_outputs[module] = None, None
        self.hooks[module] = module.register_forward_hook(self._hook)

    def _validate(self, module, inputs, output):
        assert module in self.hooks

        if self.n_inputs[module] is None:
            self.n_inputs[module] = schema(inputs)
        assert self.n_inputs[module] == schema(inputs)

        if self.n_outputs[module] is None:
            self.n_outputs[module] = schema(inputs)
        assert self.n_outputs[module] == schema(inputs)

    def _hook(self, module, inputs, output):
        self._validate(module, inputs, output)
        self.on_forward(self.labels[module], module, inputs, output)

    def clear(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.n_outputs.clear()
        self.n_inputs.clear()
        self.labels.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear()

    def on_forward(self, label, mod, inputs, output):
        raise NotImplementedError
