from .schema.tools import gettype


class BaseModuleHook:
    """Base class for module-level hooks."""
    def __init__(self):
        self.hooks, self.n_inputs, self.n_outputs, self.labels = {}, {}, {}, {}
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    def toggle(self, mode=None):
        if mode is None:
            mode = not self._enabled

        old_mode = self._enabled
        self._enabled = bool(mode)
        return old_mode

    def register(self, label, module):
        assert module not in self.hooks

        # the input-ouput hook
        self.labels[module] = label
        self.n_inputs[module], self.n_outputs[module] = None, None
        self.hooks[module] = module.register_forward_hook(self._hook)

    def _validate(self, module, inputs, output):
        assert module in self.hooks

        if self.n_inputs[module] is None:
            self.n_inputs[module] = gettype(inputs)
        assert self.n_inputs[module] == gettype(inputs)

        if self.n_outputs[module] is None:
            self.n_outputs[module] = gettype(inputs)
        assert self.n_outputs[module] == gettype(inputs)

    def _hook(self, module, inputs, output):
        if self.enabled:
            self._validate(module, inputs, output)
            self.on_forward(self.labels[module], module, inputs, output)

    def exit(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
        self.n_outputs.clear()
        self.n_inputs.clear()
        self.labels.clear()

    def __enter__(self):
        self._ctx_mode = self.toggle(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.toggle(self._ctx_mode)

    def __del__(self):
        self.exit()

    def on_forward(self, label, mod, inputs, output):
        raise NotImplementedError
