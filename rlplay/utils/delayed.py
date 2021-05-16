import sys
import warnings

from signal import signal, getsignal, raise_signal
from signal import SIGINT, SIG_IGN

from inspect import isgeneratorfunction
from functools import wraps


class DelayedKeyboardInterrupt:
    """Create an atomic section with respect to the Keyboard Interrupt.

    Parameters
    ----------
    action : str, ("delay", "raise", "ignore")
        The desired behaviour on KeyboardInterrupt. If action is "raise" then
        this effectively disables the critical section and passes the keyboard
        interrupt events through. If action is "delay", then the events are
        delayed until the scope of the critical section is left, e.g. exiting
        from the context or returning from a call. If action is "ignore" then
        the interrupts are ignored.

    Example
    -------
    Typically used as a function decorator or within the `with` context in
    the following manner:
    >>> @DelayedKeyboardInterrupt("delay")
    >>> def critical():
    >>>     pass
    >>>
    >>> with DelayedKeyboardInterrupt("ignore") as flag:
    >>>     for i in range():
    >>>         critical()
    >>>
    >>>         if flag:
    >>>             break

    Details
    -------
    The `type`, `value` and `traceback` parameters of
        [`__exit__`](https://docs.python.org/3/reference/datamodel.html#object.__exit__)
    describe the exception raised within the context. All three arguments are
    `None` if the context has been exited without an exception. Thus we may
    have three cases in `__exit__`:

        1. no exception was raised within the context
        2. a KeyboardInterrupt has fired
        3. context exited due to some other exception

    If `action='raise'`, then the exception is supplied only in cases 2 and 3,
    and `captured` is `None`, since we have not registered our handler. If we
    have `action='ignore'` or 'delay' then `type` is not `None` only in case 3,
    since our own hanlder has intercepted the interrupt, if case 2 took place.

    We prioritize other exceptions before KeyboardInterrupt in all cases.
    """

    def __call__(self, fn):
        # bypass our logic if we do not 'delay' or 'ignore'
        if self.action == 'raise':
            return fn

        if isgeneratorfunction(fn):
            return self._decorate_generator(fn)

        return self._decorate_function(fn)

    def _decorate_function(self, fn):
        """Wrap each function call in the context manager."""

        # create a new instance of self with each call
        cls, action = type(self), self.action

        @wraps(fn)
        def _function(*args, **kwargs):
            with cls(action=action):
                return fn(*args, **kwargs)

        return _function

    def _decorate_generator(self, fn):
        """Wrap each invocation of the generator within the context manager.

        Details
        -------
        Generators are suspended and unsuspended at its `yield` statements,
        therefore we make sure the interrupt signal mode is properly set every
        time the execution flow returns into the wrapped generator and restored
        when it returns through our `yield` to our caller.

        Thus we wrap each interaction with to guaranteed the required signal
        state for all operations inside the stack frame of generator, i.e.
            * `.send` (the init and core generation logic)
            * `.throw` (the exception handling logic)
            * and even `.close` (the premature shutdown logic)

        All exceptions raised by the generator (either by `.throw` or `.send`
        methods) bubble up to our caller, except for `StopIteration`, from
        which we take its special `.value` payload and `return it` instead of
        re-raising. (see docs for return-statement, PEP-342, and PEP-380).
        """

        # create a new instance of self with each interaction
        cls, action = type(self), self.action

        @wraps(fn)
        def _generator(*args, **kwargs):
            gen = fn(*args, **kwargs)

            try:
                # fire up the generator
                with cls(action=action):
                    response = gen.send(None)

                while True:
                    try:
                        # forward the response to our caller
                        request = yield response

                    except GeneratorExit:
                        # in case the generator has custom GeneratorExit logic
                        with cls(action=action):
                            gen.close()
                        raise

                    except BaseException:
                        # propagate the exception thrown at us by the caller
                        with cls(action=action):
                            response = gen.throw(*sys.exc_info())

                    else:
                        # get the generator's response to the last request
                        with cls(action=action):
                            response = gen.send(request)

            except StopIteration as e:
                # the generator is done, return whatever is in StopIteration
                return e.value

        return _generator

    def __init__(self, action='delay'):
        assert action in ('raise', 'delay', 'ignore')
        self.action = action

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.action)})'

    def __bool__(self):
        return self.captured is not None

    def handler(self, signalnum, frame):
        # we are here only if action is `delay`
        self.captured = signalnum, frame

    def __enter__(self):
        # the original handler could be a callable, SIG_IGN (ignore),
        #  SIG_DFL (system default), or None (unknown).
        self.original, self.captured = getsignal(SIGINT), None
        if self.original is None:
            warnings.warn('Unable to delay or ignore `KeyboardInterrupt`.',
                          RuntimeWarning)

        # delay or utterly ignore the INT signal
        if self.action != 'raise':
            signal(SIGINT, self.handler if self.action == 'delay' else SIG_IGN)

        return self

    def __exit__(self, type, value, traceback):
        # restore the original handler whatever it was
        original, self.original = self.original, None
        if original is not None:
            signal(SIGINT, original)

        # do not keep reference to the interrupted stack frame
        captured, self.captured = self.captured, None

        # If no exception was supplied, but we captured a KeyboardInterrupt,
        # then we are 'delay', since `captured` is always `None` otherwise.
        if type is None and captured is not None:
            # re-raise the interrupt or call the original handler
            if callable(original):
                original(*captured)
            else:
                raise_signal(SIGINT)

        # prioritize bubbling up all other exceptions
        pass


if __name__ == '__main__':
    import time

    @DelayedKeyboardInterrupt('ignore')
    def step_1():
        print('step_1 began')
        time.sleep(1.0)
        print('step_1 ended')

    @DelayedKeyboardInterrupt('delay')
    def step_2():
        print('step_2 began')
        time.sleep(1.0)
        print('step_2 ended')

    def step_3():
        print('step_3 began')
        time.sleep(1.0)
        print('step_3 ended')

    for i in range(10):
        with DelayedKeyboardInterrupt('delay'):
            step_1()
            step_2()
            step_3()  # protected by the current context
