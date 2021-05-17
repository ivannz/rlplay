import sys
import signal
import traceback

import cloudpickle

import numpy as np
import multiprocessing as mp

from threading import BrokenBarrierError
from multiprocessing import TimeoutError

from gym.spaces import Space
from gym.vector.utils import batch_space

from collections import namedtuple

from .schema import apply_single, getitem, setitem
# from .schema import infer_dtype, check_dtype, rebuild
# patch and register Dict and Tuple spaces as containers in abc
from .integration import gym_spaces as _  # noqa: F401


Endpoint = namedtuple('Endpoint', ['rx', 'tx'])


class CloudpickleSpawner:
    def __init__(self, cls, *args, **kwargs):
        self.cls, self.args, self.kwargs = cls, args, kwargs

    def __call__(self):
        return self.cls(*self.args, **self.kwargs)

    def __getstate__(self):
        return cloudpickle.dumps(self.__dict__)

    def __setstate__(self, data):
        self.__dict__.update(cloudpickle.loads(data))


class SharedArray:
    """Holder for shape, dtype and the underlying shared memory buffer."""
    __slots__ = 'shm', 'dtype', 'shape'

    def __repr__(self):
        text = repr(self.shape)[1:-1] + ', dtype=' + str(self.dtype)
        return type(self).__name__ + '(' + text + ')'

    def __init__(self, shm, dtype, shape):
        self.dtype, self.shape = dtype, shape
        self.shm = shm

    @classmethod
    def alloc(cls, shape, dtype, ctx=mp):
        from ctypes import c_byte

        # let numpy handle the input dtype specs (`dtype.itemsize` is in bytes)
        dtype = np.dtype(dtype)
        n_bytes = int(np.prod(shape) * dtype.itemsize)

        # numpy's dtype typecodes mostly coincide with codes from std::array
        #  but instead we allocate `nbytes` sized storage of `c_byte`
        return cls(ctx.RawArray(c_byte, n_bytes), dtype, shape)

    def numpy(self):
        # build a numpy array in the shared  memory
        return np.ndarray(self.shape, buffer=self.shm, dtype=self.dtype)

    @classmethod
    def from_numpy(cls, array, *leading, ctx=mp):
        shape = *leading, *array.shape
        return cls.alloc(shape, dtype=array.dtype, ctx=ctx)


def from_space(space, *leading, ctx=mp):
    """Make shared objects from nested gym spaces."""
    # spaces.MultiDiscrete
    #   a shaped product of discrete spaces with sizes in `.nvec` array.
    #   - no need for `max(el.nvec)`, since we won't check the bounds
    # spaces.Discrete
    #   a space represented by a nonnegative integer scalar with values
    #   up to `.n`.
    # spaces.MultiBinary
    #   a product of binary spaces (`np.int8`)
    # spaces.Box
    #   a real-valued n-dim vector space optionaly bounded by a box
    def _from_space(el):
        if isinstance(el, Space):
            return SharedArray.alloc(shape=leading + el.shape,
                                     dtype=el.dtype, ctx=ctx)
        raise TypeError(f'Unrecognized type `{type(el)}`')

    return apply_single(space, fn=_from_space)


def from_shared(shm):
    """Reconstruct numpy arrays form shared storage."""
    def _from_shared(el):
        if isinstance(el, SharedArray):
            return el.numpy()
        raise TypeError(f'Unrecognized type `{type(el)}`')
    return apply_single(shm, fn=_from_shared)


class vecEnvWorker:
    @classmethod
    def target(cls, index, spawner, our, their, shared, *, barrier, errors):
        return cls(index, spawner, our, their, shared, barrier, errors).run()

    def __init__(self, index, factory, our, their, shared, barrier, errors):
        if their is not None:
            # close unused ends of the pipes
            their.tx.close()
            their.rx.close()

        # get our process
        self._process = mp.current_process()

        # barrier for synchronous looping, error queue for exceptions
        self._barrier, self._errors = barrier, errors

        # command (read-only) and data (write-only) pipe endpoints
        self.comm, self.shared = our, shared

        # our index in shared memory and the environment factory
        self.index, self.factory = index, factory

    def run(self):
        try:
            self.startup()
            self._barrier.wait()

            while self.dispatch():
                self._barrier.wait()

        # another worker broke down the barrier due to an unhandled exception
        except BrokenBarrierError:
            pass

        # preclude the exception from bubbling up higher, and send it to
        #  the parent process through a queue instead, just before breaking
        #  down the barrier for all parties.
        except Exception:
            typ, val, tb = sys.exc_info()
            msg = "".join(traceback.format_exception(typ, val, tb))
            self._errors.put((self.index, self._process.name, typ, msg))

            self._barrier.abort()

        finally:
            self.shutdown()

    def startup(self):
        # `KeyboardInterrupt` is ignored, since it is issued through the
        #  parent process anyway.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # prepare the environment and shared memory
        self.env = self.factory()
        self.buf_obs, self.buf_act = from_shared(self.shared)

    def shutdown(self):
        self.comm.tx.close()
        self.comm.rx.close()
        self.env.close()

    def dispatch(self):
        try:
            # block until the request and data are ready
            name, *data = self.comm.rx.recv()

        # if the request pipe (its write endpoint) is closed, then
        #  this means that the parent process wants us to shut down.
        except EOFError:
            return False

        if name is None:
            return False

        # raise on non-existent attribute
        if not hasattr(self.env, name):
            raise RuntimeError(f'{self._process.name}: {type(self.env)}'
                               f' has no attribute/method `{name}`.')

        attr = getattr(self.env, name)
        # raise RuntimeError(f'{self._process.name}: {name} is not'
        #                    f' a method of {type(self.env)}.')
        if not callable(attr):
            self.comm.tx.send(attr)
            return True

        elif name == 'reset':
            setitem(self.buf_obs, self.index, attr())
            # self.sem_obs.release()
            self.comm.tx.send(None)
            return True

        elif name == 'step':
            # s_{t+1}, r_{t+1} \sim p(s, r \mid s_t, a_t)
            #  a_t = buf_act[i]

            # self.sem_act.acquire()  # wait for a ready action
            #  we-re synced because were unblocked by the `rx.recv()` above
            obs, rew, done, info = attr(getitem(self.buf_act, self.index))

            # done indicates that $s_{t+1}$ (`obs`) is terminal and thus
            #  its $v(s_{t+1})$ (present value of the reward flow) is zero
            if done:
                obs = self.env.reset()

            # record buf_obs[i] = $s_{t+1}$ if not `done` else $s_0$
            setitem(self.buf_obs, self.index, obs)

            # self.sem_obs.release()  # announce ready observation
            self.comm.tx.send((None, rew, done, info))
            return True

        # get the attribute and call it if the command supplied `data`
        args, kwargs = (), {}
        if data:
            # the last element is expected to be a kwarg dict
            *args, kwargs = data
            if not isinstance(kwargs, dict):
                # assume no kwargs, if the last elem is not a dict
                args, kwargs = data, {}

        name = attr.__name__
        self.comm.tx.send(attr(*args, **kwargs))

        return True


class vecEnv(object):
    def __init__(self, envs, *, method=None, timeout=15):
        ctx = mp  # .get_context(method or 'forkserver')

        # poll an environment for its specs
        # XXX what if the environments are different?
        env = envs[0]()
        self.observation_space = batch_space(env.observation_space, len(envs))
        self.action_space = batch_space(env.action_space, len(envs))
        env.close()
        del env

        # construct shared memory buffers from env's space specs
        #  for efficient exchange of large objects of fixed type.
        shm_obs = from_space(self.observation_space, ctx=ctx)
        shm_act = from_space(self.action_space, ctx=ctx)
        self.shared = shm_obs, shm_act

        # setup producer-consumer synchronization: a common Barrier that we
        #  and all workers wait at for mutual synchronization and signaling
        #  a complete batch.
        self.finished, self.timeout = ctx.Barrier(1 + len(envs)), timeout
        # self.sem_act, self.sem_obs = ctx.Semaphore(0), ctx.Semaphore(0)

        # spawn a worker processes for each environment
        state, self._errors = [], ctx.Queue()
        for j, env in enumerate(envs):
            # Create unidirectional (non-duplex) pipes, and connect them in
            # `crossover` mode between us and them (the worker). This set-up
            #  avoids rare occasions where the main process or the worker read
            #  back their own just issued message.
            ut_rx, ut_tx = ctx.Pipe(duplex=False)
            tu_rx, tu_tx = ctx.Pipe(duplex=False)

            # The connection end points have the following meanigs:
            # * `ut` and `tu` stand for `us-them` and `them-us`, respectively;
            # * [their] `ut_rx` (read) and `tu_tx` (write) ends are used by the
            #   worker to receive signals and yield results, respectively;
            # * [our] `ut_tx` (write) and `tu_rx` (read) ends are used by the
            #   main process to issue commands and read back responses.
            our, their = Endpoint(tu_rx, ut_tx), Endpoint(ut_rx, tu_tx)

            # crossover handles: share the us-them rx and them-us tx ends
            p = ctx.Process(
                args=(j, CloudpickleSpawner(env), their, our, self.shared),
                kwargs=dict(errors=self._errors, barrier=self.finished),
                target=vecEnvWorker.target, daemon=True,
            )
            p.start()

            # close handles unused by us: them-us tx (write), us-them rx (read)
            # XXX pipes and connections are commonly implemented through file
            # descriptors, which imposes a limit on their max number. Also
            # connections are closed when garbage collected (`__del__`).
            their.tx.close()  # tu_tx
            their.rx.close()  # ut_rx

            state.append((p, our))

        # rebuild numpy buffers, assign processes and establish communcations
        self.buf_obs, self.buf_act = from_shared(self.shared)
        self.processes, self.comm = zip(*state)

        # wait until all workers have started
        self._wait()

    @property
    def is_alive(self):
        return all(p.is_alive() for p in self.processes)

    def __len__(self):
        return len(self.processes)

    def _wait(self, timeout=None):
        """Wait until all workers have processed the request."""
        if not self.is_alive:
            return self

        # `Event` needs `.clear` immediately after successful `.wait`. This
        #  must be done atomically in a critical section, to avoid dropping
        #  data due to very fast workers, that mange to pass a new barrier
        #  between `.wait` and `.clear()`. We could've used a bounded
        #  semaphore but waiting at a common barrier is much cheaper.
        exception = None
        try:
            # block until a complete result is available (or timeout)
            self.finished.wait()  # timeout or self.timeout)

        except BrokenBarrierError:
            exception = TimeoutError

        finally:
            if not self._errors.empty():
                index, name, typ, msg = self._errors.get()
                exception = typ(f'`{typ.__name__}` in `{name}` idx={index}.\n{msg}')

        # re-raise any thrown exceptions
        if exception is not None:
            self.__del__()
            raise exception from None

        return self

    def _send(self, request, *args, **kwargs):
        """Send a request through them-us pipe."""
        for comm in self.comm:
            comm.tx.send((request, *args, kwargs))

        return self

    def _recv(self):
        """Get a response from us-them pipe."""
        # serially read from the input endpoints
        return [comm.rx.recv() for comm in self.comm]

    def __del__(self):
        """Shutdown all workers."""
        # workers shutdown if their `rx` end is closed, which is why we close
        # our `tx` end, instead of sending an explicit `close` signal. Also we
        # close our `rx` end of their `tx` connection, because why not.
        # self._send(None)
        for comm in self.comm:
            comm.tx.close()
            comm.rx.close()

        # shutdown event is `special`, because workers do not explicitly sync
        #  afterwards, and just terminate (which is an implicit sync).
        # self._wait()
        for process in self.processes:
            process.join()

    def reset(self, **kwargs):
        self._send('reset', **kwargs)  # action is not required
        self._wait()  # self.sem_obs.acquire()  # wait for the reset
        self._recv()
        return self.buf_obs

    def step(self, actions):
        setitem(self.buf_act, slice(None), actions)  # conforms to the buffer
        self._send('step')  # self.sem_act.release()  # we signal an action
        self._wait()  # self.sem_obs.acquire()  # and wait for an observation

        # aux data is in the pipe, the observation is in the shared memory
        _, reward, done, info = zip(*self._recv())
        return self.buf_obs, np.array(reward), np.array(done), info
        # apply_single(buffer, fn=torch.from_numpy)

    def render(self, mode='human', timeout=30):
        self._send('render', mode=mode)
        self._wait(timeout)
        return any(self._recv())

    def close(self):
        self.__del__()

    def seed(self, seeds):
        """Seed the environment of each worker."""
        assert len(seeds) == len(self)

        for comm, seed in zip(self.comm, seeds):
            comm.tx.send(('seed', seed))

        return self._wait()._recv()

    def broadcast(self, request, *args, **kwargs):
        """Broadcast a general command to all workers."""
        self._send(request, *args, kwargs)
        return self._wait()._recv()


if __name__ == '__main__':
    import gym
    import rlplay.utils.integration.gym  # noqa: F401  # fix gym's rendering

    import tqdm

    from rlplay.utils.wrappers import ToTensor, ChannelFirst, TerminateOnLostLife
    from gym_discomaze import RandomDiscoMaze
    # from gym_discomaze.ext import RandomDiscoMazeWithPosition as RandomDiscoMaze
    from functools import wraps

    # import gym

    class WrappedRandomDiscoMaze:
        @wraps(RandomDiscoMaze.__init__)
        def __new__(cls, *args, **kwargs):
            # return ToTensor(ChannelFirst(RandomDiscoMaze(*args, **kwargs)))
            # # return RandomDiscoMaze(*args, **kwargs)
            return TerminateOnLostLife(gym.make('BreakoutNoFrameskip-v4'))

    vec = vecEnv([
        lambda: WrappedRandomDiscoMaze(10, 10, field=(2, 2))
        for _ in range(4)
    ], method='spawn', timeout=30)

    # from rlplay.utils.plotting.imshow import ImageViewer
    # im = ImageViewer(caption='test', resizable=False, vsync=True, scale=(5, 5))

    print(vec.observation_space)
    print(vec.action_space)

    vec.reset()
    for _ in tqdm.tqdm(range(1000), disable=False):
        obs, reward, done, info = vec.step(vec.action_space.sample())
        # vec.render()  # yes, the exchange works fine im.imshow(obs[0])

    print(obs)
    # vec.render()
    vec.close()
