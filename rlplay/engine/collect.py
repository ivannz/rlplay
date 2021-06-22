import torch
import numpy

from collections import namedtuple

from ..utils.schema.base import unsafe_apply
from ..utils.schema.shared import aliased, torchify


State = namedtuple('State', ['obs', 'act', 'rew', 'fin'])
State.__doc__ = r"""The current state to be acted upon: the current observation
    $s_t$, the past action $a_{t-1}$, the most recently received reward $r_t$,
    and the termination flag $d_t$:

        $(x_t, a_{t-1}, r_t, d_t)$

    where $x_t = s_*$ if $d_t$ else $s_t$, boolean $d_t$ indicates if $r_t$ and
    $a_{t-1}$ must be disregarded, i.e. $x_t$ in `obs` is the start of a new
    trajectory rollout (see detailed description below).

    Attributes
    ----------
    obs : nested container of tensors with shape (batch, ...)
        A batch of observations $x_t$ from the environment containing either
        the current state $s_t$ observed as a result of past actions and
        the environment transitions, or the initial state $s_*$ observed upon
        a reset. Which is which is determined by the corresponding boolean
        flag in `.fin`.

    act : nested container of tensors with shape (batch, ...)
        The batch of the most recent actions $a_{t-1}$, taken in each
        environment causing the transition to $s_t$). Must be ignored
        upon the reset of an environment, which is indicated by
        the corresponding boolean flag in `.fin`.

    rew : tensor of float with shape (batch,)
        The batch of the most recently received rewards $r_t$ obtained from
        transitioning from $s_{t-1}$ to $s_t$ by acting $a_{t-1}$. Like `.act`
        must be ignored whenever the corresponding boolean flag in `.fin` is
        set, which indicates a reset.

    fin : tensor of bool with shape (batch,)
        The batch of boolean flags $d_t$ indicating that the previous action
        $a_{t-1}$ caused a transition to a terminal state (true $s_t$) in
        the corresponding environment, upon which it was automatically reset
        to $s_*$, which overwrote $s_t$ in `.obs`. If false, then `.obs`
        contains the true current state $s_t$.
    """

InputState = namedtuple('InputState', ['input', 'hx'])
InputState.__doc__ = """This is a convenience object, that extends `State` by
    an `hx` container of tensors, that represents the persistent, i.e.
    recurrent, state of the actor at the start of a trajectory fragment
    (rollout)."""

Fragment = namedtuple('Fragment', ['state', 'actor', 'env', 'bootstrap', 'hx'])
Fragment.__doc__ = r"""A `T x batch` fragment of the trajectories of the actor
    in a batch of environments and relevant information, e.g. rnn states,
    value estimates, etc.

    Attributes
    ----------
    state : nested container of tensors with shape (T, batch, ...)
        The obs-act-rew-fin state with almost the same semantics as described
        in `State`, except for `.obs`, which in this case is $s_{t-1}$, i.e.
        the past observation, to which the actor reacted with $a_{t-1}$, and
        which caused a transition to $s_t$, which yielded the reward $r_t$,
        with the termination flag value $d_t$. See docs for `Stepper.step`.

    actor : nested container of tensors with shape (T, batch, ...)
        The extra information returned by the actor's `.step` along with
        the action $a_t$ in response to the state $(s_t, a_{t-1}, r_t, d_t)$,
        such as the value estimates $v(s_t)$ or rnn state $h_t$.

    env : nested container of tensors with shape (T, batch, ...)
        The extra data received from the environments' `.step` upon taking
        the actions $a_t$ in them.

    bootstrap : tensor of float with shape (batch,)
        A vector of value estimates $v(s_T)$ used to bootstrap the present
        value of the reward flow following the state $s_T$, which falls beyond
        the fragment $(s_t, a_t, r_{t+1}, d_{t+1})_{t=0}^{T-1}$ recorded in
        `.state`, but is related to $r_T$ and $d_T$. If a flag in `.state.fin`
        boolean tensor is set, then the corresponding bootstrap value should be
        ignored when computing the returns, since the return from a terminal
        state is identically zero.

    hx : container of tensor, or None
        The most recent recurrent state of the actor $h_t$, which conditions
        its response along with the `.obs`, `.act`, `.rew` inputs similarly to
        $a_t, h_{t+1} = \pi(s_t, a_{t-1}, r_t, d_t; h_t)$.

    Details
    -------
    `bootstrap` here is not the statistical term, but the RL term, which means
    an estimate of the expected future return using the current value function
    approximation $v$. The value $v(s_T)$ is the expected present values of
    future rewards from trajectories starting at state $s_T$ and following
    the policy $\pi$ associated with the $v$ approximation: $
        v(s_T)
            = \mathbb{E}_\pi \bigl(
                r_{T+1} + \gamma r_{T+2} + ...
            \big\vert s_T \bigr)
    $ where $r_t$ is the reward due to $(s_{t-1}, a_{t-1}) \to s_t$ transition
    with $a_{t-1}$ being a response to $s_{t-1}$.
    """


def structured_setitem_(dst, index, value):
    """Recursively set `element[index] = value` in nested container."""
    unsafe_apply(dst, value, fn=lambda d, v: d.__setitem__(index, v))


def structured_tensor_copy_(dst, src, *, at=None):
    """Copy tensor data from the `src` nested container into torch tensors of
    the `dst` nested container at the specified index (int or tuple of ints).
    """
    if at is None:
        return unsafe_apply(dst, src, fn=torch.Tensor.copy_)

    return unsafe_apply(dst, src, fn=lambda d, s: d[at].copy_(s))


class BaseActorModule(torch.nn.Module):
    """Define convenience methods for trajectory fragment collection."""
    # actor is unaware of the number of environments it interacts with, because
    #  they are the batch dimension

    @torch.no_grad()
    def reset(self, at, hx=None):
        """Reset the specified pieces of actor's recurrent context `hx`."""
        assert hx is not None

        unsafe_apply(hx, fn=lambda x: x[:, at].zero_())
        # XXX could the actor keep `hx` unchanged in `forward`? No, under the
        # current api, since `.fin` reflects that the input is related to a
        # freshly started trajectory, i.e. only $h_t$ and $x_t$ are defined,
        # while $a_{t-1}$, $r_t$, and $d_t$ are not. `fin` does not tell us
        # anything about whether $h_{t+1}$ should be reset after the actor's
        # update $(x_t, a_{t-1}, r_t, d_t, h_t) \to (a_t, h_{t+1})$ or not.
        return hx

    @torch.no_grad()
    def value(self, obs, act=None, rew=None, fin=None, *, hx=None):
        _, _, info = self(obs, act=act, rew=rew, fin=fin, hx=hx)
        return info['value']

    @torch.no_grad()
    def step(self, obs, act=None, rew=None, fin=None, *, hx=None):
        return self(obs, act=act, rew=rew, fin=fin, hx=hx)

    def forward(self, obs, act=None, rew=None, fin=None, *, hx=None):
        r"""Get $a_t$ and $h_{t+1}$ from $(x_t, a_{t-1}, r_t, d_t)$, and $h_t$.

        Parameters
        ----------
        obs : nested container of tensors with shape (n_envs, ...)
            The current observations $x_t$ from by a batch of environments. If
            a flag in `fin` is set, then the corresponding observation in $x_t$
            is $x_*$ -- the initial state/observation of an environments after
            a reset. Otherwise, the data in `obs` is $x_t$ -- the observation
            obtained after taking the previous action $a_{t-1}$ (`act`) in
            the environment: $(s_{t-1}, a_{t-1}) \to (s_t, x_t, r_t, d_t)$.

        act : nested container of tensors with shape (n_envs, ...)
            A batch of the last actions $a_{t-1}$ taken in the environments by
            the actor. Components of `act` MUST be ignored if the corresponding
            flag in `fin` is set, because the associated environment has just
            been reset, and a new trajectory is being rolled out.

        rew : float tensor with shape (n_envs,)
            The reward that each environment yielded due to the transition to
            $x_t$ after taking action $a_{t-1}$. Components of `rew` MUST be
            ignored if the corresponding flag in `fin` is set.

        fin : bool tensor with shape (n_envs,)
            The reset flags $d_t$ indicate which of the previous actions in
            `act` and last rewards in `rew` have been invalidated by a reset
            of the corresponding environment. Also indicates if the observation
            $x_t$ is the initial observation $s_*$ after a reset (True), or
            a non-terminal observation $s_t$ (False).

        hx : container of tensors of arbitrary shape
            The recurrent context $h_t$, parts of which ought to be reset, if
            the corresponding flags in `fin` are set. If `None` then it must be
            initialized anew.

        Returns
        -------
        hx : container of tensors of arbitrary shape
            The next recurrent context $h_{t+1}$ due to $h_t$ and the inputs.
        """
        raise NotImplementedError


# we build buffer with torch (in pinned memory), then mirror them to numpy
@torch.no_grad()
def prepare(env, actor, n_steps, n_envs, *, pinned=False, shared=False):
    """Build the trajectory fragment buffer as a nested container of torch tensors.

    Idea
    ----
    Create the data structures as torch tensors residing in pinned memory.
    Then numpify them.

    https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers
    """
    obs_ = env.reset()

    # take a random action in the environment
    act_ = env.action_space.sample()

    # `info_env` is a nested container of numeric scalars or numpy arrays
    obs_, rew_, fin_, info_env = env.step(act_)

    # a single record of the collected sample: ensure correct data types
    # `obs` and `act` are a nested container of numpy arrays or scalars with
    #  env's dtypes, `rew` and `fin` are python float and bool, respectively.
    state = State(obs_, act_, numpy.float32(rew_), bool(fin_))

    # `bootstrap` is the value estimate $v(s_t)$, hence is the same as
    #  `state.rew`, but with the unit temporal dim.
    value = torchify(state.rew, 1, n_envs, shared=shared, pinned=pinned)

    # the buffers for the state and env_info data are n_steps x n_envs
    state, info_env = torchify((state, info_env), n_steps, n_envs,
                               shared=shared, pinned=pinned)

    # torchify one complete batch and make a single pass through the actor
    pyt = unsafe_apply(state, fn=lambda x: x[:1])  # 1 x n_envs x ...
    unused, hx, info_actor = actor.step(pyt.obs, pyt.act,
                                        pyt.rew, pyt.fin, hx=None)

    # get one time slice from the actor's info [B x ...]
    # XXX `info_actor` must respect the temporal and batch dims
    info_actor = torchify(unsafe_apply(info_actor, fn=lambda x: x[0]),
                          n_steps, shared=shared, pinned=pinned)

    # `hx` must not have the temporal dimension: the actor fully specifies it
    hx = torchify(hx, shared=shared, pinned=pinned)

    return Fragment(state=state, actor=info_actor, env=info_env,
                    bootstrap=value, hx=hx)


@torch.no_grad()
def startup(envs, actor, buffer, *, pinned=False):
    # `fragment` contains [T x B x ?] aliased torch tensors of actor/env
    # infos, trajectory fragment $(x_{t-1}, a_{t-1}, r_t, d_t)_{t=1}^T$,
    # and bootstrap values $v(s_T)$, possibly allocated in torch's shared
    # memory.
    fragment = aliased(buffer)  # just a copy-free pyt-npy alias

    # reset the initial recurrent state of the actor, `hx`, to zero
    unsafe_apply(fragment.pyt.hx, fn=torch.Tensor.zero_)

    # Fetch a single [B x ?] observation (a VIEW into fragment for now)
    npy, pyt = unsafe_apply((fragment.npy.state, fragment.pyt.state,),
                            fn=lambda x: x[0])

    # Flag the state as having just been reset, meaning that the previous
    #  reward and action are invalid.
    npy.fin[:] = True
    npy.rew[:] = 0.  # zero `.rew`, leave `.act` undefined
    for j, env in enumerate(envs):
        structured_setitem_(npy.obs, j, env.reset())  # x_0 = s_*
        actor.reset(j, fragment.pyt.hx)  # h_0 = h_*

    # Create `state`, a dedicated container of [B x ?] aliased copy of the data
    # and the current recurrent state of the actor $h_t$, both possibly
    # residing in torch's pinned memory.
    state = aliased(InputState(pyt, fragment.pyt.hx), copy=True, pinned=pinned)

    # writable view of `state.pyt.input` with an extra temporal dim
    # `pyt` <<--editable unsqueezed view-->> `_pyt` <<--aliased data-->> `npy`
    unsafe_apply(state.pyt.input, fn=lambda x: x.unsqueeze_(0))  # in-place!
    # XXX we do this so that the actor may rely on T x B x ... data on input
    # `pyt` is used for interacting with the actor, `npy` -- with the fragment

    return state, fragment


@torch.no_grad()
def collect(envs, actor, fragment, state, *, sticky=False):
    r"""March the actor and the environments in lockstep `n_steps` times.

    Parameters
    ----------
    envs : list of gym.Env
        The stateful environments to step through.

    actor : Actor
        The stateful actor, which steps through a batch of environments.

    fragment : Aliased Fragment
        The buffer into which the trajectory fragment is recorded *in-place*.

    state : Aliased State
        The most recent state for the actor to react to. *Updated in-place.*

    sticky : bool, default=False
        Whether to stop interacting with a terminated environment until the end
        of the current trajectory fragment, i.e. until the `collect()`. Once
        an env reached a terminal state, we put a record with its reset
        observation into the trajectory fragment and then stop interacting
        with it until the end of the fragment.

    Details
    -------
    Let $s_t$ be the observed state at time $t$, $a_t$ the action taken at
    time $t$, $r_{t+1}$ -- the reward obtained due to $(s_t, a_t) \to s_{t+1}$
    transition, and $d_{t+1}$ an indicator whether the new state $s_{t+1}$ is
    terminal. Let $s_*$ be the observation just after an environment reset.
    The present value of the reward flow obtained by following the trajectory
    after the state $s_t$, i.e. $(a_t, s_{t+1}, a_{t+1}, ...)$, is defined as $
        G_t = \sum_{j=t}^{h-1} \gamma^{j-t} r_{j+1} 1_{\neg d_{j+1}}
            + \gamma^{h-t} G_h
    $ for $h \geq t$ also known as `the t-th return'.

    Since the rewards following terminal states are identically zero, because
    the episode has ended, such sates are not actionable, and, hence, there
    is no point in storing this $t+1$-st observation. Therefore $d_t = \top$
    in `out[t-1]` indicates that the observation $s_t$, that would have been
    recorded in `out[t]`, is terminal. Instead `out[t]` records $s_*$. On the
    other hand if $d_t = \bot$, then the observation in `out[t]` is the actual
    next observation $s_t$.

    The `.step` collects this data by stepping both through the actor and
    the environment in lockstep into the `out` buffer. The `out.state` buffer
    contains the following data
        +-----+-------------------------------------+
        |   # |  obs      act      rew      fin     |
        +-----+-------------------------------------+
        |   0 |  x_0      a_0      r_1      d_1     |
        |     |      ...      ...        ...        |
        |   t |  x_t      a_t      r_{t+1}  d_{t+1} |
        |     |      ...      ...        ...        |
        | T-1 |  x_{T-1}  a_{T-1}  r_T      d_T     |
        +-----+-------------------------------------+

    with $x_{t+1} = s_*$ if $d_{t+1}$, and $s_{t+1}$ otherwise. The subscript
    indices represent the index in the trajectory sequence, while those in
    square brackets represent the corresponding record in the buffer:
        out[t] = x_t, a_t, r_{t+1}, d_{t+1}

    This non-intuitive indexing notation allows computing the return as
        G[j] = out[j].rew + gamma * (1 - out[j].fin) * G[j+1]

    for `j=0..T-1` where $G[j] \approx G_j$, and `G[T] = 0` if `out[T-1].fin`
    and `G[T] = v(s_T)` otherwise. The latter requires explicitly computing
    the value function at a non-terminal state $s_T$, when collection is
    interrupted mid-trajectory.

    The value function estimates the present value (returns) $
        v(s_t)
            \approx \mathbb{E}_{\tau \sim p \circ \pi} G_t(\tau)
    $ with $
        \tau = (s_t, a_t, r_{t+1})_{t\geq 0}
    $ and $T(\tau) = \inf\{t \geq 1 \colon s_t = \times \}$ -- the trajectory
    termination moment.

    Implementation
    --------------
    The `.step` as a whole is actually $T$ actor-environment interactions
    with the slightly shifted internal loop state `npy/pyt` written into `out`.
    If `sticky` is False, then in `.step` after the iteration `t` of the outer
    loop it is guaranteed that
        out[t] = (    x_t, a_t, r_{t+1}, d_{t+1})
        state  = (x_{t+1}, a_t, r_{t+1}, d_{t+1})

    with x_t = s_* if d_t else s_t, where $s_*$ denotes the state after
    the `env.reset`. If `sticky` is True, then
        x_t = s_t if t < T else $s_*$

    with T = \inf\{t \geq 1 : d_t = \top\}. All `out[t]` for t >= T are to be
    considered invalid. For example, at t=T-1
        out[t] = (x_{T-1}, a_{T-1}, r_T, d_T)
        state  = (    s_*, a_{T-1}, r_T, d_T)
    """

    # shorthands for fast access
    # `out[t]` is $x_t, a_t, r_{t+1}$, and $d_{t+1}$, for $x_t$ above
    out, hx = fragment.npy.state, state.pyt.hx

    # `pyt/npy` is its jagged edge: $x_t, a_{t-1}, r_t$, $d_t$, and $h_t$.
    npy, pyt = state.npy.input, state.pyt.input

    # write the initial recurrent state of the actor to the shared buffer
    structured_tensor_copy_(fragment.pyt.hx, hx)

    # XXX stepper uses a single actor for a batch of environments:
    # * one mode of exploration, poor randomness
    for t in range(len(out.fin)):
        # copy $s_t$ to out[t]
        structured_setitem_(out.obs, t, npy.obs)

        # $a_t$ is a reaction to $s_t, a_{t-1}, r_t, d_t$ (`npy/pyt`)
        # XXX the actor is responsible for copying torch tensors of the state
        #  to a device. At the same time it may return device-resident tensors.
        # The `hx` state associated with an environment with the corresponding
        #  `.fin` set should not be updated (only reset).
        # the actor's outputs, except `hx`, respect time and batch dims, actor
        #  must not update `hx` in-place, just process the context and yield
        #  a new one!
        act_, hx, info_actor = actor.step(pyt.obs, pyt.act,
                                          pyt.rew, pyt.fin, hx=hx)

        structured_tensor_copy_(pyt.act, act_)
        if info_actor:
            # fragment.pyt is likely to have is_shared() = True, so it
            #  cannot be in the pinned memory.
            structured_tensor_copy_(fragment.pyt.actor, info_actor,
                                    at=slice(t, t+1))

        # `.step` through a batch of envs
        for j, env in enumerate(envs):
            # Only recorded interactions can get stuck: if `.fin` is True,
            #  when `t=0`, this means the env has been reset elsewhere.
            if sticky and t > 0 and npy.fin[j]:
                # `npy = (s_*, ?, 0, True)` (`.obs` and `.fin` are stuck)
                npy.rew[j] = 0.

                # hx is not stuck, so it needs to be reset
                actor.reset(j, hx)
                continue

            # get $s_{t+1}$, $r_{t+1}$, and $d_{t+1}$
            obs_, rew_, fin_, info_env = env.step(npy.act[j])
            # structured_setitem_(npy.next_obs, j, obs_)
            if info_env:
                structured_setitem_(fragment.npy.env, (t, j), info_env)

            # `fin_` indicates if `obs_` is terminal and a reset is needed
            if fin_:
                # substitute the terminal $s_{t+1}$ with an initial $s_*$ and
                # zero the actor's recurrent state $h_t = h_*$
                obs_ = env.reset()
                actor.reset(j, hx)

            structured_setitem_(npy.obs, j, obs_)
            npy.rew[j] = rew_
            npy.fin[j] = fin_

        # XXX here the `state` update is complete
        pass

        # finalize out[t]: store $a_t$, $r_{t+1}$ and $d_{t+1}$
        structured_setitem_(out.act, t, npy.act)
        out.rew[t] = npy.rew
        out.fin[t] = npy.fin
        # structured_setitem_(out.next_obs, t, npy.obs)

    # push the last observation into the special T+1-th slot
    # XXX does not resolve the issue of losing terminal observations mid-rollout
    # structured_setitem_(out.obs, t + 1, npy.obs)

    # compute the bootstrapped value estimate for each env.
    if hasattr(fragment.pyt, 'bootstrap'):
        bootstrap = actor.value(pyt.obs, pyt.act,
                                pyt.rew, pyt.fin, hx=hx)
        structured_tensor_copy_(fragment.pyt.bootstrap, bootstrap)

    # write back the most recent recurrent state for the next rollout
    structured_tensor_copy_(state.pyt.hx, hx)

    return True
