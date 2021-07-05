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

Context = namedtuple('Context', ['state', 'hx', 'next_obs'])
Context.__doc__ = r"""This is a convenience object, that extends `State` by
    an `hx` and a `next_obs` containers of tensors. The first represents the
    persistent context, e.g. the recurrent state, of the actor at the start of
    a trajectory fragment (rollout). The second stores the original true next
    observation $x_{t+1}$ before any reset took place.
    """

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
    """Do `dst[index] = value` in `dst` nested container with values coming
    from another with IDENTICAL structure.
    """
    unsafe_apply(dst, value, fn=lambda d, v: d.__setitem__(index, v))


def structured_tensor_copy_(dst, src, *, at=None):
    """Copy tensor data from the `src` nested container into torch tensors
    of the `dst` nested container (with IDENTICAL structure) at the specified
    index (int, tuple of ints, or slices).
    """
    if at is None:
        return unsafe_apply(dst, src, fn=torch.Tensor.copy_)

    return unsafe_apply(dst, src, fn=lambda d, s: d[at].copy_(s))


class BaseActorModule(torch.nn.Module):
    """Define convenience methods for trajectory fragment collection.

    Details
    -------
    The actor is unaware of the number of environments it interacts with,
    and they are communicated through the batch dimension of the input (dim=1).

    Subclassing
    -----------
    Actor models derived from this class must not store any reference to env
    instances, because envs are unlikely to be easily picklable and shareable
    between subprocesses.
    """

    @torch.no_grad()
    def reset(self, at, hx=None):
        """Reset the specified pieces of actor's recurrent context `hx`.

        Parameters
        ----------
        at : int, slice
            The index or range of environments for which to reset the recurrent
            context.

        hx : nested container of tensors
            The recurrent state is a container of at least 2d tensors of shape
            `: x n_envs x ...`, i.e. having `n_envs` as their second dimension.
            The index or range of environments for which to reset the recurrent
            context.

        Returns
        -------
        hx : nested container of tensors
            The recurrent state updated in-place.

        Details
        -------
        Similar to `torch.nn.LSTM` and other recurrent layers [citation_needed]
        we assume the initial recurent state to be a ZERO non-differentiable
        tensor.

        The seemingly awkward shapes of the tensors in `hx` are rooted in the
        shapes if hiddent states of torch.nn recurrent layers. For example, 
        the LSTM layer has `hx = (c, h)`, both having shape

            num_layers * num_directions x n_batch x n_hidden

        (with the GRU layer's `hx` is a tensor, not a tuple). This 
        """
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
        """The value function estimate of the observations in a one-step-long
        batched input and the recurrent context.

        Details
        -------
        This is a special interface method for the rollout collector, used only
        when a bootstrap value estimate is required in the trajectory fragment.
        """
        _, _, info = self(obs, act=act, rew=rew, fin=fin, hx=hx)
        return info['value']

    @torch.no_grad()
    def step(self, obs, act=None, rew=None, fin=None, *, hx=None):
        """The model's response to a one-step-long batched input and
        the recurrent context.

        Details
        -------
        This is a special interface method for the rollout collector.
        """
        return self(obs, act=act, rew=rew, fin=fin, hx=hx)

    def forward(self, obs, act=None, rew=None, fin=None, *, hx=None):
        r"""Get $a_t$ and $h_{t+1}$ from $(x_t, a_{t-1}, r_t, d_t)$, and $h_t$.

        Warning
        -------
        The shape of the input data is `n_seq x n_batch x ...`, which is
        consistent with `batch_first=False` setting in torch's recurrent
        layer.

        The actor MUST NOT update or change the inputs and `hx` in-place, and
        SHOULD return only newly created tensors.

        Parameters
        ----------
        obs : nested container of tensors with shape = `n_seq, n_envs, ...`
            The current observations $x_t$ from by a batch of environments. If
            a flag in `fin` is set, then the corresponding observation in $x_t$
            is $x_*$ -- the initial state/observation of an environments after
            a reset. Otherwise, the data in `obs` is $x_t$ -- the observation
            obtained after taking the previous action $a_{t-1}$ (`act`) in
            the environment: $(s_{t-1}, a_{t-1}) \to (s_t, x_t, r_t, d_t)$.

        act : nested container of tensors with shape = `n_seq, n_envs, ...`
            A batch of the last actions $a_{t-1}$ taken in the environments by
            the actor. Components of `act` MUST be ignored if the corresponding
            flag in `fin` is set, because the associated environment has just
            been reset, and a new trajectory is being rolled out.

        rew : float tensor, shape = `n_seq, n_envs,`
            The reward that each environment yielded due to the transition to
            $x_t$ after taking action $a_{t-1}$. Components of `rew` MUST be
            ignored if the corresponding flag in `fin` is set.

        fin : bool tensor, shape = `n_seq, n_envs,`
            The reset flags $d_t$ indicate which of the previous actions in
            `act` and last rewards in `rew` have been invalidated by a reset
            of the corresponding environment. Also indicates if the observation
            $x_t$ is the initial observation $s_*$ after a reset (True), or
            a non-terminal observation $s_t$ (False).

        hx : container of tensors
            The recurrent context $h_t$, parts of which ought to be reset, if
            the corresponding flags in `fin` are set.

            NOTE: If `None` then it must be initialized ANEW and conform to
            the batch size of other inputs (see `BaseActorModule.reset` for
            details). It is recommended to let torch's recurrent layers, .e.g.
            torch.nn.GRU, handle `hx = None` automatically.

        Returns
        -------
        act : nested container of tensors with shape = `n_seq, n_envs, ...`
            The actions $a_t$ to be taken in the environment in response to
            the current observations $x_t$ (obs), the previous actions $a_{t-1}$
            (act), the recently obtained reward $r_t$ (rew), the terminaltion
            flag $d_t$ (fin), and the recurrent state $h_t$ (hx).

        hx : nested container of tensors
            The next recurrent context $h_{t+1}$ resulting from $h_t$ and other
            inputs. The data MUST be returned in NEWLY allocated tensors.

        info : dict of tensor containers, shape = `n_seq, ...`
            The auxiliary output the model wishes to return, e.g. a nested
            container of tensors with policy logits (n_seq x n_envs x ...).

            The container MUST include a 'value' key for the value-function
            estimates (n_seq x n_envs) for the observations $x_t$ in `obs` and
            conditional on the other inputs.
        """
        raise NotImplementedError


# we build buffer with torch (in pinned memory), then mirror them to numpy
@torch.no_grad()
def prepare(
    env, actor, n_steps, n_envs, *, pinned=False, shared=False, device=None
):
    """Build the trajectory fragment buffer as a nested container of torch tensors.

    Parameters
    ----------
    env : gym.Env
        The reference environment used to initialize the structure and shapes
        of observation and action buffer.

    actor : Actor
        The actor used to initialize the buffers for the recurrent context and
        auxiliary info for a batch of environments.

    n_steps : int
        The length of the trajectory fragment (rollout) to be stored in the
        constructed buffer (dim=0).

    n_envs : int
        The number of environments in a single buffer (dim=1).

    pinned: bool, default=False
        The underlying storage of the newly created tensors resides in pinned
        memory (non-paged) for faster host-device transfers.

        Cannot be used with `shared=True`.

    shared: bool, default=False
        Allocates the underlying storage of new tensors using torch's memory
        interprocess memory sharing logic which makes it so that all changes
        to the data are reflected between all processes.

        Cannot be used with `pinned=True`.

    device : torch.device, or None
        The device to use when getting the example output from the provided
        actor.

    Returns
    -------
    fragment : nested container of torch tensors
        The buffer into which the trajectory fragment will be recorded.
    """
    obs_ = env.reset()

    # take a random action in the environment
    act_ = env.action_space.sample()

    # `d_env_info` is a nested container of numeric scalars or numpy arrays
    # representing auxiliary environment information associated with the
    # transition.
    obs_, rew_, fin_, d_env_info = env.step(act_)

    # ensure correct data types for `rew_` (to float32) and `fin_` (to bool),
    # while leaving `obs_` and `act_` intact as thery are nested containers of
    # numpy arrays or scalars with environment's proper dtypes.
    rew_, fin_ = numpy.float32(rew_), bool(fin_)

    # create 1 x n_envs buffer for bootstrap value estimate $v(s_T)$
    bootstrap = torchify(rew_, 1, n_envs, shared=shared, pinned=pinned)

    # the buffer for the aux env info data is n_steps x n_envs
    d_env_info = torchify(d_env_info, n_steps, n_envs,
                          shared=shared, pinned=pinned)

    # allocate n_steps x n_envs torch tensor data buffers for actions, rewards
    # and termination flags
    act, rew, fin = torchify((act_, rew_, fin_), n_steps, n_envs,
                             shared=shared, pinned=pinned)

    # unlike others, the `obs` buffer is (n_steps + 1) x n_envs to accommodate
    # the observation used to compute the bootstrap value. This doesn't resolve
    # the issue of losing terminal observations mid-rollout, but allows SARSA
    # and Q-learning.
    obs = torchify(obs_, n_steps + 1, n_envs, shared=shared, pinned=pinned)
    state = State(obs, act, rew, fin)

    # make a single pass through the actor with one 1 x n_envs x ... batch
    pyt = unsafe_apply(state, fn=lambda x: x[:1].to(device))
    unused_act, hx, d_act_info = actor.step(pyt.obs, pyt.act,
                                            pyt.rew, pyt.fin, hx=None)
    # XXX `act_` is expected to have identical structure to `unused_act`
    # XXX `d_act_info` must respect the temporal and batch dims

    # get one time slice from the actor's info [n_envs x ...]
    d_act_info = torchify(unsafe_apply(d_act_info, fn=lambda x: x[0].cpu()),
                          n_steps, shared=shared, pinned=pinned)

    # the actor fully specifies its context `hx`, so we torchify it as is
    hx = torchify(unsafe_apply(hx, fn=torch.Tensor.cpu),
                  shared=shared, pinned=pinned)

    # bundle the buffers into a trajectory fragment
    return Fragment(state=state, actor=d_act_info, env=d_env_info,
                    bootstrap=bootstrap, hx=hx)


@torch.no_grad()
def startup(envs, actor, buffer, *, pinned=False):
    """Alias the fragment buffer and allocate aliased running context for
    rollout collection.

    Parameters
    ----------
    envs : list of gym.Env
        The stateful environments to be reset.

    actor : BaseActorModule
        The stateful actor, the recurrent context of which is to be reset.

    buffer : Fragment
        The reference buffer used to gather the specs and the nested structure
        for the running env-state context.

    pinned: bool, default=False
        Determines if the underlying storage of newly created tensors for
        the running env-state context should reside in pinned memory for
        faster host-device transfers.

    Returns
    -------
    ctx : aliased Context
        The running env-state context which contains properly time synchronised
        input data for the actor: `.state` is $x_t, a_{t-1}, r_t, d_t$, and
        `.hx` is $h_t$. The data in the context is aliased, i.e. `.npy` arrays
        and `.pyt` tensors reference the same underlying data, which allows
        changes in one be reflected in the other.

        Although `.npy` arrays are `n_evs x ...`, while `.pyt` tensors are
        `1 x n_envs x ...`, but both alias the SAME underlying data storage.

    fragment : aliased Fragment
        The numpy-torch aliased trajectory fragment buffer. It is created in
        a zero-copy manner, so it also aliases the `buffer` input parameters.

    Details
    -------
    The created nested container `context` houses torch tensors, that reside in
    shared or pinned memory. Only after having been created, the tensors are
    aliased by numpy arrays (see details in `rlplay.utils.schema.shared`) for
    zero-copy data interchange.
    """
    # `fragment` contains [T x B x ?] aliased torch tensors of actor/env
    # infos, trajectory fragment $(x_{t-1}, a_{t-1}, r_t, d_t)_{t=1}^T$,
    # and bootstrap values $v(s_T)$, possibly allocated in torch's shared
    # memory.
    fragment = aliased(buffer)  # just a zero-copy pyt-npy alias

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

    # Create `context`, a dedicated container of [B x ?] aliased copy of
    # the data and the current recurrent state `hx` of the actor $h_t$, both
    # possibly residing in torch's pinned memory.
    context = aliased(Context(pyt, fragment.pyt.hx, pyt.obs),
                      copy=True, pinned=pinned)

    # writable view of `context.pyt.state` with an extra temporal dim
    unsafe_apply(context.pyt.state, fn=lambda x: x.unsqueeze_(0))  # in-place!
    # XXX we do this so that the actor may rely on T x B x ... data on input

    # `pyt` is used for interacting with the actor, `npy` -- with the fragment
    #  and both are just different interfaces to the same underlying data.
    # anon. torch storage <<--editable unsqueezed view-->> `pyt` torch tensors
    #        ditto        <<--__array__ data aliasing -->> `npy` numpy arrays
    return context, fragment


@torch.no_grad()
def context(*envs, pinned=False):
    r"""Allocate aliased running state for rollout collection.

    Parameters
    ----------
    *envs : gym.Env
        The environments used to initialize the structure and shapes of
        observation and action buffers in the context. Determines the number
        of environments in the context (dim=1).

        WARNING: Each environment is reset AT LEAST once. One environment is
        stepped through EXACTLY once using one action sampled from its space.

    pinned: bool, default=False
        The underlying storage of the newly created tensors resides in pinned
        memory (non-paged) for faster host-device transfers.

    Returns
    -------
    ctx : aliased State
        The running env-state context which contains properly time synchronised
        input data for the actor $x_t$, $a_{t-1}$, $r_t$, and $d_t$, EXCEPT for
        the recurrent state $h_t$.
        See docs of `startup` for details of `ctx` aliasing.

    Details
    -------
    This is a version of `startup()`, specialized for actor-less context
    initialization. It returns a simplified context, which contains only
    the State, e.g. the `obs-act-rew-fin` data.
    """

    env = envs[0]

    # prepare the running context from data some environment, which is reset.
    obs_ = env.reset()
    act_ = env.action_space.sample()
    _, rew_, fin_, _ = env.step(act_)

    # ensure correct data types for `rew_` (to float32) and `fin_` (to bool)
    state_ = State(obs_, act_, numpy.float32(rew_), bool(fin_))

    # torchify and alias, then add unit-time dim to `.pyt` in-place
    state = aliased(torchify(state_, len(envs), pinned=pinned, copy=True))
    unsafe_apply(state.pyt, fn=lambda x: x.unsqueeze_(0))

    # Flag the state as having just been reset, meaning that the previous
    #  reward and action are invalid.
    state.npy.fin[:] = True
    state.npy.rew[:] = 0.  # zero `.rew`, leave `.act` undefined
    for j, env in enumerate(envs):
        structured_setitem_(state.npy.obs, j, env.reset())  # x_0 = s_*

    return state


@torch.no_grad()
def collect(envs, actor, fragment, context, *, sticky=False, device=None):
    r"""Collect the trajectory fragment (rollout) by marching the actor and
    the environments, in lockstep (`actor` and `envs`, respectively) updating
    `context` and recording everything into `fragment`.

    Parameters
    ----------
    envs : list of gym.Env
        The stateful environments to step through.

    actor : Actor
        The stateful actor, which steps through a batch of environments.

    fragment : Aliased Fragment
        The buffer into which the trajectory fragment is recorded *in-place*.

    context : Aliased Context
        The most recent state for the actor to react to. *Updated in-place.*

    sticky : bool, default=False
        Whether to stop interacting with a terminated environment until the end
        of the current trajectory fragment, i.e. until the `collect()`. Once
        an env reached a terminal state, we put a record with its reset
        observation into the trajectory fragment and then stop interacting
        with it until the end of the fragment.

    device : torch.device, or None
        The device onto which to put the input data ($x_t$ obs, $a_{t-1}$ act,
        $r_t$ rew, $d_t$ fin, and $h_t$ hx) when steping through the environments.

    Details
    -------
    Let $s_t$ be the observed state at time $t$, $a_t$ the action taken at
    time $t$, $r_{t+1}$ -- the reward obtained due to $(s_t, a_t) \to s_{t+1}$
    transition, and $d_{t+1}$ an indicator whether the new state $s_{t+1}$ is
    terminal. Let $s_*$ be the observation just after an environment reset.
    The present value of the reward flow obtained by following the trajectory
    after the state $s_t$, i.e. $(a_t, s_{t+1}, a_{t+1}, ...)$, is defined as
    $
        G_t = \sum_{j=t}^{h-1} \gamma^{j-t} r_{j+1} 1_{\neg d_{j+1}}
            + \gamma^{h-t} G_h
    $
    for $h \geq t$ also known as `the t-th return'.

    The `.step` collects this data by stepping both through the actor and
    the environment in lockstep into the `out` buffer.

    The `fragment.state`, a.k.a. `out` buffer, contains the following data

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

    Since the rewards following any terminal state are always zero, because
    the episode has ended, such sates are not actionable, and there is no point
    in storing this $t+1$-st observation. Therefore $d_t = \top$ in `out[t-1]`
    indicates that instead of recording a terminal $s_t$ into `out[t]`, we have
    reset the environment and put an init observation $s_*$ into `out[t]`. On
    the other hand $d_t = \bot$ in `out[t-1]` means that the observation data
    in `out[t]` is the actual observation $s_t$ following $s_{t-1}$.

    This non-intuitive indexing notation allows computing the return as

        G[j] = out[j].rew + gamma * (1 - out[j].fin) * G[j+1]

    for `j=0..T-1` where $G[j] \approx G_j$, and `G[T] = 0` if `out[T-1].fin`
    and `G[T] = v(s_T)` otherwise. The latter requires explicitly computing
    the value function at a non-terminal state $s_T$, when collection is
    interrupted mid-trajectory.

    The value function estimates the present value (returns)
    $
        v(s_t)
            \approx \mathbb{E}_{\tau \sim p \circ \pi} G_t(\tau)
    $
    with
    $
        \tau = (s_t, a_t, r_{t+1})_{t\geq 0}
    $
    and $T(\tau) = \inf\{t \geq 1 \colon s_t = \times \}$ -- the trajectory
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

    # copy the `n_overlap` records from the last fragment to the beginning
    #  of the current fragment
    n_overlap = 0

    # write the initial recurrent state of the actor to the shared buffer
    structured_tensor_copy_(fragment.pyt.hx, context.pyt.hx)

    # shorthands for fast access
    # `out[t]` is $x_t, a_t, r_{t+1}$, and $d_{t+1}$, for $x_t$ above
    out, hx = fragment.npy.state, context.pyt.hx
    npy_next_obs = context.npy.next_obs

    # `pyt/npy` is its jagged edge: $x_t, a_{t-1}, r_t$, $d_t$, and $h_t$.
    npy, pyt = context.npy.state, context.pyt.state

    # Allocate on-device context and recurrent state, if device is not None
    pyt_ = pyt
    if device is not None:
        pyt_, hx = unsafe_apply((pyt_, hx), fn=lambda x: x.to(device))

    # XXX stepper uses a single actor for a batch of environments:
    # * one mode of exploration, poor randomness
    for t in range(n_overlap, len(out.fin)):  # `fin` is (T + H) x B
        # copy $s_t$ to out[t]
        structured_setitem_(out.obs, t, npy.obs)

        # move the updated context to its device-resident copy (`hx` is OK)
        if pyt_ is not pyt:
            structured_tensor_copy_(pyt_, pyt)

        # $a_t$ is a reaction to $s_t, a_{t-1}, r_t, d_t$ (`npy/pyt`) and `hx`
        # XXX the actor's outputs respect time and batch dims, except `hx`
        act_, hx, info_actor = actor.step(pyt_.obs, pyt_.act,
                                          pyt_.rew, pyt_.fin, hx=hx)
        # XXX The actor MUST NOT update or change the inputs and `hx` in-place,
        #  and must only newly created tensors (if it wished to do so).
        # XXX The `hx` of an environment that has the corresponding `.fin` flag
        # set SHOULD NOT be updated (dim=1).

        # the actor may return device-resident tensors, so we copy them here
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
            structured_setitem_(npy_next_obs, j, obs_)
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
        if hasattr(out, 'next_obs'):
            structured_setitem_(out.next_obs, t, npy_next_obs)

    # write back the most recent recurrent state for the next rollout
    structured_tensor_copy_(context.pyt.hx, hx)

    # push the last observation into the special T+1-th slot. This is enough
    # for DQN since obs[t] (x_t) and obs[t+1] (x_{t+1}) are consecutive if
    # fin[t] is False (d_{t+1}=\bot), and DQN methods ignore the target q-value
    # at x_{t+1} if it is terminal (d_{t+1}=\top and x_{t+1}=s_*).
    structured_setitem_(out.obs, t + 1, npy.obs)

    # compute the bootstrapped value estimate for each env
    if hasattr(fragment.pyt, 'bootstrap'):
        if pyt_ is not pyt:
            structured_tensor_copy_(pyt_, pyt)
        bootstrap = actor.value(pyt_.obs, pyt_.act, pyt_.rew, pyt_.fin, hx=hx)
        structured_tensor_copy_(fragment.pyt.bootstrap, bootstrap)

    return True


@torch.no_grad()
def evaluate(envs, actor, *, n_steps=None, render=False, device=None):
    """Evaluate the actor module in the environment.

    Parameters
    ----------
    envs : list of gym.Env
        The evaluation environments to step through.

    actor : BaseActorModule
        The stateful actor, the recurrent context of which is to be reset.

    n_steps : int, default=None
        The maximum nuber of steps to take in each test environment. If `None`,
        then the limit is lifted.

    render : bool, default=False
        Whether to render the visuzlization of the envronment interaction.

        WARNING: can only be used in len(envs) == 1

    device : torch.device, default=None
        The device onto which to put the input data ($x_t$ obs, $a_{t-1}$ act,
        $r_t$ rew, $d_t$ fin, and $h_t$ hx) for the actor when steping through
        the test environments.

    Returns
    -------
    rewards : numpy.array
        The sum of the obtained rewards accumulated during the rollout in each
        test environment.

    Details
    -------
    This function is very similar to `collect()`, except that it does not
    record the rollout data, except for the rewards from the environment.
    """

    n_steps = n_steps or float('+inf')
    assert len(envs) == 1 or len(envs) > 1 and not render

    # always pin the runtime context if the device is 'cuda'
    device = torch.device('cpu') if device is None else device
    pinned = device.type == 'cuda'

    # prepare a for the specified number of envs running context
    ctx = context(*envs, pinned=pinned)

    # fast access to context's aliases
    npy, pyt, hx = ctx.npy, ctx.pyt, None

    # Allocate on-device context and recurrent state, if device is not None
    pyt_ = pyt
    if device is not None:
        pyt_ = unsafe_apply(pyt_, fn=lambda x: x.to(device))

    # render ony in case of a single-env evaluation
    fn_render = envs[0].render if len(envs) == 1 and render else lambda: True

    rewards, done, t = [], False, 0
    while not done and t < n_steps and fn_render():
        # move the updated context to its device-resident copy
        if pyt_ is not pyt:
            structured_tensor_copy_(pyt_, pyt)

        # actor's step: commit `a_t` into the running context
        act_, hx, _ = actor.step(pyt_.obs, pyt_.act, pyt_.rew, pyt_.fin, hx=hx)
        structured_tensor_copy_(pyt.act, act_)

        # `.step` through a batch of envs
        for j, env in enumerate(envs):
            # ease interacting with terminated envs
            if npy.fin[j] and t > 0:
                npy.rew[j] = 0.
                continue

            # env's step: commit $x_{t+1}$, $r_{t+1}$, and $d_{t+1}$
            obs_, rew_, fin_, info_env = env.step(npy.act[j])

            structured_setitem_(npy.obs, j, obs_)
            npy.rew[j], npy.fin[j] = rew_, fin_

        # stop only if all environments have been terminated
        done = numpy.all(npy.fin)

        # track rewards only
        rewards.append(npy.rew.copy())
        t += 1

    return numpy.stack(rewards, axis=0)
