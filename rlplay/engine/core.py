import torch
import numpy

from collections import namedtuple

from .utils.plyr import suply, tuply, setitem, getitem
from .utils.shared import aliased, torchify


State = namedtuple('State', ['obs', 'act', 'rew', 'fin'])
State.__doc__ = r"""The current state to be acted upon: the current observation
    $x_t$, the past action $a_{t-1}$, the most recently received reward $r_t$,
    and the termination flag $d_t$:

        $(x_t, a_{t-1}, r_t, d_t)$

    where the boolean $d_t$ indicates if a new trajectory has been started, and
    $r_t$ and $a_{t-1}$ must be disregarded as unrelated.

    Attributes
    ----------
    obs : nested container of tensors with shape (batch, ...)
        A batch of observations $x_t$ from the environment containing either
        the current observation resulting from past actions and transitions,
        or the initial observation $x_*$ from a new trajectory. Which is which
        is determined by the corresponding boolean flag in `.fin`.

    act : nested container of tensors with shape (batch, ...)
        The batch of the most recent actions $a_{t-1}$, taken in each
        environment, which caused the transition to $(x_t, s_t)$). Must be
        ignored upon the reset of an environment, which is indicated by
        the corresponding boolean flag in `.fin`.

    rew : tensor of float with shape (batch,)
        The batch of the most recently received rewards $r_t$ obtained from
        transitioning from $s_{t-1}$ to $(s_t, x_t)$ by acting $a_{t-1}$. Like
        `.act` must be ignored whenever the corresponding boolean flag in `.fin`
        is set, which indicates a reset.

    fin : tensor of bool with shape (batch,)
        The batch of boolean flags $d_t$ indicating that the previous action
        $a_{t-1}$ caused a transition to a terminal state (true $s_t$) in
        the corresponding environment, upon which it was automatically reset
        to $s_*$, which overwrote $x_t$ in `.obs` with $x_*$. If False, then
        `.obs` contains the true current observation $x_t$.

    See Also
    --------
    See docs for `BaseActorModule.forward`.
    """

Context = namedtuple('Context', ['state', 'hx', 'next_obs'])
Context.__doc__ = r"""This is a convenience object, that extends `State` by
    an `hx` and a `next_obs` containers of tensors. The first represents the
    persistent context, e.g. the recurrent state, of the actor at the start of
    a trajectory fragment (rollout). The second stores the original true next
    observation $x_{t+1}$ before any reset took place.
    """

Fragment = namedtuple('Fragment', ['state', 'actor', 'env', 'bootstrap', 'hx'])
Fragment.__doc__ = r"""A `(1 + T) x batch` fragment of the trajectories of
    the actor in a batch of environments and relevant information, e.g. rnn
    states, value estimates, policy logits, etc..

    Attributes
    ----------
    state : nested container of tensors with shape (1+T, batch, ...)
        The obs-act-rew-fin state with the same semantics as described in
        `State`.

    actor : nested container of tensors with shape (T, batch, ...)
        The extra information returned by the actor's `.step` along with
        the action $a_t$ in response to $(x_t, a_{t-1}, r_t, d_t, h_t)$,
        such as the value estimates $v(s_t)$ or rnn state $h_{t+1}$.

    env : nested container of tensors with shape (T, batch, ...)
        The extra data received from the environments' `.step` upon taking
        the actions $a_t$ in them.

    bootstrap : tensor of float with shape (batch,)
        A vector of value estimates $v(s_T)$ used to bootstrap the present
        value of the reward flow following the state $s_T$, which falls beyond
        the fragment $(s_{t+1}, a_t, r_{t+1}, d_{t+1})_{t=0}^{T-1}$ recorded in
        `.state`, but is related to $r_T$ and $d_T$. If a flag in `.state.fin`
        is set, then the corresponding bootstrap value should be ignored when
        computing the returns, since the reward flow from any terminal state
        is identically zero.

    hx : container of tensor, or None
        The most recent recurrent state of the actor $h_t$, which conditions
        its response along with the `.obs`, `.act`, `.rew` inputs similarly to
        $(s_t, a_{t-1}, r_t, d_t; h_t) \to (a_t, h_{t+1})$.

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


def tensor_copy_(dst, src, *, at=None):
    """Copy tensor data from the `src` nested container into torch tensors
    of the `dst` nested container (with IDENTICAL structure) at the specified
    index (int, tuple of ints, or slices).
    """
    if at is not None:
        dst = suply(getitem, dst, index=at)

    suply(torch.Tensor.copy_, dst, src)


def numpy_copy_(dst, src, *, at=None):
    """Copy numpy data between nested containers with IDENTICAL structure."""
    if at is not None:
        dst = suply(getitem, dst, index=at)

    suply(numpy.copyto, dst, src, casting='same_kind')


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
        we assume the initial recurrent state to be a ZERO non-differentiable
        tensor.

        The seemingly awkward shapes of the tensors in `hx` are rooted in the
        shapes if hidden states of torch.nn recurrent layers. For example,
        the LSTM layer has `hx = (c, h)`, both tensors shaped like

            (num_layers * num_directions) x n_batch x n_hidden

        (with the GRU layer's `hx` being just a single tensor, not a tuple).
        """
        assert hx is not None

        suply(lambda x: x[:, at].zero_(), hx)
        # XXX could the actor keep `hx` unchanged in `forward`? No, under the
        # current API, since `.fin` reflects that the input is related to a
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
            the actor.

            Components of `act` SHOULD be ignored if the corresponding flag
            in `fin` is set, because the associated environment has just been
            reset, and a new trajectory is being rolled out.

        rew : float tensor, shape = `n_seq, n_envs,`
            The reward that each environment yielded due to the transition to
            $(s_t, x_t)$ after taking action $a_{t-1}$ at $(s_{t-1}, x_{t-1})$.

            Components of `rew` MUST be ignored if the corresponding flag in
            `fin` is set.

        fin : bool tensor, shape = `n_seq, n_envs,`
            The reset flags $d_t$ indicate which of the previous actions in
            `act` and last rewards in `rew` have been invalidated by a reset
            of the corresponding environment. Also indicates if the observation
            $x_t$ is the initial observation $x_*$ after a reset (True), or
            a non-terminal observation $x_t$ (False).

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
            (act), the recently obtained reward $r_t$ (rew), the termination
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

    # create `1 x n_envs` buffer for bootstrap value estimate $v(s_T)$
    bootstrap = torchify(rew_, 1, n_envs, shared=shared, pinned=pinned)

    # the buffer for the aux env info data is `n_steps x n_envs x ...`
    d_env_info = torchify(d_env_info, n_steps, n_envs,
                          shared=shared, pinned=pinned)

    # allocate `(1 + n_steps) x n_envs x ...` torch tensor buffers for
    #  the observations, actions, rewards and termination flags. So that
    #  they can be folded
    state = torchify(State(obs_, act_, rew_, fin_),
                     1 + n_steps, n_envs, shared=shared, pinned=pinned)

    # make a single pass through the actor with one `1 x n_envs x ...` batch
    pyt = suply(lambda x: x[:1].to(device), state)
    unused_act, hx, d_act_info = actor.step(pyt.obs, pyt.act,
                                            pyt.rew, pyt.fin, hx=None)
    # XXX `act_` is expected to have identical structure to `unused_act`
    # XXX `d_act_info` must respect the temporal and batch dims

    # get one time slice from the actor's info `n_envs x ...` and expand into
    #  an `n_steps x n_envs x ...` structured buffer.
    d_act_info = torchify(suply(lambda x: x[0].cpu(), d_act_info),
                          n_steps, shared=shared, pinned=pinned)

    # the actor fully specifies its context `hx`, so we torchify it as is
    hx = torchify(suply(torch.Tensor.cpu, hx), shared=shared, pinned=pinned)

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
    # `fragment` contains aliased torch tensors of actor/env infos (`.actor`
    #  and `.env` [T x B x ...]), rollout $(x_t, a_{t-1}, r_t, d_t)_{t=0}^T$
    #  (`.state` [1 + T x B x ...]), and the bootstrap values $v(s_T)$, all
    #  possibly residing in torch's shared memory.
    fragment = aliased(buffer)  # just a zero-copy pyt-npy alias

    # reset the initial recurrent state of the actor, `hx`, to zero
    suply(torch.Tensor.zero_, fragment.pyt.hx)

    # Fetch a single [B x ?] observation (a VIEW into fragment for now)
    npy, pyt = suply(getitem, (fragment.npy.state, fragment.pyt.state,),
                     index=0)

    # Flag the state as having just been reset, meaning that the previous
    #  reward and action are invalid.
    npy.fin[:] = True
    npy.rew[:] = 0.  # zero `.rew`, leave `.act` undefined
    for j, env in enumerate(envs):
        suply(setitem, npy.obs, env.reset(), index=j)  # x_0 = s_*
        actor.reset(j, fragment.pyt.hx)  # h_0 = h_*

    # Create `context`, a dedicated container of [B x ?] aliased copy of
    # the data and the current recurrent state `hx` of the actor $h_t$, both
    # possibly residing in torch's pinned memory.
    context = aliased(Context(pyt, fragment.pyt.hx, pyt.obs),
                      copy=True, pinned=pinned)

    # writable view of `context.pyt.state` with an extra temporal dim
    suply(torch.Tensor.unsqueeze_, context.pyt.state, dim=0)  # in-place!
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
    suply(torch.Tensor.unsqueeze_, state.pyt, dim=0)

    # Flag the state as having just been reset, meaning that the previous
    #  reward and action are invalid.
    state.npy.fin[:] = True
    state.npy.rew[:] = 0.  # zero `.rew`, leave `.act` undefined
    for j, env in enumerate(envs):
        suply(setitem, state.npy.obs, env.reset(), index=j)  # x_0 = s_*

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
    Let $s_t$ be the environment's state at time $t$ (inaccessible), and $x_t$,
    $r_t$ be the observation and the reward emitted by the env's transition to
    $
        (s_{t-1}, a_{t-1}) \to (s_t, x_t, r_t, d_t, E_t)
    $,
    with $d_t$ indicating if $s_t$ is terminal and $E_t$ is the env's extra
    info. The action $a_t$ is taken at in response to the current reward $r_t$
    and observation $x_t$, the last action $a_{t-1}$ and actor's current
    internal state $h_t$:
    $
        (x_t, a_{t-1}, r_t, d_t, h_t) \to (a_t, h_{t+1}, A_{t+1})
    $,
    with $A_{t+1}$ being the actors's extra info computed on the history up to
    and including $t$ and related to the composite `actor-env` transition
    $
        s_t \longrightarrow a_t \longrightarrow s_{t+1}
    $.
    Let $(s_*, x_*)$ be the true state and the observation just after a reset
    of an environment.

    For simplicity assume that $x_t = s_t$, i.e. the env is fully observed. Let
    $h > t$. The present value $G_\pi(s_t)$ of the reward flow obtained by
    following the policy $\pi$ starting at $s_t$, i.e.
    $
        \tau = (a_t, r_{t+1}, s_{t+1}, a_{t+1}, ..., a_{h-1}, s_h, r_h)
    $,
    with stochastic transition dynamics
    $
        a_j \sim \pi_j(a \mid s_j)
    $
    and
    $
        s_{j+1}, r_{j+1} \sim p(s, r \mid s_j, a_j)
    $
    for $j=t...h-1$ is defined as `the t-th return'
    $
        G_\pi(s_t)
            = \mathbb{E}_\tau
                \sum_{j=t}^{h-1} \gamma^{j-t} r^\dagger_{j+1}
                + \gamma^{h-t} G_\pi(s^\dagger_h)
    $,
    provided the trajectory hasn't been is terminated mid-way (otherwise we
    consider a `ceased` reward and a stopped observation processes, i.e.
    $r^\dagger$ and $s^\dagger$ are `frozen` at the **stopping time**
    $T(s_t) \geq t$ with
    $
        r^\dagger_j = 0 if j > T(s_t) else r_j
    $
    and
    $
        s^\dagger_j = s_{T(s_t) \wedge j}
    $.
    Essentially we have
    $
        d_j = 1_{j \geq T(s_t)}
    $.
    This is because the rewards following terminal states are assumed to always
    be zero, since the episode has ended.

    This data is collected by stepping through the actor and the environment in
    lockstep and recording it into the `fragment` rollout buffer and keeping
    updated `context`. The simplified transitions
    $
        (s_t, a_t) \to s_{t+1}
    $
    determine the timing in the subscripts. The `fragment.state`, aka `out`,
    with a terminal observation occurring at relative time $t+1$, contains
    the following data:

    if `sticky` is False

      +---+-----+-----------------------------------------+-----------+
      |   |     |               .state                    | (env's    |
      |   |  #  +-----------------------------------------+  actual   |
      |   |     |  obs       act       rew       fin      |   state)  |
      +---+-----+-----------------------------------------+-----------+
      | f |     |                                         |           |
      | r |   0 |  x_k       a_{k-1}   r_k       d_k      |  s_k      |
      | a | ... |                                         |           |
      | g |   t |  x_t       a_{t-1}   r_t       \bot     |  s_t      |
      | m | t+1 |  x'_*      a_t       r_{t+1}   \top     |  s'_*  <<-- reset
      | e | t+2 |  x'_1      a'_0      r'_1      \bot     |  s'_1     |
      | n | ... |                                         |           |
      | t | N-1 |  x'_{j-1}  a'_{j-2}  r'_{j-1}  d'_{j-1} |  s'_{j-1} |
      |   |   N |  x'_j      a'_{j-1}  r'_j      d'_j     |  s'_j   ------+
      | p |     |                                         |           |   |
      +---+-----+-----------------------------------------+-----------+ clone
      |   |     |                                         |           |   |
      | p |   0 |  x'_j      a'_{j-1}  r'_j      d'_j     |  s'_j  <<-----+
      | + |   1 |  x'_{j+1}  a'_j      r'_{j+1}  d'_{j+1} |  s'_{j+1} |
      | 1 | ... |                                         |           |
      |   |     |                                         |           |
      +---+-----+-----------------------------------------+-----------+

      (the apostrophe indicates a new trajectory within the same fragment)

    Notice that the record `out[t]` still contains a VALID last action $a_t$
    and the true received terminal reward $r_{t+1}$. However this compact data
    recording does not contain the true terminal observation $x_t$, because it
    has been overwritten by $x_*$.

    In short, if `out.state.fin[t]` is True, then the action `out.state.act[t]`
    taken from `out.state[t-1]` has led to a terminal state with reward
    `out.state.rew[t]`, and the observation in `out.state.obs[t]` is already
    the initial observation $x_*$ form a newly started trajectory. Otherwise,
    if `out.state.fin[t]` is False, then `out.state[t-1]` and `out.state[t]`
    are truly consecutive records from the same trajectory.

    This non-intuitive indexing notation allows computing the return for the
    action $a_t$ in `out.state.act[t+1]` using

        G[t] = out.state.rew[t+1] + gamma * (1 - out.fin[t+1]) * G[t+1]

    where `G[t] $\approx G_\pi(s_t)$`, and `G[N] = 0` if `out.state.fin[T]` is
    True, and otherwise `G[T]` is $v(s_N)$, i.e. the bootstrap value --
    an estimate of future return $G_\pi(s_{N+1})$.

    If `sticky` is True, then any interaction with the terminated environment
    is ceased until the start of the next fragment.

      +---+-----+-----------------------------------------+-----------+
      |   |     |               .state                    | (env's    |
      |   |  #  +-----------------------------------------+  actual   |
      |   |     |  obs       act       rew       fin      |   state)  |
      +---+-----+-----------------------------------------+-----------+
      | f |     |                                         |           |
      | r |   0 |  x_k       a_{k-1}   r_k       d_k      |  s_k      |
      | a |    ...                                       ...          |
      | g |   t |  x_t       a_{t-1}   r_t       \bot     |  s_t      |
      | m | t+1 |  x'_*      a_t       r_{t+1}   \top     |  s'_*  <<-- reset
      | e | t+2 |  x'_*      a_t       0         \top     |  s'_*     |
      | n |    ...                                       ...          |
      | t | N-1 |  x'_*      a_t       0         \top     |  s'_*     |
      |   |   N |  x'_*      a_t       0         \top     |  s'_*  -------+
      | p |     |                                         |           |   |
      +---+-----+-----------------------------------------+-----------+ clone
      |   |     |                                         |           |   |
      | p |   0 |  x'_0      a_{-1}    r_0       \top     |  s_0  <<------+
      | + |   1 |  x'_1      a_0       r_1       d_1      |  s_1      |
      | 1 |    ...                                       ...          |
      |   |     |                                         |           |
      +---+-----+-----------------------------------------+-----------+

    This option is more friendly towards CUDNN and torch's packed sequences,
    but it iseems that manuall stepping through time is more efficient.

    Below we depict the synchronisation of the records in the rollout fragment.
    The horizonal arrows indicate the sub-steps of the composite `actor-env`
    transition.

      +---+-----+---------------------------------------+-----------+
      |   |  #  |  .state   .hx  ->  .actor -> .env     | actual    |
      +---+-----+---------------------------------------+-----------+
      | f |     |                                       |           |
      | r |   0 |  Z_k      h_k      A_{k+1}   E_{k+1}  |  s_{k+1}  |
      | a |    ...                                     ...          |
      | g |   t |  Z_t      h_t      A_{t+1}   E_{t+1}  |  s'_*   <<-- reset
      | m | t+1 |  Z'_0     h'_*     A'_1      E'_1     |  s'_1     |
      | e | t+2 |  Z'_1     h'_1     A'_2      E'_2     |  s'_2     |
      | n |    ...                                     ...          |
      | t | N-1 |  Z'_{j-1} h'_{j-1} A'_j      E'_j     |  s'_j     |
      |   |   N |  Z'_j     h'_j                        |           |
      | p |     |   |        |                          |           |
      +---+-----+---|--------|--------------------------+-----------+
      |   |     |   V        V                          |           |
      | p |   0 |  Z'_j     h'_j     A'_{j+1}  E'_{j+1} |  s'_{j+1} |
      | + |   1 |  Z'_{j+1} h'_{j+1} A'_{j+2}  E'_{j+2} |  s'_{j+2} |
      | 1 |    ...                                     ...          |
      |   |     |                                       |           |
      +---+-----+---------------------------------------+-----------+

    (the evolution of `.hx` is not recorded, only its initial value $h_k$)

    To summarize
      * `.state[t]` -->> `.state[t+1].act` and `.actor[t]`
      * `.state[t], .state[t+1].act` -->> `.env[t]` and rest of `.state[t+1]`
    """
    device = torch.device('cpu') if device is None else device
    # assert isinstance(device, torch.device)
    on_host = device.type == 'cpu'

    # write the initial recurrent state of the actor to the shared buffer
    tensor_copy_(fragment.pyt.hx, context.pyt.hx)

    # shorthands for fast access
    # `out[t]` is $x_t, a_{t-1}, r_t$, and $d_t$, for $x_t$ defined above
    out, hx = fragment.npy.state, context.pyt.hx
    npy_next_obs = context.npy.next_obs

    # `pyt/npy` is $x_t, a_{t-1}, r_t$, $d_t$, and $h_t$
    npy, pyt = context.npy.state, context.pyt.state

    # Allocate on-device context and recurrent state, if device is not `host`
    pyt_ = pyt
    # XXX even if the device is host, `suply` creates a new nested
    #  contianer, which makes the check on L825 always succeed.
    if not on_host:
        # XXX this also copies data in `pyt` into `pyt_`
        pyt_, hx = suply(torch.Tensor.to, (pyt_, hx), device=device)

    # XXX stepper uses a single actor for a batch of environments:
    # * one mode of exploration, poor randomness
    for t in range(len(out.fin) - 1):  # `fin` is (1 + T) x B
        # After each iteration we construct the state `t+1` from `t`:
        # * `.state[t]` is $(x_t, a_{t-1}, r_t, d_t)$
        # * `.actor[t], hx` are actor's response to `.state[t]` and $h_t$
        # * `.env[t]` is env's info from the $s_t, a_t \to s_{t+1}$ step
        # * `context` is $(x_{t+1}, a_t, r_{t+1}, d_{t+1})$
        # * `hx` is $h_{t+1}$
        pass

        # copy the state $x_t, a_{t-1}, r_t$, and $d_t$ from `ctx` to `out[t]`
        suply(setitem, out, npy, index=t)  # XXX is torch faster?
        if hasattr(out, 'next_obs'):
            suply(setitem, out.next_obs, npy.next_obs, index=t)

        # REACT: $(a_t, h_{t+1})$ are actor's reaction to `.state[t]` and `hx`,
        #  i.e. $(x_t, a_{t-1}, r_t, d_t)$, and $h_t$, respectively.
        # XXX the actor's outputs respect time and batch dims, except `hx`
        act_, hx, info_actor = actor.step(pyt_.obs, pyt_.act,
                                          pyt_.rew, pyt_.fin, hx=hx)
        # XXX The actor MUST NOT update or change the inputs and `hx` in-place,
        #  and must only newly created tensors (if it wished to do so).
        # XXX The `hx` of an environment that has the corresponding `.fin` flag
        # set SHOULD NOT be updated (dim=1).

        # the actor may return device-resident tensors, so we copy them here
        tensor_copy_(pyt.act, act_)  # commit $a_t$ into `ctx`
        if info_actor:
            # fragment.pyt is likely to have `is_shared() = True`, so it cannot
            #  be in the pinned memory.
            tensor_copy_(fragment.pyt.actor, info_actor,
                         at=slice(t, t + 1))  # `.actor[t] <- info`

        # STEP + EMIT: `.step` through a batch of envs
        for j, env in enumerate(envs):
            # Only recorded interactions can get stuck: if `.fin` is True,
            #  when `t=0`, this means the env has been reset elsewhere.
            if sticky and t > 0 and npy.fin[j]:
                # `npy = (s_*, ?, 0, True)` (`.obs` and `.fin` are stuck)
                npy.rew[j] = 0.

                # hx is not stuck, so it needs to be reset
                actor.reset(j, hx)
                continue

            # get $(s_t, a_t) \to (s_{t+1}, x_{t+1}, r_{t+1}, d_{t+1})$
            obs_, rew_, fin_, info_env = env.step(npy.act[j])
            suply(setitem, npy_next_obs, obs_, index=j)
            if info_env:
                suply(setitem, fragment.npy.env, info_env,
                      index=(t, j))  # `.env[t, j] <- info`

            # `fin_` indicates if `obs_` is terminal and a reset is needed
            # XXX DO NOT alter the received reward from the terminal step!
            if fin_:
                # substitute the terminal observation $x_{t+1}$ with an initial
                #  $x_*$, reset the (unobserved) $s_{t+1}$ to $s_*$ and zero
                #  the actor's recurrent state $h_* \to h_{t+1}$ at env $j$
                obs_ = env.reset()  # s_{t+1} \to s_*, emit x_* from s_*
                actor.reset(j, hx)  # reset h_{t+1} to h_0 as j-th env

            # update the j-th env in the running context:
            #  commit $x_{t+1}, r_{t+1}, d_{t+1}$ into `ctx`
            suply(setitem, npy.obs, obs_, index=j)
            npy.rew[j] = rew_  # used in present value even if $d_{t+1}=\top$
            npy.fin[j] = fin_

        # copy the updated `ctx` to its device-resident copy (`hx` is OK)
        if pyt_ is not pyt:
            tensor_copy_(pyt_, pyt)

    # write back the most recent recurrent state for the next rollout
    tensor_copy_(context.pyt.hx, hx)

    # record the $(x_T, a_{T-1}, r_T, d_T)$ from `ctx` into `out[T]`
    suply(setitem, out, npy, index=t + 1)  # t is len(out.fin) - 1
    # XXX This is OK for DQN-methods and SARSA since `out[t]` and `out[t+1]`
    #  are consecutive if $d_{t+1}$ (`out.fin[t+1]`) is False, and together
    #  contain $x_t, a_t, r_{t+1}$ and $x_{t+1}$. Also DQN methods ignore
    #  the target q-value at $x_{t+1}$ anyway if $s_{t+1}$ is terminal, i.e.
    #  $d_{t+1}=\top$, and x_{t+1}=s_*.

    if hasattr(out, 'next_obs'):
        suply(setitem, out.next_obs, npy_next_obs, index=t + 1)

    # compute the bootstrapped value estimate for each env
    if hasattr(fragment.pyt, 'bootstrap'):
        bootstrap = actor.value(pyt_.obs, pyt_.act, pyt_.rew, pyt_.fin, hx=hx)
        tensor_copy_(fragment.pyt.bootstrap, bootstrap)

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
        Whether to render the visualization of the environment interaction.

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
    pinned, on_host = device.type == 'cuda', device.type == 'cpu'

    # prepare a running context for the specified number of envs
    ctx = context(*envs, pinned=pinned)
    # `ctx` is $x_*, a_{-1}, r_0, \top, h_0$, where `r_0` is undefined

    # fast access to context's aliases
    npy, pyt = ctx.npy, ctx.pyt

    # Allocate an on-device context and recurrent state, if not on 'host'
    pyt_ = pyt
    if not on_host:
        # XXX this also copies data in `pyt` into `pyt_`
        pyt_ = suply(torch.Tensor.to, pyt_, device=device)

    # render ony in case of a single-env evaluation
    fn_render = envs[0].render if len(envs) == 1 and render else lambda: True

    # collect the evaluation data: let the actor init `hx` for us
    rewards, done, t, bootstrap, hx = [], False, 0, None, None
    while not done and t < n_steps and fn_render():
        # REACT: $(x_t, a_{t-1}, r_t, d_t, h_t) \to a_t$ and commit $a_t$
        act_, hx, info_actor = actor.step(pyt_.obs, pyt_.act,
                                          pyt_.rew, pyt_.fin, hx=hx)
        tensor_copy_(pyt.act, act_)

        # fetch the bootstrap value $v(x_t)$ (a new `1 x n_envs` tensor)
        value = info_actor['value'].cpu().numpy()[0]
        if bootstrap is not None:
            # update according to the mask of terminated envs
            numpy.copyto(bootstrap, value, where=~npy.fin)

        else:
            bootstrap = value

        # STEP + EMIT: `.step` through a batch of envs
        for j, env in enumerate(envs):
            # cease interaction with terminated envs
            if npy.fin[j] and t > 0:
                npy.rew[j] = 0.
                continue

            # get $(s_t, a_t) \to (s_{t+1}, x_{t+1}, r_{t+1}, d_{t+1})$
            obs_, rew_, fin_, info_env = env.step(npy.act[j])

            # update the j-th env's '$x_{t+1}, r_{t+1}, d_{t+1}$ in `ctx`
            suply(setitem, npy.obs, obs_, index=j)
            npy.rew[j], npy.fin[j] = rew_, fin_

        # stop only if all environments have been terminated
        done = numpy.all(npy.fin)

        # track rewards only
        rewards.append(npy.rew.copy())
        t += 1

        # move the updated `ctx` to its device-resident torch copy
        if pyt_ is not pyt:
            tensor_copy_(pyt_, pyt)

    return numpy.stack(rewards, axis=0), bootstrap
