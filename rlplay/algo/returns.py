import numpy
import torch


# returns, baselined or not, or advantage estimates are not diff-able in PG
def npy_returns(rew, fin, *, gamma, bootstrap=0., omega=None, r_bar=None):
    r"""Compute the on-policy returns (the present value of the future rewards).

        G_t = r_{t+1}
              + \gamma \omega_{t+1} r_{t+2}
              + \gamma^2 \omega_{t+1} \omega_{t+2} r_{t+3} + ...
            = \sum_{j\geq t} r_{j+1} \gamma^{j-t} \prod_{s=t+1}^j \omega_s
            = r_{t+1} + \gamma \omega_{t+1} G_{t+1}
    """
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    n_steps, *shape = rew.shape
    G_t = numpy.zeros((1 + n_steps, *shape), dtype=rew[-1].dtype)

    if omega is not None:
        # \rho_t = \min\{ \bar{\rho}, \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
        rho = numpy.minimum(numpy.exp(omega), r_bar or float('+inf'))
        rho = rho.reshape(*rho.shape, *trailing)

    # rew[t], fin[t] is r_{t+1} and d_{t+1}
    G_t[-1] = bootstrap
    for j in range(1, n_steps + 1):
        # get G_t = r_{t+1} + \gamma \rho_t G_{t+1} 1_{\neg d_{t+1}}
        # XXX G_t[-j-1] is all zeros
        numpy.multiply(G_t[-j], gamma, where=~fin[-j], out=G_t[-j-1])
        if omega is not None:
            G_t[-j-1] *= rho[-j]

        G_t[-j-1] += rew[-j]

    return G_t[:-1]


def npy_multistep(
    rew,
    fin,
    val,
    *,
    gamma,
    n_lookahead=None,
    bootstrap=0.,
    omega=None,
    r_bar=None,
):
    r"""Compute the h-lookahead multistep returns bootstrapped with values.

        G_t = r_{t+1}
              + \gamma \omega_{t+1} r_{t+2}
              + \gamma^2 \omega_{t+1} \omega_{t+2} r_{t+3} + ...
            = \sum_{j\geq t} r_{j+1} \gamma^{j-t} \prod_{s=t+1}^j \omega_s
            = \sum_{j=0}^{h-1} \gamma^j r_{t+j+1} \prod_{s=1}^j \omega_{t+s}
              + \gamma^h \prod_{s=1}^h \omega_{t+s}
                  G_{t+h}
            \approx
                \sum_{j=0}^{h-1}
                    \gamma^j \Bigl( \prod_{s=1}^j \omega_{t+s} \Bigr)
                    r_{t+j+1}
                + \gamma^h \Bigl( \prod_{s=1}^h \omega_{t+s} \Bigr)
                  v_{t+h}
    """
    # r(t) = rew[t] = r_{t+1}, ditto for d = fin,
    # v(t) = val[t] if t < T, bsv if t=T, 0 o/w
    # let op F be def-nd as
    #     (F x)(t) := \omega_{t+1} d_{t+1} x(t+1) = d[t] * x[t+1]
    # then the m-step lookahead bootstrapped value estimate is
    #     v_0 = v, v_{j+1} = r + \gamma F v_j, j=0..h-1
    # or after unrolling:
    #     v_h = \sum_{j=0}^{h-1} \gamma^j F^j r + F^h v
    n_steps, *shape = rew.shape
    n_lookahead = n_lookahead or n_steps
    # XXX this function has at most the same complexity as `npy_returns`
    #  for `n_lookahead = None`.

    # eff[t] = (~fin[t]) * gamma = 1_{\neg d_{t+1}} \gamma, t=0..T-1
    eff = numpy.where(fin, 0., gamma)
    if omega is not None:
        # \rho_t = \min\{ \bar{\rho}, \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
        eff *= numpy.minimum(numpy.exp(omega), r_bar or float('+inf'))

    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    eff = eff.reshape(*eff.shape, *trailing)

    # out[t] = val[t] = v(s_t), t=0..T-1, out[T] = bsv = v(s_T)
    # compute the multistep returns by shifting t to t-1 repeatedly
    out = numpy.concatenate([val, bootstrap], axis=0)  # assume bsv has len = 1
    # XXX no need for double buffering
    for _ in range(n_lookahead):
        # out[t] = rew[t] + eff[t] * out[t+1], t=0..T-1
        #        = r_{t+1} + \gamma 1_{\neg d_{t+1}} v(s_{t+1})
        out[:-1] = rew + eff * out[1:]  # out[-1] is to be kept intact!

    # do not cut off incomplete returns
    return out[:-1]


def npy_deltas(rew, fin, val, *, gamma, bootstrap=0., omega=None, r_bar=None):
    r"""Compute the importance weighted td-error estimates:

        \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
        # \delta^v_s = 0 for all s \geq t if d_t = \top
    """
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    n_steps, *shape = rew.shape

    a_hat = numpy.zeros_like(rew)
    a_hat[-1:] = bootstrap
    a_hat[:-1] = val[1:]

    # \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
    numpy.copyto(a_hat, 0., where=fin)  # `.putmask` is weird with broadcasting
    a_hat *= gamma
    a_hat += rew
    a_hat -= val

    if omega is not None:
        # \rho_t = \min\{ \bar{\rho}, \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
        rho = numpy.minimum(numpy.exp(omega), r_bar or float('+inf'))
        a_hat *= rho.reshape(*rho.shape, *trailing)

    return a_hat


def npy_gae(rew, fin, val, *, gamma, C, bootstrap=0.):
    r"""Compute the Generalized Advantage Estimator (C is `lambda`).

        \delta^v_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)

        A_t = \delta^v_t + (\gamma \lambda) \delta^v_{t+1}
              + (\gamma \lambda)^2 \delta^v_{t+2} + ...
            = \delta^v_t + \gamma \lambda A_{t+1} 1_{\neg d_{t+1}}
    """
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    n_steps, *shape = rew.shape

    gae_t = numpy.zeros((1 + n_steps, *shape), dtype=rew[-1].dtype)
    delta = numpy.zeros(shape, dtype=rew[-1].dtype)

    # rew[t], fin[t], val[t] is r_{t+1}, d_{t+1} and v(s_t)
    # t is -j, t+1 is -j-1 (j=1..T)
    for j in range(1, n_steps + 1):
        # \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
        # numpy.multiply(bootstrap, gamma, out=delta)
        # numpy.putmask(delta, fin[-j], 0.)
        numpy.multiply(bootstrap, gamma, out=delta, where=~fin[-j])
        bootstrap = val[-j]  # v(s_t) is next iter's bootstrap
        delta += rew[-j] - bootstrap

        # A_t = \delta_t + \lambda \gamma A_{t+1} 1_{\neg d_{t+1}}
        numpy.multiply(gae_t[-j], C * gamma, out=gae_t[-j-1], where=~fin[-j])
        gae_t[-j-1] += delta

        # reset delta for the next conditional multiply
        delta[:] = 0

    return gae_t[:-1]


def npy_vtrace(rew, fin, val, omega, *, gamma, r_bar, c_bar, bootstrap=0.):
    r"""Compute the V-trace value estimates ($n \to \infty$ limit):

        \delta^v_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
        # \delta^v_s = 0 for all s \geq t if d_t = \top
        # \hat{v}^n_s = 0 for all s \geq t if d_t = \top

        \hat{v}^n_t
            = v(s_t) + \sum_{j=t}^{t+n-1} \gamma^{j-t}
                       \delta^v_j \rho_j \prod_{p=t}^{j-1} c_p

            = v(s_t) + \gamma c_t \bigl( \hat{v}^n_{t+1} - v(s_{t+1}) \bigr)
                     + \rho_t \delta^v_t
                     - \gamma^n
                       \delta^v_{t+n} \rho_{t+n} \prod_{p=t}^{t+n-1} c_p

        \hat{v}^\infty_t
            = v(s_t) + \rho_t \delta^v_t + \gamma c_t \bigl(
                \hat{v}^\infty_{t+1} - v(s_{t+1}) \bigr) 1_{\neg d_{t+1}}

        where $c_j = \min\{e^\omega_j, \bar{c} \}$ and $
            \rho_j = \min\{e^\omega_j, \bar{\rho} \}
        $, $\omega_t = \log \pi(a_t \mid x_t) - \log \mu(a_t \mid x_t)$, and
        $\mu$ is the behavior policy, while $\pi$ is the target policy.

        Let $
            \hat{a}_t := \hat{v}^\infty_{t+1} - v(s_{t+1}
        $, then
        \hat{a}_t
            = \rho_t \delta^v_t
            + \gamma c_t \hat{a}_{t+1} 1_{\neg d_{t+1}}
    """
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    # clamp(max=a) is the same is min(..., a)
    rho = numpy.minimum(numpy.exp(omega), r_bar or float('+inf'))
    see = numpy.minimum(numpy.exp(omega), c_bar or float('+inf'))

    # V-trace uses importance weights to correct for off-policy PG
    n_steps, *shape = rew.shape

    a_hat = numpy.zeros((1 + n_steps, *shape), dtype=rew[-1].dtype)
    delta = numpy.zeros(shape, dtype=rew[-1].dtype)

    # rew[t], fin[t], val[t] is r_{t+1}, d_{t+1} and v(s_t)
    # t is -j, t+1 is -j-1 (j=1..T)
    rho = rho.reshape(*rho.shape, *trailing)
    see = see.reshape(*see.shape, *trailing)
    for j in range(1, n_steps + 1):
        # \rho_t \bigl( r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t) \bigr)
        numpy.multiply(bootstrap, gamma, out=delta, where=~fin[-j])
        delta += rew[-j] - val[-j]
        delta *= rho[-j]

        # A_t = \rho_t \delta_t + \c_t \gamma A_{t+1} 1_{\neg d_{t+1}}
        numpy.multiply(a_hat[-j], gamma, out=a_hat[-j-1], where=~fin[-j])
        a_hat[-j-1] *= see[-j]
        a_hat[-j-1] += delta

        # reset delta for the next conditional multiply
        bootstrap = val[-j]  # v(s_t) is next iter's bootstrap
        delta[:] = 0

    return a_hat[:-1] + val


@torch.no_grad()
def pyt_returns(rew, fin, *, gamma, bootstrap=0., omega=None, r_bar=None):
    r"""Compute the importance weighted present-value estimate:

        G_t = r_{t+1} + \gamma \rho_t G_{t+1} 1_{\neg d_{t+1}}
    """
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    bootstrap = torch.as_tensor(bootstrap)

    # v(s_t) ~ G_t = r_{t+1} + \gamma G_{t+1} 1_{\neg d_{t+1}}
    # r_{t+1}, s_{t+1} \sim p(r, s, \mid s_t, a_t), a_t \sim \pi(a \mid s_t)
    # d_{t+1} indicates if $s_{t+1}$ is terminal
    if omega is not None:
        # \rho_t = \min\{ \bar{\rho}, \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
        rho = omega.exp().clamp_(max=r_bar or float('+inf'))
        rho = rho.reshape(*rho.shape, *trailing)

    n_steps, *shape = rew.shape
    G_t = rew.new_zeros((1 + n_steps, *shape))

    G_t[-1].copy_(bootstrap)  # bootstrap of \approx (r_{H+k+1})_{k\geq 0}
    for j in range(1, n_steps + 1):
        # G_t = \rho_t \delta_t + \gamma \rho_t G_{t+1} 1_{\neg d_{t+1}}
        # XXX G[-j] is G_{t+1} and G[-j-1] is G_t, and G_t[-j-1] is all zeros
        if omega is not None:
            G_t[-j-1].addcmul_(G_t[-j], rho[-j], value=gamma)
        else:
            G_t[-j-1].add_(G_t[-j], alpha=gamma)

        G_t[-j-1].masked_fill_(fin[-j], 0.)
        G_t[-j-1].add_(rew[-j])  # add the received reward r_{t+1}

    return G_t[:-1]


@torch.no_grad()
def pyt_multistep(
    rew,
    fin,
    val,
    *,
    gamma,
    n_lookahead=None,
    bootstrap=0.,
    omega=None,
    r_bar=None,
):
    r"""Compute the importance weighted present-value estimate:

        G_t = r_{t+1} + \gamma \rho_t G_{t+1} 1_{\neg d_{t+1}}
    """
    # v(s_t) ~ G_t = r_{t+1} + \gamma G_{t+1} 1_{\neg d_{t+1}}
    # r_{t+1}, s_{t+1} \sim p(r, s, \mid s_t, a_t), a_t \sim \pi(a \mid s_t)
    # d_{t+1} indicates if $s_{t+1}$ is terminal
    n_steps, *shape = rew.shape
    n_lookahead = n_lookahead or n_steps

    # eff[t] = (~fin[t]) * gamma = 1_{\neg d_{t+1}} \gamma, t=0..T-1
    eff = rew.new_full(fin.shape, gamma).masked_fill_(fin, 0.)
    if omega is not None:
        # \rho_t = \min\{ \bar{\rho}, \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
        eff.mul_(omega.exp().clamp_(max=r_bar or float('+inf')))

    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    eff = eff.reshape(*eff.shape, *trailing)

    # a double buffer for the intermediate calculations is autodiff-friendly
    # XXX for autodiff it is better to make `val` have n_steps+1 length
    #  with its last value being the bootstrap
    out = val.new_zeros((2, n_steps + 1, *shape))
    out[:, -1:].copy_(torch.as_tensor(bootstrap))
    out[:, :-1].copy_(val)

    j = 0  # index into double buffer that we read from
    for _ in range(n_lookahead):
        # out[1-j, t] = rew[t] + eff[t] * out[j, t+1], t=0..T-1
        # out[1-j, :-1] = torch.addcmul(rew, eff, out[j, 1:])
        torch.addcmul(rew, eff, out[j, 1:], out=out[1-j, :-1])

        # flip the buffer
        j = 1 - j

    # do not cut off incomplete returns
    return out[j, :-1]


@torch.no_grad()
def pyt_deltas(rew, fin, val, *, gamma, bootstrap=0., omega=None, r_bar=None):
    r"""Compute the importance weighted td-error estimates:

        \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
        # \delta^v_s = 0 for all s \geq t if d_t = \top
    """
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    bootstrap = torch.as_tensor(bootstrap)

    # a_hat[t] = val[t+1]
    a_hat = torch.empty_like(rew).copy_(bootstrap)
    a_hat[:-1].copy_(val[1:])

    # \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
    a_hat.masked_fill_(fin, 0.).mul_(gamma).add_(rew).sub_(val)
    if omega is None:
        return a_hat

    # \rho_t = \min\{ \bar{\rho}, \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
    rho = omega.exp().clamp_(max=r_bar or float('+inf'))
    return a_hat.mul_(rho.reshape(*rho.shape, *trailing))


@torch.no_grad()
def pyt_gae(rew, fin, val, *, gamma, C, bootstrap=0.):
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    n_steps, *shape = rew.shape
    bootstrap = torch.as_tensor(bootstrap)

    gae_t, delta = rew.new_zeros((1 + n_steps, *shape)), rew.new_zeros(shape)
    # rew[t], fin[t], val[t] is r_{t+1}, d_{t+1} and v(s_t)
    # t is -j, t+1 is -j-1 (j=1..T)
    for j in range(1, n_steps + 1):
        # \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
        delta.add_(bootstrap, alpha=gamma).masked_fill_(fin[-j], 0.)
        delta.add_(rew[-j]).sub_(val[-j])  # add r_{t+1} - v(s_t)

        # A_t = \delta_t + \lambda \gamma A_{t+1} 1_{\neg d_{t+1}}
        gae_t[-j-1].add_(gae_t[-j], alpha=C * gamma).masked_fill_(fin[-j], 0.)
        gae_t[-j-1].add_(delta)

        bootstrap = val[-j]  # v(s_t) is next iter's bootstrap
        delta.zero_()

    return gae_t[:-1]


@torch.no_grad()
def pyt_vtrace(rew, fin, val, *, gamma, bootstrap=0., omega=None, r_bar, c_bar):
    # add extra trailing unitary dims for broadcasting
    trailing = (1,) * max(rew.ndim - fin.ndim, 0)
    fin = fin.reshape(*fin.shape, *trailing)

    # raise NotImplementedError
    n_steps, *shape = rew.shape
    bootstrap = torch.as_tensor(bootstrap)

    # \rho_t = \min\{ \bar{\rho},  \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
    rho = omega.exp().clamp_(max=r_bar or float('+inf'))
    rho = rho.reshape(*rho.shape, *trailing)

    # c_t = \min\{ \bar{c},  \frac{\pi_t(a_t)}{\mu_t(a_t)} \}
    see = omega.exp().clamp_(max=c_bar or float('+inf'))
    see = see.reshape(*see.shape, *trailing)

    a_hat, delta = rew.new_zeros((1 + n_steps, *shape)), rew.new_zeros(shape)
    # rew[t], fin[t], val[t] is r_{t+1}, d_{t+1} and v(s_t)
    # t is -j, t+1 is -j-1 (j=1..T)
    for j in range(1, n_steps + 1):
        # \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
        delta.add_(bootstrap, alpha=gamma).masked_fill_(fin[-j], 0.)
        delta.add_(rew[-j]).sub_(val[-j])  # add r_{t+1} - v(s_t)

        # A_t = \rho_t \delta_t + \gamma \c_t A_{t+1} 1_{\neg d_{t+1}}
        a_hat[-j-1].addcmul_(a_hat[-j], see[-j], value=gamma)
        a_hat[-j-1].masked_fill_(fin[-j], 0.)
        a_hat[-j-1].addcmul_(delta, rho[-j])

        bootstrap = val[-j]  # v(s_t) is next iter's bootstrap
        delta.zero_()

    return a_hat[:-1] + val
