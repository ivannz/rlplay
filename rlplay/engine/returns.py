import numpy
import torch


# returns, baselined or not, or advantage estimates are not diff-able in PG
def np_compute_returns(rew, fin, *, gamma, bootstrap=0.):
    r"""Compute the on-policy returns (the present value of the future rewards).

        G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ...
            = r_{t+1} + \gamma G_{t+1}
    """
    n_steps, shape = len(rew), rew[-1].shape
    G_t = numpy.zeros((1 + n_steps, *shape), dtype=rew[-1].dtype)

    # rew[t], fin[t] is r_{t+1} and d_{t+1}
    G_t[-1] = bootstrap
    for j in range(1, n_steps + 1):
        # get G_t = r_{t+1} + \gamma G_{t+1} 1_{\neg d_{t+1}}
        # XXX G_t[-j-1] is all zeros
        numpy.multiply(G_t[-j], gamma, where=~fin[-j], out=G_t[-j-1])
        G_t[-j-1] += rew[-j]

    return G_t[:-1]


@torch.no_grad()
def tr_compute_returns(rew, fin, *, gamma, bootstrap=0.):
    n_steps, *shape = rew.shape

    # G_t = v(s_t) = r_{t+1} + \gamma G_{t+1} 1_{\neg T_{t+1}}
    # r_{t+1}, s_{t+1} \sim p(r, s, \mid s_t, a_t), a_t \sim \pi(a \mid s_t)
    # T_{t+1} indicates if $s_{t+1}$ is terminal
    G_t = rew.new_zeros((1 + n_steps, *shape))

    G_t[-1].copy_(bootstrap)  # bootstrap of \approx (r_{H+k+1})_{k\geq 0}
    for j in range(1, n_steps + 1):
        # compute $G_t = r_{t+1} + \gamma G_{t+1}$ for t = L - j
        # NB G[-j] is G_{t+1} and G[-j-1] is G_t

        # get 1_{\neg T_{t+1}} v(s_{t+1}) \gamma + r_{t+1}
        G_t[-j-1].copy_(G_t[-j]).mul_(gamma)  # get \hat{G}_{t+1}(\tau) \gamma
        G_t[-j-1].masked_fill_(fin[-j], 0.)  # reset[-j] means s_{t+1} is terminal
        G_t[-j-1].add_(rew[-j])  # add the received reward r_{t+1}

    return G_t[:-1]


def np_compute_gae(rew, fin, val, *, gamma, C, bootstrap=0.):
    r"""Compute the Generalized Advantage Estimator (C is `lambda`).

        \delta^v_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)

        A_t = \delta^v_t + (\gamma \lambda) \delta^v_{t+1}
              + (\gamma \lambda)^2 \delta^v_{t+2} + ...
            = \delta^v_t + \gamma \lambda A_{t+1} 1_{\neg d_{t+1}}
    """
    n_steps, shape = len(rew), rew[-1].shape

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


def np_compute_vtrace(rew, fin, val, omega, *, gamma, bootstrap, r_bar, c_bar):
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
    """
    raise NotImplementedError

    # V-trace uses importance weights to correct for off-policy PG
    n_steps, shape = len(rew), rew[-1].shape

    v_hat = numpy.zeros((1 + n_steps, *shape), dtype=rew[-1].dtype)
    delta = numpy.zeros(shape, dtype=rew[-1].dtype)
    # rew[t], fin[t], val[t] is r_{t+1}, d_{t+1} and v(s_t)
    # t is -j, t+1 is -j-1 (j=1..T)
    for j in range(1, n_steps + 1):
        # \delta_t = r_{t+1} + \gamma v(s_{t+1}) 1_{\neg d_{t+1}} - v(s_t)
        numpy.multiply(bootstrap, gamma, out=delta, where=~fin[-j])
        delta += rew[-j] - val[-j]

        # A_t = \delta_t + \lambda \gamma A_{t+1} 1_{\neg d_{t+1}}
        numpy.multiply(v_hat[-j], C * gamma, out=v_hat[-j-1], where=~fin[-j])
        v_hat[-j-1] += delta

        # reset delta for the next conditional multiply
        bootstrap = val[-j]  # v(s_t) is next iter's bootstrap
        delta[:] = 0

    return v_hat[:-1]
