import pytest

import numpy
import torch

from collections import namedtuple

from rlplay.engine.utils.shared import aliased
from rlplay.algo.returns import pyt_gae, npy_gae
from rlplay.algo.returns import pyt_deltas, npy_deltas
from rlplay.algo.returns import pyt_returns, npy_returns
from rlplay.algo.returns import pyt_vtrace, npy_vtrace

RewardsData = namedtuple('RewardsData', [
    'rew', 'fin', 'val', 'bootstrap', 'omega'
])


def random_reward_data(T=120, B=10, dtype=float, M=None):
    rew = torch.randn(T, B, M or 1, dtype=dtype)
    fin = torch.randint(2, size=(T, B,), dtype=bool)
    val = torch.randn(T, B, M or 1, dtype=dtype)
    bootstrap = torch.randn(1, B, M or 1, dtype=dtype)

    if M is None:
        bootstrap = bootstrap.squeeze(-1)
        rew = rew.squeeze(-1)
        val = val.squeeze(-1)

    # log importance weight of the taken action (target vs. behavioral)
    omega = torch.randn(T, B, dtype=dtype)
    return aliased(RewardsData(rew, fin, val, bootstrap, omega))


def npy_manual_present_value(rew, fin, *, gamma, rho, bootstrap):
    # compute returns
    results, discounts = [], (~fin) * gamma

    trailing = (1,) * max(rew.ndim - discounts.ndim, 0)
    discounts = discounts.reshape(*discounts.shape, *trailing)
    rho = rho.reshape(*rho.shape, *trailing)

    # backward accumulation
    present = numpy.array(bootstrap, copy=True)
    for t in range(1, 1 + len(rew)):
        present = rew[-t] + rho[-t] * discounts[-t] * present
        results.append(present)

    results.reverse()

    return numpy.concatenate(results, axis=0)


def npy_manual_deltas(rew, fin, val, *, gamma, rho, bootstrap):
    # get td errors
    discounts = (~fin) * gamma

    trailing = (1,) * max(rew.ndim - discounts.ndim, 0)
    discounts = discounts.reshape(*discounts.shape, *trailing)
    rho = rho.reshape(*rho.shape, *trailing)

    vtp1 = numpy.concatenate([val[1:], bootstrap], axis=0)

    return rho * (rew + discounts * vtp1 - val)


@pytest.mark.parametrize('gamma', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
@pytest.mark.parametrize('M', [
    None, 5
])
def test_returns(gamma, M, T=120, B=10):
    data = random_reward_data(T, B, M=M)

    expected = npy_manual_present_value(
        data.npy.rew, data.npy.fin, gamma=gamma,
        rho=numpy.ones_like(data.npy.fin, float),
        bootstrap=data.npy.bootstrap)

    npy = npy_returns(data.npy.rew, data.npy.fin,
                      gamma=gamma, omega=None,
                      bootstrap=data.npy.bootstrap[0])
    assert numpy.allclose(npy, expected)

    pyt = pyt_returns(data.pyt.rew, data.pyt.fin,
                      gamma=gamma, omega=None,
                      bootstrap=data.pyt.bootstrap[0])
    assert numpy.allclose(pyt.numpy(), expected)

    # test importance weights
    for r_bar in [0.5, 1.0, 2.0, None]:
        rho = numpy.minimum(numpy.exp(data.npy.omega), r_bar or float('+inf'))

        expected = npy_manual_present_value(
            data.npy.rew, data.npy.fin, gamma=gamma,
            rho=rho, bootstrap=data.npy.bootstrap)

        omega = numpy.log(rho)

        npy = npy_returns(
            data.npy.rew, data.npy.fin, gamma=gamma, r_bar=None,
            omega=omega, bootstrap=data.npy.bootstrap[0])
        assert numpy.allclose(npy, expected)

        pyt = pyt_returns(
            data.pyt.rew, data.pyt.fin, gamma=gamma, r_bar=None,
            omega=torch.from_numpy(omega), bootstrap=data.pyt.bootstrap[0])

        assert numpy.allclose(pyt.numpy(), expected)


@pytest.mark.parametrize('gamma', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
@pytest.mark.parametrize('M', [
    None, 5
])
def test_deltas(gamma, M, T=120, B=10):
    data = random_reward_data(T, B, M=M)

    expected = npy_manual_deltas(
        data.npy.rew, data.npy.fin, data.npy.val, gamma=gamma,
        rho=numpy.ones_like(data.npy.fin, float),
        bootstrap=data.npy.bootstrap)

    npy = npy_deltas(data.npy.rew, data.npy.fin, data.npy.val,
                     gamma=gamma, bootstrap=data.npy.bootstrap[0])
    assert numpy.allclose(npy, expected)

    pyt = pyt_deltas(data.pyt.rew, data.pyt.fin, data.pyt.val,
                     gamma=gamma, bootstrap=data.pyt.bootstrap[0])

    assert numpy.allclose(pyt.numpy(), expected)

    # test importance weights
    for r_bar in [0.5, 1.0, 2.0, None]:
        rho = numpy.minimum(numpy.exp(data.npy.omega), r_bar or float('+inf'))

        expected = npy_manual_deltas(
            data.npy.rew, data.npy.fin, data.npy.val, gamma=gamma,
            rho=rho, bootstrap=data.npy.bootstrap)

        omega = numpy.log(rho)

        npy = npy_deltas(
            data.npy.rew, data.npy.fin, data.npy.val, gamma=gamma, r_bar=None,
            omega=omega, bootstrap=data.npy.bootstrap[0])
        assert numpy.allclose(npy, expected)

        pyt = pyt_deltas(
            data.pyt.rew, data.pyt.fin, data.pyt.val, gamma=gamma, r_bar=None,
            omega=torch.from_numpy(omega), bootstrap=data.pyt.bootstrap[0])

        assert numpy.allclose(pyt.numpy(), expected)


@pytest.mark.parametrize('gamma', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
@pytest.mark.parametrize('C', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
@pytest.mark.parametrize('M', [
    None, 5
])
def test_gae(gamma, M, C, T=120, B=10):
    data = random_reward_data(T, B, M=M)

    # GAE is essentially present value of one-step td-errors
    deltas = npy_manual_deltas(
        data.npy.rew, data.npy.fin, data.npy.val,
        rho=numpy.ones_like(data.npy.fin, float),
        gamma=gamma, bootstrap=data.npy.bootstrap)

    expected = npy_manual_present_value(
        deltas, data.npy.fin,
        rho=numpy.ones_like(data.npy.fin, float), gamma=gamma * C,
        bootstrap=numpy.zeros_like(data.npy.bootstrap))

    npy = npy_gae(data.npy.rew, data.npy.fin, data.npy.val,
                  gamma=gamma, C=C, bootstrap=data.npy.bootstrap[0])
    assert numpy.allclose(npy, expected)

    pyt = pyt_gae(data.pyt.rew, data.pyt.fin, data.pyt.val,
                  gamma=gamma, C=C, bootstrap=data.pyt.bootstrap[0])

    assert numpy.allclose(pyt.numpy(), expected)


@pytest.mark.parametrize('gamma', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
@pytest.mark.parametrize('r_bar', [
    0.5, 1.0, 2.0, None
])
@pytest.mark.parametrize('c_bar', [
    0.5, 1.0, 2.0, None
])
@pytest.mark.parametrize('M', [
    None, 5
])
def test_vtrace(gamma, M, r_bar, c_bar, T=120, B=10):
    # gamma, r_bar, c_bar, T, B = 0.5, 1.0, 1.0, 120, 10
    data = random_reward_data(T, B, M=M)

    # compute the vtrace estimates manually
    ratio = numpy.exp(data.npy.omega)
    rho = numpy.minimum(ratio, r_bar) if r_bar is not None else ratio
    see = numpy.minimum(ratio, c_bar) if c_bar is not None else ratio

    # v-trace is essentially GAE with C=1 and different importance weights
    #  shifted by the value estimates
    deltas = npy_manual_deltas(
        data.npy.rew, data.npy.fin, data.npy.val,
        rho=rho,
        gamma=gamma, bootstrap=data.npy.bootstrap)

    advantage = npy_manual_present_value(
        deltas, data.npy.fin,
        rho=see, gamma=gamma,
        bootstrap=numpy.zeros_like(data.npy.bootstrap))

    expected = advantage + data.npy.val

    npy = npy_vtrace(
        data.npy.rew, data.npy.fin, data.npy.val, omega=data.npy.omega,
        gamma=gamma, r_bar=r_bar, c_bar=c_bar, bootstrap=data.npy.bootstrap[0])

    assert numpy.allclose(npy, expected)

    pyt = pyt_vtrace(
        data.pyt.rew, data.pyt.fin, data.pyt.val, omega=data.pyt.omega,
        gamma=gamma, r_bar=r_bar, c_bar=c_bar, bootstrap=data.pyt.bootstrap[0])

    assert numpy.allclose(pyt.numpy(), expected)
