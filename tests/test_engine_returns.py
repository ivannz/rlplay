import pytest

import numpy
import torch

from collections import namedtuple

from rlplay.utils.schema.shared import aliased
from rlplay.engine.returns import pyt_gae, npy_gae
from rlplay.engine.returns import pyt_returns, npy_returns

RewardsData = namedtuple('RewardsData', [
    'rew', 'fin', 'val', 'bootstrap', 'omega'
])


def random_reward_data(T=120, B=10):
    rew, val = torch.randn(T, B), torch.randn(T, B)
    fin = torch.randint(2, size=(T, B,), dtype=bool)
    bootstrap = torch.randn(1, B)

    # log importance weight of the taken action (target vs. behavioral)
    omega = torch.randn(T, B)
    return aliased(RewardsData(rew, fin, val, bootstrap, omega))


@pytest.mark.parametrize('gamma', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
def test_returns(gamma, T=120, B=10):

    data = random_reward_data(T, B)
    npy = npy_returns(data.npy.rew, data.npy.fin,
                      gamma=gamma, bootstrap=data.npy.bootstrap[0])

    pyt = pyt_returns(data.pyt.rew, data.pyt.fin,
                      gamma=gamma, bootstrap=data.pyt.bootstrap[0])

    assert numpy.allclose(pyt.numpy(), npy)


@pytest.mark.parametrize('gamma', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
@pytest.mark.parametrize('C', [
    0.0, 0.5, 0.9, 0.999, 1.0
])
def test_gae(gamma, C, T=120, B=10):
    data = random_reward_data(T, B)

    npy = npy_gae(data.npy.rew, data.npy.fin, data.npy.val,
                  gamma=gamma, C=C, bootstrap=data.npy.bootstrap[0])

    pyt = pyt_gae(data.pyt.rew, data.pyt.fin, data.pyt.val,
                  gamma=gamma, C=C, bootstrap=data.pyt.bootstrap[0])

    assert numpy.allclose(pyt.numpy(), npy, rtol=1e-5, atol=1e-6)
