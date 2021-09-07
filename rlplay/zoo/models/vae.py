import torch
from math import log

from torch.distributions import Independent, Normal, Bernoulli
from torch.nn.functional import softplus
from torch.nn import Flatten

import torch.distributions as dist
from torch.distributions.utils import broadcast_all
from torch.autograd import Function


# My own stable implementation of differentiable Continuous Bernoulli distribution.
# See details in ~/Github/general-scribbles/coding/vq-VAE.ipynb
def cb_log_const_forward(z, atol=1e-3):
    # z \mapsto \log e^{-\lvert z \rvert} - 1}{-\lvert z \rvert}
    zabs = abs(z)
    output = zabs.neg().expm1_().div_(zabs).neg_().log_()
    # add \max\{0, z\}
    output = output.add_(z.relu())

    # 4th order Taylor around 0 (actually \pm O(z^6))
    z2 = z.mul(z)
    approx = z2.div_(-120).add_(1)
    approx.mul_(z).div_(12).add_(1)
    approx.mul_(z).div_(2)

    log_const = torch.where(zabs.ge(atol), output, approx)

    return log_const


def cb_mean_forward(z, atol=1e-2):
    # computes \mu(z) = -\frac{z + e^{-z} - 1}{z (e^{-z} - 1)} for z > 0
    zabs = abs(z)
    em1 = zabs.neg().expm1_()
    output = zabs.add(em1).div_(em1).div_(zabs).neg_()
    output = torch.where(z.signbit(), 1 - output, output)

    # 5th order Taylor approx around z=0
    z2 = z.mul(z)
    approx = z2.div(42).sub_(1)
    approx.mul_(z2).div_(60).add_(1)
    approx.mul_(z).div_(6).add_(1)
    approx.div_(2)

    m_z = torch.where(zabs.ge(atol), output, approx)

    return m_z


class cb_log_const(Function):
    @staticmethod
    def forward(ctx, z, atol=1e-2):
        ctx.atol = atol
        ctx.save_for_backward(z)
        return cb_log_const_forward(z, atol=atol)

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        return cb_mean_forward(z, atol=ctx.atol) * grad_output, None


def cb_grad_mean_forward(z, atol=5e-1):
    # computes \partial_z \mu(z) = \frac1{z^2} - \frac1{2 \cosh{z} - 2}
    term1 = z.mul(z).reciprocal_()
    term2 = z.cosh().sub_(1).mul_(2).reciprocal_()
    output = term1.sub_(term2)

    # 6th order Taylor approx around z=0
    z2 = z.mul(z)
    approx = z2.mul(7).div_(200).sub_(1)
    approx.mul_(z2).mul_(5).div_(126).add_(1)
    approx.mul_(z2).div_(20).sub_(1)
    approx.div_(12).neg_()

    m_z = torch.where(abs(z).ge(atol), output, approx)

    return m_z


class cb_mean(Function):
    @staticmethod
    def forward(ctx, z, atol=5e-1):
        ctx.atol = atol
        ctx.save_for_backward(z)
        return cb_mean_forward(z, atol=atol)

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        return cb_grad_mean_forward(z, atol=ctx.atol) * grad_output, None


def cb_rsmaple_forward(z, u, atol=5e-1):
    # Expects u \sim U[0, 1) and z \in \mathbb{R}
    # exact inverse transfrom
    logits = u.logit()
    output = softplus(logits.add(z)).sub_(softplus(logits)).div_(z)

    # 3-rd order Taylor approx
    #  approx = u + u * (1 - u) / 2 * z * (1  + (1 - u - u) / 3 * z)
    uu = u * (1 - u)
    approx = uu.mul(6).sub_(1).div_(-4)
    approx.mul_(z).add_(1).sub_(u).sub_(u).div_(3)
    approx.mul_(z).add_(1).mul_(uu).div_(2)
    approx.mul_(z).add_(u)
    approx.clamp_(0, 1)

    output = torch.where(abs(z).ge(atol), output, approx)
    return output


def cb_grad_rsmaple_forward(z, u, x_z, atol=5e-1):
    # exact
    output = u.logit().add_(z).sigmoid_().sub_(x_z).div_(z)

    # 2nd order Taylor
    uu = u * (1 - u)
    approx = uu.mul(6).sub_(1).div_(-8).mul_(3)
    approx.mul_(z).add_(1).sub_(u).sub_(u).div_(3).mul_(2)
    approx.mul_(z).add_(1).mul_(uu).div_(2)

    output = torch.where(abs(z).ge(atol), output, approx)
    return output


class cb_rsample(Function):
    @staticmethod
    def forward(ctx, z, u, atol=5e-1):
        ctx.atol = atol
        x_zu = cb_rsmaple_forward(z, u, atol=atol)
        ctx.save_for_backward(z, u, x_zu)
        return x_zu

    @staticmethod
    def backward(ctx, grad_output):
        grad = cb_grad_rsmaple_forward(*ctx.saved_tensors, atol=ctx.atol)
        return grad * grad_output, None, None


class ContinuousBernoulli(Bernoulli):
    """Numerically stable implementation of CB from

        [Loaiza-Ganem et al. (2019)](https://papers.nips.cc/paper/2019/hash/f82798ec8909d23e55679ee26bb26437-Abstract.html)

    Uses applies logit reparameterization and employs analytic approximations
    for tiny or large fp (see funcs above).
    """
    has_enumerate_support = False
    has_rsample = True

    @property
    def mean(self):
        return cb_mean.apply(self.logits)

    @property
    def variance(self):
        raise NotImplementedError

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        rand = torch.rand(shape, dtype=self.logits.dtype, device=self.logits.device)
        return cb_rsample.apply(self.logits, rand)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return logits * value - cb_log_const.apply(logits)

    def entropy(self):
        return cb_log_const.apply(self.logits) - self.logits * self.mean

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def _log_normalizer(self, x):
        return cb_log_const.apply(self.logits)


def vbayes(
    enc,
    dec,
    X,
    Y=None,
    *,
    prior=None,
    beta=1.,
    n_draws=1,
    iwae=False,
):
    r"""Compute the SGVB or the IWAE objective.

    enc is the approximate posterior q(z \mid x)
    dec is the approximate model p(y \mid z)
    prior is the custom prior \pi(z \mid x) (defaults to `enc.prior`)

    See the supplementary material of

        [Bachman and Precup (2015)](http://proceedings.mlr.press/v37/bachman15.html)

    for some brief but clear discussion of what turns out to be the idea below,
    a variational trasncoder if X \neq Y.
    """
    pi = enc.prior if prior is None else prior
    assert callable(pi)

    Y = X if Y is None else Y  # auto-encode is Y is not X

    # get posterior approx q(z \mid x). `X=Y` is `*batch x *dec.event_shape`
    q = enc(X)  # q.batch_shape is `batch`

    # get pi(z) = \pi(z \mid x) -- conditional prior
    pi = pi(X)

    # `Z` is `n_draws x *q.batch_shape x *q.event_shape`
    Z = q.rsample([n_draws])  # XXX diff-able sampling with (implicit) rep-trick!

    # get the model p(x\mid z)
    p = dec(Z)

    # `log_p` has shape `n_draws x *q.batch_shape x *q.event_shape`
    log_p = p.log_prob(Y)  # XXX may consume a lot of mem!

    ll = log_p.mean()  # dim=(0, 1)
    if iwae and n_draws > 1:
        # (iwae)_k = E_{x y} E_{S~q^k(z|x)} log E_{z~S} p(y|z) pi(z) / q(z|x)
        #  * like (sgvb)_k but E_z and log are interchanged
        #  * beta-anneal the pi(z) / q(z|x) ratio
        log_iw = log_p + (pi.log_prob(Z) - q.log_prob(Z)) * beta
        loss = log(n_draws) - torch.logsumexp(log_iw, dim=0).mean()  # dim=0

    else:
        # (sgvb)_k = E_{x y} E_{S~q^k(z|x)} E_{z~S} log p(y|z) pi(z) / q(z|x)
        kl_q_pi = dist.kl_divergence(q, pi)
        loss = beta * kl_q_pi.mean() - ll  # dim=0

    return loss  # neg elbo/iwae


def batchify(fn, input, *, end):
    # flatten the composite batch dims, apply `fn`, then restore original dims
    end = input.ndim - end if end < 0 else end

    # b_0 x ... x b_{m} x ... -->> [b_0 x ... x b_{m}] x ...
    out = fn(input.flatten(0, end))
    return out.view(input.shape[:end + 1] + out.shape[1:])


class Reshape(torch.nn.Module):
    """The reshape layer -- sort of inverse to `torch.nn.Flatten`.
    """
    def __init__(self, *shape, start_dim=1, end_dim=-1):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        shape = torch.Size(shape)

        super().__init__()
        self.shape, self.start_dim, self.end_dim = shape, start_dim, end_dim

    def extra_repr(self) -> str:
        return '{}, start_dim={}, end_dim={}'.format(
            tuple(self.shape), self.start_dim, self.end_dim
        )

    def forward(self, input):
        a, z = self.start_dim, self.end_dim
        a = input.ndim + a if a < 0 else a
        z = input.ndim + z if z < 0 else z

        # `.view` raises on non-contiguous tensors
        return input.view(
            *input.shape[:a],
            *self.shape,
            *input.shape[z+1:],
        )


class AsIndependentDistribution(torch.nn.Module):
    """A factorized distribution parameterized by a deep neural network."""
    def __init__(
        self,
        module,              # dims In: (B*)CS* -->> dims Out: \1FS*  # re syntax
        n_dim_in=1,          # determines the number of trailing dims designated as input features
        n_dim_out=1,         # the trailing dims in the output allotted to a single event (event_size)
        transforms=None,     # transformations of the output for each parameter
        batch_first=None,    # bool, the order of batch and sequence dims for recurrent nets
                             # XXX might be very awkward to implement...
        validate_args=None,  # ensure the observations belong to the support
    ):
        assert n_dim_out >= 1
        assert batch_first is None
        transforms = (lambda x: x,) if transforms is None else transforms
        assert transforms and all(callable(f) for f in transforms)

        super().__init__()

        self.n_dim_in, self.n_dim_out = n_dim_in, n_dim_out
        self.transforms, self.batch_first = transforms, batch_first
        self.validate_args = validate_args
        self.wrapped = module

    def forward(self, input, *rest):
        # If the input a tuple, then interpret its as a ready loc-scale pair.
        # otherwise pass through the wrapped network and split the dims
        if isinstance(input, torch.Tensor) and self.n_dim_in is not None:
            # we need this to assign proper batch dims to the returned
            # distribution object, and to make sure not to confuse events dims.
            assert self.n_dim_in >= 1

            # flatten the batch dims, keeping feature dims intact, then undo
            at = input.ndim - self.n_dim_in
            rest = [x.view(-1, *x.shape[at:]) for x in rest]

            out = self.wrapped(input.view(-1, *input.shape[at:]), *rest)
            out = out.view(*input.shape[:at], *out.shape[-self.n_dim_out:])

            # the specified dim of the output must be divisible by `n_par`!
            input = torch.chunk(out, len(self.transforms), dim=-self.n_dim_out)

            # apply the transformations, e.g. a +ve monotonic xform to scale
            input = [f(x) for f, x in zip(self.transforms, input)]

        else:
            # warnings.warn('Bypassing chunking logic', SimpleWarning)
            pass

        # assume the shapes are proper
        return self.as_distribution(*input)

    def prior(self, input=None):
        # the default standard gaussian prior (can be overridden)
        raise NotImplementedError

    def as_distribution(self, *args, **kwargs):
        raise NotImplementedError


class AsIndependentGaussian(AsIndependentDistribution):
    """The Factorized Gaussian distribution parameterized by a deep neural
    network.
    """
    def __init__(
        self,
        module,              # dims In: (B*)CS* -->> dims Out: \1FS*  # re syntax
        n_dim_in=1,          # determines the number of trailing dims designated as input features
        n_dim_out=1,         # the number of trailing dims allotted to a single random draw (event_size)
        fn_loc=lambda x: x,  # transformation of the location output
        fn_scale=softplus,   # The transformation to apply to the output desiganted as scale
                             # XXX can use fn_scale = torch.ones_like
        prior=None,          # The prior associated with this Gaussian, standard if None
    ):
        super().__init__(
            module,
            n_dim_in,
            n_dim_out,
            transforms=[fn_loc, fn_scale],
            batch_first=None,
            validate_args=False,
        )

        # construct the standard Gaussian prior as default
        if not callable(prior):
            # zero and one for the prior, kept as a buffer for sync
            self.register_buffer(
                'nilone',
                torch.tensor([0., 1.]).view(2, *(1,) * self.n_dim_out)
            )
            prior = None

        self.prior_ = prior

    def prior(self, input=None):
        if self.prior_ is None:
            return self.as_distribution(*self.nilone)

        return self.prior_(input)

    def as_distribution(self, loc, scale):
        # cannot auto-infer the event dim, rely on the `n_dim_out` parameter
        normal = Normal(loc, scale, self.validate_args)
        return Independent(normal, self.n_dim_out)


class AsIndependentBernoulli(AsIndependentDistribution):
    """The Factorized Bernoulli distribution with logits parameterized by
    a deep neural network.
    """
    def __init__(
        self,
        module,              # dims In: (B*)CS* -->> dims Out: \1S*  # re syntax
        n_dim_in=1,          # determines the number of trailing dims designated as input features
        n_dim_out=1,         # the number of trailing dims allotted to a single random draw (event_size)
        validate_args=None,
    ):
        super().__init__(
            module,
            n_dim_in,
            n_dim_out,
            transforms=None,
            batch_first=None,
            validate_args=validate_args,
        )

    def as_distribution(self, logits):
        # cannot auto-infer the event dim, rely on the `n_dim_out` parameter
        bernoulli = Bernoulli(logits=logits, validate_args=self.validate_args)
        return Independent(bernoulli, self.n_dim_out)


class AsIndependentContinuousBernoulli(AsIndependentBernoulli):
    """The Factorized Continuous Bernoulli distribution with logits
    parameterized by a deep neural network.
    """
    def __init__(
        self,
        module,              # dims In: (B*)CS* -->> dims Out: \1S*  # re syntax
        n_dim_in=1,          # determines the number of trailing dims designated as input features
        n_dim_out=1,         # the number of trailing dims allotted to a single random draw (event_size)
    ):
        super().__init__(
            module,
            n_dim_in,
            n_dim_out,
            validate_args=False,
        )

    def as_distribution(self, logits):
        cb = ContinuousBernoulli(logits=logits, validate_args=self.validate_args)
        return Independent(cb, self.n_dim_out)
