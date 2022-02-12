import torch
from math import log

from torch.distributions import Independent, Normal, Bernoulli
from torch.nn.functional import softplus
from torch.nn import Flatten

import torch.distributions as dist
from torch.distributions.utils import broadcast_all
from torch.autograd import Function

# for vq-vae
from typing import Tuple
import torch.nn.functional as F


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


class LegacyVQEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float = 0.,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm=None,
            padding_idx=None,
            scale_grad_by_freq=False,
            sparse=False,
        )

        self.alpha, self.eps = alpha, eps

        # if `alpha` is zero then `.weight` is updated by other means
        self.register_buffer('ema_vecs', None)
        self.register_buffer('ema_size', None)
        if self.alpha <= 0:
            return

        # allocate buffer for tracking k-means cluster cenrtoid updates
        self.register_buffer(
            'ema_vecs', torch.zeros_like(torch.zeros_like(self.weight)),
        )
        self.register_buffer(
            'ema_size', torch.zeros_like(self.ema_vecs[:, 0]),
        )

        # demote `.weight` to a buffer and disable backprop for it
        # XXX can promote buffer to parameter, but not back, so we `delattr`.
        #  Also non-inplace `.detach` creates a copy not reflected in referrers.
        weight = self.weight
        delattr(self, 'weight')
        self.register_buffer('weight', weight.detach_())

    @torch.no_grad()
    def _update(self, input: torch.Tensor, indices: torch.LongTensor):
        """Update the embedding vectors by Exponential Moving Average.
        """
        # `input` is `B x F x *spatial`, `indices` are `B x *spatial`
        affinity = F.one_hot(indices, self.num_embeddings).to(input)
        # XXX 'affinity' is `B x *spatial x C`

        # sum the F-dim input vectors into bins by affinity
        # XXX can also use torch.bincount(ix.flatten())
        upd_vecs = torch.einsum('bf..., b...c -> cf', input, affinity)
        upd_size = torch.einsum('...c -> c', affinity)

        # track cluster size and unnormalized vecs with EMA
        self.ema_vecs.lerp_(upd_vecs, self.alpha)
        self.ema_size.lerp_(upd_size, self.alpha)

        # Apply \epsilon-Laplace correction
        n = self.ema_size.sum()
        coef = n / (n + self.num_embeddings * self.eps)
        size = coef * (self.ema_size + self.eps).unsqueeze(1)
        self.weight.data.copy_(self.ema_vecs / size)

    @torch.no_grad()
    def lookup(self, input: torch.Tensor):
        """Lookup the index of the nearest embedding."""
        # batch x n_dim x *spatial
        _, n_dim, *spatial = input.shape

        # n_embedings x n_dim
        x = self.weight
        sqr = (x * x).sum(dim=1).reshape(1, -1, *[1] * len(spatial))
        cov = torch.einsum('bd..., nd -> bn...', input, x)
        inx = torch.argmin(sqr - 2 * cov, dim=1)
        # no need to compute the norm fully since we do not backprop
        #  through the input when doing clustering here.

        # compute the perplexity
        # x -- input, y -- target: get `- \sum_n y_n (\log y_n - x_n)`
        counts = inx.flatten().bincount(minlength=self.num_embeddings)
        entropy = -F.kl_div(x.new_zeros(()), counts / inx.numel(),
                            log_target=False, reduction='sum')

        return inx, entropy.exp()

    def fetch(seld, indices: torch.LongTensor, at: int = 1):
        """fetch embeddings and put their dim at position `at`"""
        vectors = super().forward(indices)  # call Embedding.forward

        # indices.shape is batch x *spatial
        dims = list(range(indices.ndim))
        at = (vectors.ndim + at) if at < 0 else at
        # vectors.permute(0, input.ndim-1, *range(1, input.ndim-1))
        return vectors.permute(*dims[:at], indices.ndim, *dims[at:])

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor]:
        """vq-VAE clustering with straight-through estimator and commitment
        losses.

        Details
        -------
        Implements

            [van den Oord et al. (2017)](https://arxiv.org/abs/1711.00937).

        See further details in the class docstring.
        """
        # lookup the index of the nearest embedding and fetch it
        indices, perplexity = self.lookup(input)
        vectors = self.fetch(indices, at=1)

        # commitment loss terms (identical in value, but not in backprop graph)
        emb_loss = F.mse_loss(vectors, input.detach(), reduction='sum')
        enc_loss = F.mse_loss(input, vectors.detach(), reduction='sum')

        # build the straight-through grad estimator
        output = input + (vectors - input).detach()

        # update the weights only if we are in training mode
        if self.training and self.alpha > 0:
            self._update(input, indices)

        return output, emb_loss, enc_loss, float(perplexity)


class VQEmbedding(torch.nn.Embedding):
    r"""Vector-quantized mebedding layer.

    Note
    ----
    My own implementation taken from
        ~/Github/general-scribbles/coding/vq-VAE.ipynb

    Details
    -------
    The key idea of the [vq-VAE](https://arxiv.org/abs/1711.00937) is how
    to train the nearest-neighbour-based quantization embeddings and how
    the gradients are to be backpropped through them:

    $$
    \operatorname{vq}(z; e)
        = \sum_k e_k 1_{R_k}(z)
        \,,
        \partial_z \operatorname{vq}(z; e) = \operatorname{id}
        \,. $$

    This corresponds to a degenerate conditional categorical rv
    $k^\ast_z$ with distribution $
        p(k^\ast_z = j\mid z)
            = 1_{R_j}(z)
    $ where

    $$
    R_j = \bigl\{
        z\colon
            \|z - e_j\|_2 < \min_{k\neq j} \|z - e_k\|_2
        \bigr\}
    \,, $$

    are the cluster affinity regions w.r.t. $\|\cdot \|_2$ norm. Note that we
    can compute

    $$
    k^\ast_z
        := \arg \min_k \frac12 \bigl\| z - e_k \bigr\|_2^2
        = \arg \min_k
            \frac12 \| e_k \|_2^2
            - \langle z, e_k \rangle
        \,. $$

    The authors propose STE for grads and mutual consistency losses for
    the embeddings:
    * $\| \operatorname{sg}(z) - e_{k^\ast_z} \|_2^2$ -- forces the embeddings
    to match the latent cluster's centroid (recall the $k$-means algo)
      * **NB** in the paper they use just $e$, but in the latest code they use
      the selected embeddings
      * maybe we should compute the cluster sizes and update to the proper
      centroid $
          e_j = \frac1{
              \lvert {i: k^\ast_{z_i} = j} \rvert
          } \sum_{i: k^\ast_{z_i} = j} z_i
      $.
    * $\| z - \operatorname{sg}(e_{k^\ast_z}) \|_2^2$ -- forces the encoder
    to produce the latents, which are consistent with the cluster they are
    assigned to.
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float = 0.,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            max_norm=None,
            padding_idx=None,
            scale_grad_by_freq=False,
            sparse=False,
        )

        self.alpha, self.eps = alpha, eps

        # if `alpha` is zero then `.weight` is updated by other means
        self.register_buffer('ema_vecs', None)
        self.register_buffer('ema_size', None)
        if self.alpha <= 0:
            return

        # demote `.weight` to a buffer and disable backprop for it
        # XXX can promote buffer to parameter, but not back, so we `delattr`.
        #  Also non-inplace `.detach` creates a copy not reflected in referrers.
        weight = self.weight
        delattr(self, 'weight')
        self.register_buffer('weight', weight.detach_())

        # allocate buffer for tracking k-means cluster centroid updates
        self.register_buffer(
            'ema_vecs', self.weight.clone(),
        )
        self.register_buffer(
            'ema_size', torch.zeros_like(self.ema_vecs[:, 0]),
        )

    @torch.no_grad()
    def _update(
        self,
        input: torch.Tensor,
        indices: torch.LongTensor,
    ) -> None:
        """Update the embedding vectors by Exponential Moving Average.
        """

        # `input` is `... x F`, `indices` are `...`
        affinity = F.one_hot(indices, self.num_embeddings).to(input)
        # XXX 'affinity' is `... x C`

        # sum the F-dim input vectors into bins by affinity
        #  S_j = \sum_i 1_{k_i = j} x_i
        #  n_j = \lvert i: k_i=j \rvert
        upd_vecs = torch.einsum('...f, ...k -> kf', input, affinity)
        upd_size = torch.einsum('...k -> k', affinity)

        # track cluster size and unnormalized vecs with EMA
        self.ema_vecs.lerp_(upd_vecs, self.alpha)
        self.ema_size.lerp_(upd_size, self.alpha)

        # Apply \epsilon-Laplace correction
        n = self.ema_size.sum()
        coef = n / (n + self.num_embeddings * self.eps)
        size = coef * (self.ema_size + self.eps).unsqueeze(1)
        self.weight.data.copy_(self.ema_vecs / size)

    @torch.no_grad()
    def lookup(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Lookup the index of the nearest embedding."""
        emb = self.weight
        # k(z) = \arg \min_k \|E_k - z\|^2
        #      = \arg \min_k \|E_k\|^2 - 2 E_k^\top z + \|z\|^2
        # XXX no need to compute the norm fully since we do not
        #  backprop through the input when clustering.

        sqr = (emb * emb).sum(dim=1)
        cov = torch.einsum('...j, kj -> ...k', input, emb)
        return torch.argmin(sqr - 2 * cov, dim=-1)

    def fetch(
        self,
        indices: torch.LongTensor,
        at: int = -1,
    ) -> torch.Tensor:
        """fetch embeddings and put their dim at position `at`"""
        vectors = super().forward(indices)  # call Embedding.forward

        # indices.shape is batch x *spatial
        dims = list(range(indices.ndim))
        at = (vectors.ndim + at) if at < 0 else at
        # vectors.permute(0, input.ndim-1, *range(1, input.ndim-1))
        return vectors.permute(*dims[:at], indices.ndim, *dims[at:])

    def forward(
        self,
        input: torch.Tensor,
        reduction: str = 'sum',
    ) -> Tuple[torch.Tensor]:
        """vq-VAE clustering with straight-through estimator and commitment
        losses.

        Details
        -------
        Implements

            [van den Oord et al. (2017)](https://arxiv.org/abs/1711.00937).

        See further details in the class docstring.
        """
        # lookup the index of the nearest embedding and fetch it
        indices = self.lookup(input)
        vectors = self.fetch(indices)

        # commitment and embedding losses, p. 4 eq. 3.
        # loss = - \log p(x \mid q(x))
        #      + \|[z(x)] - q(x)\|^2     % embedding loss (dictionary update)
        #      + \|z(x) - [q(x)]\|^2   % encoder's commitment loss
        # where z(x) is output of the encoder network
        #       q(x) = e_{k(x)}, for k(x) = \arg\min_k \|z(x) - e_k\|^2
        # XXX p.4 `To make sure the encoder commits to an embedding and
        #          its output does not grow, since the volume of the embedding
        #          space is dimensionless`
        # XXX the embeddings receive no gradients from the reconstruction loss
        embedding = F.mse_loss(vectors, input.detach(), reduction=reduction)
        commitment = F.mse_loss(input, vectors.detach(), reduction=reduction)

        # the straight-through grad estimator: copy grad from q(x) to z(x)
        output = input + (vectors - input).detach()

        # update the weights only if we are in training mode
        if self.training and self.alpha > 0:
            self._update(input, indices)
            # XXX `embedding` loss is non-diffable if we use ewm updates

        return output, indices, embedding, commitment
