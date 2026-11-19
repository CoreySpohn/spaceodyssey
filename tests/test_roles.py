"""Role weights from one refit cloud: reweighting, charts, hard edges, calibration and
members by ratio.

Every toy here is 1-D: the state is one number `x` per draw (one planet, no gases),
the claim is `x > 1`, and the role priors are Gaussian.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax.scipy.special import ndtr
from jax.scipy.stats import norm
from jax.test_util import check_grads
from numpy.testing import assert_allclose

from spaceodyssey._standin.diagnostics import mc_se
from spaceodyssey.belief import q, role_log_weights, weights
from spaceodyssey.claim import AbstractConjunct, Claim
from spaceodyssey.information import diagnose
from spaceodyssey.roles import AbstractRolePrior, Roles
from spaceodyssey.types import Atoms, Pool, WeightedCloud

STAR = ()
PLANET = jnp.asarray(0)
N_DRAWS = 20_000
PROPOSAL_SD = 1.5  # g = N(0, 1.5^2); the refit target is p0 = N(0, 1)
Q_B = 1.0 - 0.6914624612740131  # 1 - Phi(0.5) = 0.3085 under role B = N(0.5, 1)


class Above(AbstractConjunct):
    """`x > threshold` for the one planet of each draw."""

    threshold: float = 1.0

    def prob(self, atoms, star):
        return (atoms.cloud.samples["x"] > self.threshold).astype(float)


class Gaussian(AbstractRolePrior):
    """N(mean, sd^2) on x, or its density on y = exp(x) when the measure is "y".

    On "y" the log Jacobian -x is added unless `jacobian` is False (the deliberate
    one-sided omission). Outside `x < upper` the density is zero.
    """

    mean: jax.Array = eqx.field(converter=jnp.asarray)
    sd: jax.Array = eqx.field(converter=jnp.asarray, default=1.0)
    upper: jax.Array = eqx.field(converter=jnp.asarray, default=jnp.inf)
    jacobian: bool = eqx.field(static=True, default=True)

    def log_prob(self, atoms, star):
        x = atoms.cloud.samples["x"][:, 0]
        log_p = norm.logpdf(x, self.mean, self.sd)
        if self.measure == "y" and self.jacobian:
            log_p = log_p - x
        return jnp.where(x < self.upper, log_p, -jnp.inf)

    def sample(self, key, star):
        top = (self.upper - self.mean) / self.sd
        x = self.mean + self.sd * jax.random.truncated_normal(key, -jnp.inf, top, (1,))
        return _atoms(x, jnp.zeros(1), jnp.zeros(1), self.measure)


class LookAlikeRatio(AbstractRolePrior):
    """Multiplies the weight of the look-alike draws (x <= 1) by `w`.

    A ratio against the refit target, not a normalized density, so it has no sampler:
    it can judge as a verdict member but cannot draw a truth.
    """

    w: jax.Array = eqx.field(converter=jnp.asarray)

    def log_prob(self, atoms, star):
        x = atoms.cloud.samples["x"][:, 0]
        return jnp.where(x > 1.0, 0.0, jnp.log(self.w))

    def sample(self, key, star):
        raise NotImplementedError("a likelihood-ratio member has no sampler")


def _atoms(x, log_weight, log_ref, measure):
    """Atoms of a 1-D cloud: one planet per draw, an empty gas block, no rows."""
    n = x.shape[0]
    cloud = WeightedCloud({"x": x[:, None]}, log_weight, log_ref, measure)
    empty = jnp.zeros((n, 1, 0))
    return Atoms(
        cloud,
        empty,
        empty,
        jnp.zeros(0, bool),
        jnp.ones((n, 1), bool),
    )


def _empty_pool(n_roles):
    """A pool with no expected planets."""
    marks = _atoms(jnp.zeros(1), jnp.zeros(1), jnp.zeros(1), "x")
    flat = jnp.zeros((n_roles, 1))
    no_mass = jnp.full(n_roles, -jnp.inf)
    return Pool(marks, flat, flat, no_mass)


def _cloud(key, n, measure="x"):
    """Draws from g = N(0, 1.5^2), weighted to p0 = N(0, 1) on `measure`.

    On "y" the reference density carries the log Jacobian -x.
    """
    x = PROPOSAL_SD * jax.random.normal(key, (n,))
    log_p0 = norm.logpdf(x)
    log_weight = log_p0 - norm.logpdf(x, 0.0, PROPOSAL_SD)
    log_ref = log_p0 - x if measure == "y" else log_p0
    return _atoms(x, log_weight, log_ref, measure)


def _roles(measure="x", b_mean=0.5, upper=jnp.inf, jacobian=True):
    """Forecast A = N(0, 1); one verdict member B = N(b_mean, 1), its leaves stacked."""
    stack = lambda v: jnp.full(1, v)  # noqa: E731
    member = Gaussian(measure, stack(b_mean), stack(1.0), stack(upper), jacobian)
    return Roles(Gaussian(measure, 0.0), member)


def _belief(atoms, roles):
    n_draws = atoms.cloud.log_weight.shape[0]
    return role_log_weights(atoms, roles, jnp.zeros((2, n_draws)), STAR, _empty_pool(2))


CLAIM = Claim((Above(),))


@pytest.fixture(scope="module")
def clouds():
    """The same draws (key 0) as a cloud on x and as a cloud on y = exp(x)."""
    key = jax.random.PRNGKey(0)
    return {m: _cloud(key, N_DRAWS, m) for m in ("x", "y")}


def test_reweighting_matches_the_truth(clouds):
    """Under role B = N(0.5, 1) the estimate of P(x > 1) is 1 - Phi(0.5) = 0.3085 and
    the weighted mean of x is 0.5, each within 3 standard errors. Role A = p0 = N(0, 1)
    gives 1 - Phi(1) = 0.1587. The refit weight log p0 - log g is not constant, so it
    cannot be dropped silently. 20,000 draws: se is about 0.004.

    Fails if the refit weight `log_weight` is dropped or enters with the wrong sign,
    or if `log_ref_density` (log p0) enters with the wrong sign: A and p0 are not
    constant on the support, so each error moves both roles' estimates.
    """
    b = _belief(clouds["x"], _roles())
    d = diagnose(b, CLAIM, STAR, PLANET, jnp.zeros(2))
    truth = jnp.array([1.0 - ndtr(1.0), Q_B])
    assert jnp.all(jnp.abs(d.q - truth) < 3.0 * d.mc_se), (d.q, truth, d.mc_se)

    x = clouds["x"].cloud.samples["x"][:, 0]
    log_w_b = jnp.log(weights(b)[1])
    mean_b = jnp.exp(log_w_b) @ x
    se_mean = mc_se(log_w_b, x)
    assert jnp.abs(mean_b - 0.5) < 3.0 * se_mean, (mean_b, se_mean)


def test_two_charts_and_a_one_sided_jacobian(clouds):
    """On y = exp(x) the reference density and the prior both carry -x, so the weights
    and q are those of the x chart. Dropping the Jacobian from the prior only tilts
    B = N(0.5, 1) by exp(x) to N(1.5, 1): the estimate becomes 1 - Phi(-0.5) = 0.6915,
    far from the truth 0.3085. A prior on another measure than the cloud's raises.

    Fails if the reference density is not subtracted (the charts then disagree), if
    the test could not see a one-sided Jacobian, or if a measure mismatch is accepted.
    """
    on_x = q(_belief(clouds["x"], _roles("x")), CLAIM, STAR, PLANET)
    on_y = q(_belief(clouds["y"], _roles("y")), CLAIM, STAR, PLANET)
    assert_allclose(on_y, on_x, rtol=1e-10)

    tilted = _belief(clouds["y"], _roles("y", jacobian=False))
    d = diagnose(tilted, CLAIM, STAR, PLANET, jnp.zeros(2))
    assert jnp.abs(d.q[1] - (1.0 - ndtr(-0.5))) < 3.0 * d.mc_se[1]
    assert jnp.abs(d.q[1] - Q_B) > 20.0 * d.mc_se[1]

    with pytest.raises(ValueError, match="measure"):
        _belief(clouds["y"], _roles("x"))
    mixed = Roles(Gaussian("y", 0.0), Gaussian("x", jnp.zeros(1), jnp.ones(1)))
    with pytest.raises(ValueError, match="verdict"):
        _belief(clouds["y"], mixed)


def test_a_hard_edged_prior_keeps_weights_and_gradients_finite(clouds):
    """B = N(0.5, 1) truncated to x < 2 has zero density on part of the cloud. Its
    q is the truncated truth (Phi(1.5) - Phi(0.5)) / Phi(1.5) = 0.2590 within 3 se, the
    weights are finite and zero beyond the edge, and the gradient of q in B's mean
    matches finite differences. One draw outside p0's support (log_weight and
    log_ref_density both -inf) gets weight zero, not nan.

    Fails if a zero prior or a zero target gives nan weights (-inf - (-inf)), or the
    gradient through the `jnp.where` edge is nan or wrong.
    """
    atoms = clouds["x"]
    dead = eqx.tree_at(
        lambda a: (a.cloud.log_weight, a.cloud.log_ref_density),
        atoms,
        (
            atoms.cloud.log_weight.at[0].set(-jnp.inf),
            atoms.cloud.log_ref_density.at[0].set(-jnp.inf),
        ),
    )

    def q_b(mean):
        return q(_belief(dead, _roles(b_mean=mean, upper=2.0)), CLAIM, STAR, PLANET)[1]

    b = _belief(dead, _roles(upper=2.0))
    w = weights(b)
    x = atoms.cloud.samples["x"][:, 0]
    assert jnp.all(jnp.isfinite(w))
    assert jnp.all(w[:, 0] == 0.0)
    assert jnp.all(w[1, x >= 2.0] == 0.0)

    d = diagnose(b, CLAIM, STAR, PLANET, jnp.zeros(2))
    truncated = (ndtr(1.5) - ndtr(0.5)) / ndtr(1.5)
    assert jnp.abs(d.q[1] - truncated) < 3.0 * d.mc_se[1]

    check_grads(q_b, (0.5,), order=1, modes=["fwd", "rev"])


def test_mc_se_is_calibrated():
    """Over 200 replicates of the test-1 cloud with 2,000 draws, the spread of q_B
    matches the mean reported mc_se within 15 percent. The Monte Carlo error of the
    spread is 1 / sqrt(2 * 200) = 5 percent, so 15 percent is 3 sigma.

    Fails if mc_se is the error of the wrong estimator (unnormalized weights, or
    without the delta-method centering on q), or is off by a factor of sqrt(n).
    """

    @jax.jit
    def one(key):
        b = _belief(_cloud(key, 2000), _roles())
        d = diagnose(b, CLAIM, STAR, PLANET, jnp.zeros(2))
        return d.q[1], d.mc_se[1]

    q_hat, se = jax.vmap(one)(jax.random.split(jax.random.PRNGKey(0), 200))
    assert_allclose(jnp.std(q_hat, ddof=1), jnp.mean(se), rtol=0.15)


def test_members_by_ratio():
    """Two draws of base weight 0.71 (Earth-box) and 0.29 (look-alike); member w
    multiplies the look-alike weight. psi(w) = 0.71 / (0.71 + 0.29 w): 0.8032 at
    w = 0.6, 0.7100 at w = 1 and 0.6048 at w = 1.6. The Kish size of each member is
    (0.71 + 0.29 w)^2 / (0.71^2 + (0.29 w)^2).

    Fails if the stacked members are not applied one per row of the role axis (each
    row would get the same prior, or the forecast's), or Kish is inverted.
    """
    atoms = _atoms(
        jnp.array([2.0, 0.0]), jnp.log(jnp.array([0.71, 0.29])), jnp.zeros(2), "x"
    )
    member_w = jnp.array([0.6, 1.0, 1.6])
    roles = Roles(LookAlikeRatio("x", 1.0), LookAlikeRatio("x", member_w))
    b = role_log_weights(atoms, roles, jnp.zeros((4, 2)), STAR, _empty_pool(4))
    d = diagnose(b, CLAIM, STAR, PLANET, jnp.zeros(4))
    w = jnp.concatenate([jnp.ones(1), member_w])
    look = 0.29 * w
    assert_allclose(d.q, 0.71 / (0.71 + look), rtol=1e-12)
    assert_allclose(d.q[1:], [0.8032, 0.7100, 0.6048], atol=1e-3)
    assert_allclose(d.n_eff, (0.71 + look) ** 2 / (0.71**2 + look**2), rtol=1e-12)
