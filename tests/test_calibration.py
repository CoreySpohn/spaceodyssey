"""Calibration of the certificate guarantee over worlds (layer 4 test 6).

Over worlds drawn from the verdict's own prior, with rows from the model the verdict
inverts and exact q, the claim is false on an issued positive certificate with
probability 1 - q on average, for any rule of issue that reads only the data. The test
draws worlds, takes their rows with `step` under `vmap`, replays them through `replay`,
`absorb` and `replay_belief`, issues adaptively, and checks the identity.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.stats import norm

from spaceodyssey.belief import absorb, q, replay, replay_belief
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.design import Mode, Resources
from spaceodyssey.log import grant_rows, start_log, step
from spaceodyssey.roles import Roles
from spaceodyssey.toy import GaussianGasModel
from spaceodyssey.types import (
    AbstractRolePrior,
    Action,
    Atoms,
    Grant,
    Refit,
    WeightedCloud,
)
from spaceodyssey.world import draw_worlds, run_keys

KEY = jax.random.key(7)
N_WORLD, N_DRAW = 3000, 1000
ALPHA = 0.2  # a loose risk, so enough certificates issue
MODELS = (GaussianGasModel(snr_one_day=[2.5]),)  # a one-day row reads x with sd 1
MODE = Mode(0, 0.0)
CLAIM = Claim((GasAbove(0, 0.0),))  # x > 0
PLANET = jnp.asarray(0)
STAR = ()
GATE = {"min_n_eff": 100.0, "max_khat": 0.7}  # every world passes it here


class ExpGasPrior(AbstractRolePrior):
    """N(mean, sd^2) on the gas abundance x, its density taken on the chart
    y = exp(x): the log Jacobian -x is part of the density.
    """

    mean: jax.Array = eqx.field(converter=jnp.asarray)
    sd: jax.Array = eqx.field(converter=jnp.asarray, default=1.0)

    def log_prob(self, atoms, star):
        x = atoms.x_mean[:, 0, 0]
        return norm.logpdf(x, self.mean, self.sd) - x

    def sample(self, key, star):
        raise NotImplementedError("read only as a weight")


def _cloud(key, mean, sd, prior):
    """Point draws of x from the proposal N(mean, sd^2), weighted to the refit's
    prior `prior` (no rows), which is N(prior.mean, prior.sd^2) on x.

    The weight is a ratio of two densities on one chart, so a Jacobian cancels in it.
    """
    x = mean + sd * jax.random.normal(key, (N_DRAW,))
    log_p0 = norm.logpdf(x, prior.mean, prior.sd)
    one = jnp.ones((N_DRAW, 1))
    cloud = WeightedCloud(
        {"hz": one, "size": one},
        log_p0 - norm.logpdf(x, mean, sd),
        log_p0 - x if prior.chart == "exp" else log_p0,
        jnp.nan,
        prior,
    )
    block = x[:, None, None]
    return Atoms(cloud, block, jnp.zeros_like(block), one > 0)


def _issue(toy, roles, cloud):
    """The adaptive rule over worlds: issue after the first row if its q reaches
    1 - alpha, else after the second if that does. Returns q at issue, whether issued,
    and the truth bit.
    """
    worlds = draw_worlds(KEY, toy.GasPrior("toy"), jnp.arange(1), N_WORLD)
    truth = np.asarray(worlds.unseal().x_mean[:, 0, 0, 0, 0] > 0.0)

    no_int = jnp.zeros(0, int)
    grant = Grant(Action(no_int, no_int, jnp.zeros(0)))
    one = jax.tree.map(lambda leaf: leaf[0], worlds)
    payload = one.reveal(KEY, grant, MODELS, (MODE,))
    empty = start_log(grant_rows(grant, payload, (MODE,)), 2)
    log = toy.stack(*[empty] * N_WORLD)
    noise = jax.vmap(lambda i: run_keys(jax.random.fold_in(KEY, i))[1])(
        jnp.arange(N_WORLD)
    )
    take = eqx.filter_jit(
        eqx.filter_vmap(step, in_axes=(0, 0, 0, None, None, None, None))
    )
    action, budget = Action(0, 0, 1.0), Resources(10.0)
    for _ in range(2):
        log = take(noise, log, worlds, MODELS, action, MODE, budget)

    refit, pool = Refit(cloud, jnp.zeros(2, bool), 0), toy.empty_pool(2, 1)

    @jax.jit
    @jax.vmap
    def stage_q(log):
        first = replay(refit, log, MODELS, STAR, jnp.asarray(1.0))
        second = absorb(first, log, MODELS, STAR, jnp.asarray(2.0))
        return jnp.stack(
            [
                q(
                    replay_belief(r, roles, MODELS, STAR, pool, **GATE),
                    CLAIM,
                    STAR,
                    PLANET,
                )[1]
                for r in (first, second)
            ]
        )

    q1, q2 = np.asarray(stage_q(log)).T
    first = q1 >= 1.0 - ALPHA
    second = ~first & (q2 >= 1.0 - ALPHA)
    return np.where(first, q1, q2), first | second, truth


def _check(q_at, issued, truth):
    """The statistic T and the bound on wrong certificates."""
    q_s, wrong = q_at[issued], ~truth[issued]
    spread = np.sqrt(np.sum(q_s * (1.0 - q_s)))
    t = np.sum(wrong - (1.0 - q_s)) / spread
    assert issued.sum() > 600  # power: about 900 of 3,000 worlds issue
    assert abs(t) < 3.0, t
    assert wrong.sum() <= ALPHA * issued.sum() + 3.0 * spread


@pytest.mark.medium(reason="3,000 worlds on two clouds, about 10 s with compiles")
def test_certificates_are_calibrated_over_worlds(toy):
    """Wrong certificates match 1 - q among the issued, with an adaptive rule.

    3,000 worlds with x ~ N(0, 1), two rows y = x + eps with eps ~ N(0, 1), the claim
    x > 0, the verdict prior N(0, 1); a positive certificate issues at alpha = 0.2
    after the first row whose q reaches 0.8. With T = sum over issued (wrong - (1 - q))
    / sqrt(sum q (1 - q)), |T| < 3 and the wrong certificates number at most
    alpha times the issued plus 3 sqrt(sum q (1 - q)). Run twice: 1,000 draws from the
    prior on x, which is also the refit's prior; then 1,000 draws from N(1.0, 1.4^2),
    weighted to a refit prior pi_0 = N(0.25, 1.2^2) and read under the verdict by the
    ratio N(0, 1) / pi_0, every density taken on y = exp(x).

    Fails if a Jacobian is dropped from one side (here from pi_0), the prior ratio
    enters with the wrong sign, or the refit weight is dropped: each moves q away from
    the frequency of the claim among the issued. Measured on the second run: the
    issued fall to 513 (T = -6.0), T = 11.0 and T = 6.8 for these three, against
    T = -1.5 and 0.1 here, about 950 issued.
    """
    same = Roles(toy.GasPrior("toy"), toy.GasPrior("toy", mean=[0.0], sd=[1.0]))
    exact = _cloud(jax.random.key(1), 0.0, 1.0, toy.GasPrior("toy"))
    _check(*_issue(toy, same, exact))
    on_exp = Roles(ExpGasPrior("exp", 0.0), ExpGasPrior("exp", [0.0], [1.0]))
    wider = _cloud(jax.random.key(2), 1.0, 1.4, ExpGasPrior("exp", 0.25, 1.2))
    _check(*_issue(toy, on_exp, wider))
