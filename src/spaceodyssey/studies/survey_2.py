"""Adaptive search: a revisit policy on a detection-count belief.

The policy reads the per-subject mean and sample count of its belief summary, turns
them into a signal-to-noise ratio, and revisits the subject that is closest to a
detection. A detection is a ratio above the objective's threshold, so the belief
implies a detection count, and the policy stops when every candidate subject is
detected. Declared `ignore_belief`, the same policy takes its plan entries and reads
nothing: the study then equals `survey_1`.

With `every_ticks > 0` the observer is also refit at checkpoints (the refresh
protocol), which changes the belief's values by no more than a rounding error and never
wakes the policy. `GaussianSignals` and `GaussianMean` give the study a conjugate
Gaussian prior whose posterior is calibrated for any design that reads only data.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.kinds import (
    AbstractPolicy,
    AbstractWorldSource,
    ObserverCapabilities,
)
from spaceodyssey.core.snapshot import MeanBelief
from spaceodyssey.studies import toy

# The survey plan of `survey_1`: subject and channel id of each acquisition.
SURVEY_PLAN = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))


class RevisitPolicy(AbstractPolicy):
    """Observe the candidate subject closest to a detection, one acquisition at a time.

    The candidates are `subjects`. A subject with no sample yet comes first, lowest
    index first; after that the choice is the undetected subject with the highest
    signal-to-noise ratio, `mean sqrt(n) / noise`, lowest index on a tie, where `noise`
    and the detection threshold are the study's own (`channel_model.noise[0]` and
    `objective.threshold`, as the objective uses them). When every candidate is
    detected the policy stops. The channel is that of the plan entry at the position
    reached, and the plan length bounds the number of acquisitions.

    With `ignore_belief` the policy takes plan entry `k` at its `k`-th decision and
    never stops early. The policy reads only its belief summary and uses no random
    draw (the draw it returns is recorded, as the plan policy's is).
    """

    plan: tuple = eqx.field(default=SURVEY_PLAN, static=True)
    subjects: tuple = eqx.field(default=(0, 1), static=True)
    ignore_belief: bool = eqx.field(default=False, static=True)

    def declare(self):
        """Observation effects and the closing campaign drain."""
        return Declaration(
            effects=frozenset((EffectKind.OBSERVE, EffectKind.STOP)),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        )

    def decide(self, pstate, view, study, key):
        """Revisit the subject closest to a detection, or take the next plan entry."""
        plan = jnp.asarray(self.plan, I64)
        pos = pstate[0].astype(I64)
        spent = pos >= plan.shape[0]
        at = jnp.minimum(pos, plan.shape[0] - 1)
        channel = jnp.argmax(study.channels.id == plan[at, 1]).astype(I64)
        if self.ignore_belief:
            subject, stop = plan[at, 0], spent
        else:
            subject, done = self._choice(view, study)
            stop = spent | done
        eff = toy._effects(
            study.caps,
            kind=jnp.stack(
                [
                    jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE),
                    jnp.asarray(EffectKind.NONE, I64),
                ]
            ),
            subject=subject,
            channel_index=channel,
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        # the policy's own draw is unused; it is recorded as the plan policy's is
        new_ps = pstate.at[0].add((~stop).astype(F64))
        return eff, new_ps, jr.uniform(key, dtype=F64)

    def _choice(self, view, study):
        """The subject to revisit, and whether every candidate is already detected."""
        candidates = jnp.asarray(self.subjects, I64)
        n = view.belief.n[candidates]
        mean = view.belief.mean[candidates]
        noise = study.channel_model.noise[0]
        snr = mean * jnp.sqrt(n.astype(F64)) / noise
        detected = snr > study.objective.threshold
        score = jnp.where(n == 0, jnp.inf, jnp.where(detected, -jnp.inf, snr))
        return candidates[jnp.argmax(score)], jnp.all(detected & (n > 0))


class SearchMean(toy.ToyRunningMean):
    """The running mean, folded as records are released and refit at checkpoints.

    Its refit folds the subject's rows in order, so an installed belief equals the
    incremental one up to rounding.
    """

    def capabilities(self):
        """Incremental and refit-capable on measurement rows."""
        return ObserverCapabilities(refit=True, refit_kinds=("measurement",))


class GaussianSignals(AbstractWorldSource):
    """Independent Gaussian signals: the prior `GaussianMean` is calibrated for."""

    mean: jax.Array = eqx.field(default=1.0, converter=lambda x: jnp.asarray(x, F64))
    std: jax.Array = eqx.field(default=0.3, converter=lambda x: jnp.asarray(x, F64))

    def draw(self, key, capacities):
        """One signal per subject from N(mean, std^2)."""
        z = jr.normal(key, (capacities.n_subjects,), dtype=F64)
        return toy.ToyTruth(signal=self.mean + self.std * z)


class GaussianMean(toy.ToyRunningMean):
    """The conjugate posterior of a Gaussian signal under a Gaussian prior.

    `mean` holds the posterior mean and `n` the sample count; the posterior standard
    deviation is `posterior_std(n)`. Every channel is taken to have noise `noise`. The
    posterior is calibrated, whatever policy chose the samples, when the truth is drawn
    from the same prior; a wrong `prior_std` or `noise` breaks that.
    """

    prior_mean: float = eqx.field(default=1.0, static=True)
    prior_std: float = eqx.field(default=0.3, static=True)
    noise: float = eqx.field(default=0.5, static=True)

    def init(self, caps):
        """Every subject starts at the prior mean with no samples."""
        base = super().init(caps)
        return MeanBelief(jnp.full_like(base.mean, self.prior_mean), base.n, base.aux)

    def fold(self, belief, rec, caps, key=None):
        """Add one completed measurement to the subject's posterior."""
        t = rec.subject
        take = (
            rec.valid
            & (rec.kind == self.measurement_kind)
            & (rec.status == RecordStatus.COMPLETED)
        )
        prior_precision = 1.0 / self.prior_std**2
        data_precision = 1.0 / self.noise**2
        n = belief.n[t].astype(F64)
        precision = prior_precision + n * data_precision
        new = (belief.mean[t] * precision + rec.payload[0] * data_precision) / (
            precision + data_precision
        )
        return MeanBelief(
            mean=belief.mean.at[t].set(jnp.where(take, new, belief.mean[t])),
            n=belief.n.at[t].add(take.astype(I64)),
            aux=belief.aux,
        )

    def posterior_std(self, n):
        """Posterior standard deviation after `n` samples."""
        n = jnp.asarray(n, F64)
        return jax.lax.rsqrt(1.0 / self.prior_std**2 + n / self.noise**2)


def gaussian_pit(observer):
    """The PIT function of `observer`'s posterior: the CDF at the true signal."""

    def pit(belief, truth):
        z = (truth.signal - belief.mean) / observer.posterior_std(belief.n)
        return jstats.norm.cdf(z), belief.n > 0

    return pit


def survey_2(every_ticks=0, policy=None, **overrides):
    """Build the adaptive search, with or without a refresh grid.

    `every_ticks=0` is the plain survey: the running-mean observer, `survey_1`'s
    capacities and no checkpoints. A positive value adds a checkpoint grid every
    `every_ticks` ticks and the refit-capable `SearchMean` observer (`observer` in
    `overrides` replaces it); the study then needs a refit boundary to run. `policy`
    defaults to the adaptive `RevisitPolicy`.
    """
    policy = RevisitPolicy() if policy is None else policy
    if every_ticks == 0:
        return toy.survey_1(policies=(policy,), **overrides)
    overrides.setdefault("observer", SearchMean())
    return toy.survey_refresh(every_ticks, policies=(policy,), **overrides)


__all__ = [
    "SURVEY_PLAN",
    "GaussianMean",
    "GaussianSignals",
    "RevisitPolicy",
    "SearchMean",
    "gaussian_pit",
    "survey_2",
]
