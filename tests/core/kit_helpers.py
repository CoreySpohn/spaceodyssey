"""Observers and shared runs for the observer kit (not a test file).

The kit study is the toy survey with four subjects observed twice each on channels
of one noise level, so a flat-prior Gaussian posterior with that noise is exact.
Subjects 0 and 2 are observed on channel 11 before channel 10: their release order
is not their canonical order. Each observer runs once, 400 worlds batched in one
compiled run, cached for the process (one compiled program per observer type).
"""

import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats

from spaceodyssey.core import (
    F64,
    CompiledExecutor,
    MeanBelief,
    ObserverCapabilities,
    where_tree,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance import make_worlds
from spaceodyssey.studies import toy

NOISE = 0.5
N_WORLDS = 400
N_STEPS = 24
REFIT = ObserverCapabilities(refit=True, refit_kinds=("measurement",))
PLAN = ((0, 11), (1, 10), (2, 11), (3, 10), (0, 10), (1, 11), (2, 10), (3, 11))


class RefitMean(toy.ToyRunningMean):
    """The toy running mean, declared refit-capable on measurement rows.

    `ignore_odd` is the calibration control, off by default: when set, a fold
    counts every record but takes the value of even acquisitions only, so on a
    subject with two records the posterior claims both and uses one (too narrow by
    the square root of two). It is an array leaf, so the control shares the
    compiled program of the honest observer; when off, the fold is the toy
    running mean's, bit for bit.
    """

    ignore_odd: jax.Array = eqx.field(
        default=False, converter=lambda x: jnp.asarray(x, bool)
    )

    def capabilities(self):
        return REFIT

    def fold(self, belief, rec, caps, key=None):
        folded = super().fold(belief, rec, caps, key=key)
        ignored = self.ignore_odd & (rec.acquisition_index % 2 == 1)
        mean = jnp.where(ignored, belief.mean, folded.mean)
        return MeanBelief(mean, folded.n, folded.aux)


class RefitSample(toy.ToySampleObserver):
    """The toy sampling observer, declared refit-capable on measurement rows."""

    def capabilities(self):
        return REFIT


def fold_last_again(observer, fitted, rows, caps, row_keys):
    """`fitted` with the last counting row of `rows` folded a second time."""
    last = rows.valid.shape[0] - 1 - jnp.argmax(rows.valid[::-1])
    row = jax.tree_util.tree_map(lambda x: x[last], rows)
    again = observer.fold(observer.analyze(fitted), row, caps, key=row_keys[last])
    return where_tree(rows.valid.any(), again, fitted)


class LateTwice(RefitMean):
    """A refit that weights its latest record twice."""

    def refit(self, belief0, rows, caps, key, row_keys):
        fitted = super().refit(belief0, rows, caps, key, row_keys)
        return fold_last_again(self, fitted, rows, caps, row_keys)


class ResampleSample(RefitSample):
    """A statistical refit: the refit subject's posterior draw uses the refit key.

    Its draw is another sample of the posterior the incremental fold sampled, so
    the two agree in distribution, not pathwise. The per-world mean projection
    difference, over the kit study's eight rows and two seeds, has a standard
    deviation near 0.21, so 400 worlds give a standard error near 0.011; the
    declared 0.02 covers it, and the margin is four times that.
    """

    statistical = True
    margin = 0.08
    seeds = (0, 1)
    standard_error = 0.02

    def refit(self, belief0, rows, caps, key, row_keys):
        fitted = super().refit(belief0, rows, caps, key, row_keys)
        subject = rows.subject[jnp.argmax(rows.valid)]
        n = fitted.n[subject]
        scale = self.noise / jnp.sqrt(jnp.maximum(n, 1).astype(F64))
        draw = fitted.mean[subject] + scale * jr.normal(key, dtype=F64)
        sample = fitted.sample.at[subject].set(
            jnp.where(n > 0, draw, fitted.sample[subject])
        )
        return dataclasses.replace(fitted, sample=sample)


class LateTwiceSample(ResampleSample):
    """The statistical refit, weighting its latest record twice."""

    def refit(self, belief0, rows, caps, key, row_keys):
        fitted = super().refit(belief0, rows, caps, key, row_keys)
        return fold_last_again(self, fitted, rows, caps, row_keys)


OBSERVERS = {
    "mean": RefitMean(),
    "sample": RefitSample(noise=NOISE),
    "ignores_half": RefitMean(ignore_odd=True),
}


def kit_study(observer):
    """The kit study run by `observer`, with its refit kinds lowered."""
    caps = toy.toy_registry().lower(Capacities(), observer)
    return toy.survey_1(
        caps=caps,
        observer=observer,
        policies=(toy.FixedPlanPolicy(plan=PLAN),),
        channel_model=toy.ToyPhotometry(noise=jnp.full(caps.n_channels, NOISE, F64)),
        horizon_tick=jnp.asarray(4000),
        phase_until=(4000,),
    )


def pit(belief, truth):
    """Flat-prior Gaussian PIT of each subject's signal, and whether it has data."""
    n = jnp.maximum(belief.n, 1).astype(F64)
    z = (truth.signal - belief.mean) * jnp.sqrt(n) / NOISE
    return jstats.norm.cdf(z), belief.n > 0


@functools.cache
def kit_run(name):
    """`(study, run, truths, roots)` of a named observer: 400 worlds, compiled.

    Every observer sees the same worlds.
    """
    study = kit_study(OBSERVERS[name])
    truths, roots = make_worlds(study, N_WORLDS)
    run = CompiledExecutor().run(study, truths, roots, N_STEPS)
    return study, run, truths, roots


def first_worlds(tree, n):
    """The first `n` worlds of a batched run, or of any tree with a world axis."""
    return jax.tree_util.tree_map(lambda x: x[:n], tree)
