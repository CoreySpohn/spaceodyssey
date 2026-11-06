"""Observers and the shared batch of worlds for the observer kit (not a test file).

The kit study is the toy survey with four subjects observed twice each on channels
of one noise level, so a flat-prior Gaussian posterior with that noise is exact.
Subjects 0 and 2 are observed on channel 11 before channel 10: their release order
is not their canonical order. The log holds exactly the eight records.

One compiled run of the running mean over 2,000 worlds is the measurement batch.
Every other observer's beliefs over the same records come from `replay_beliefs`:
the plan reads no belief, so the records do not depend on the observer. The batch
is checked once against a replay of its own observer.
"""

import dataclasses
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.stats as jstats
import numpy as np

from spaceodyssey.core import (
    F64,
    GO_TOLERANCE,
    CompiledExecutor,
    MeanBelief,
    ObserverCapabilities,
    RunResult,
    where_tree,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance import make_worlds, replay_beliefs
from spaceodyssey.studies import toy

NOISE = 0.5
N_WORLDS = 2000
N_STEPS = 24
REFIT = ObserverCapabilities(refit=True, refit_kinds=("measurement",))
PLAN = ((0, 11), (1, 10), (2, 11), (3, 10), (0, 10), (1, 11), (2, 10), (3, 11))


def _flag(default=False):
    return eqx.field(default=default, converter=lambda x: jnp.asarray(x, bool))


class RefitMean(toy.ToyRunningMean):
    """The toy running mean, declared refit-capable on measurement rows."""

    def capabilities(self):
        return REFIT


class RefitSample(toy.ToySampleObserver):
    """The toy sampling observer, declared refit-capable on measurement rows.

    Its belief keeps the draw of its latest fold, so its canonical-order refit is
    another sample: it is order-sensitive. Over the kit study's 2,000 worlds the
    largest per-stratum standard error of that projection difference is near
    0.011; the declared 0.02 covers it, and the margin is five times that.
    """

    order_sensitive = True
    margin = 0.1
    seeds = (0,)
    standard_error = 0.02

    def capabilities(self):
        return REFIT


class ResampleSample(RefitSample):
    """A statistical refit: the refit subject's posterior draw uses the refit key.

    Its draw is another sample of the posterior the incremental fold sampled, so
    the two agree in distribution, not pathwise. Averaged over two seeds, the
    difference has a standard deviation near 0.61 for a subject with one row, a
    standard error near 0.014 over 2,000 worlds.
    """

    statistical = True
    seeds = (0, 1)

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


def fold_last_again(observer, fitted, rows, caps, row_keys):
    """`fitted` with the last counting row of `rows` folded a second time."""
    last = rows.valid.shape[0] - 1 - jnp.argmax(rows.valid[::-1])
    row = jax.tree_util.tree_map(lambda x: x[last], rows)
    again = observer.fold(observer.analyze(fitted), row, caps, key=row_keys[last])
    return where_tree(rows.valid.any(), again, fitted)


class MeanMutant(RefitMean):
    """The running mean with array-valued faults, all off by default.

    Faults are array leaves, so every mutant shares one compiled program.

    - `ignore_odd`: a fold counts every record but takes the value of even
      acquisitions only (the posterior claims records it ignored).
    - `late_twice`: a refit folds its latest record a second time.
    - `noop_install`: `install` returns the belief unchanged.
    - `canonical_bug`: a refit adds one to its mean when its first counting row is
      not its subject's first acquisition, which only canonical order produces.
    """

    ignore_odd: jax.Array = _flag()
    late_twice: jax.Array = _flag()
    noop_install: jax.Array = _flag()
    canonical_bug: jax.Array = _flag()

    def fold(self, belief, rec, caps, key=None):
        folded = super().fold(belief, rec, caps, key=key)
        ignored = self.ignore_odd & (rec.acquisition_index % 2 == 1)
        mean = jnp.where(ignored, belief.mean, folded.mean)
        return MeanBelief(mean, folded.n, folded.aux)

    def refit(self, belief0, rows, caps, key, row_keys):
        fitted = super().refit(belief0, rows, caps, key, row_keys)
        fitted = where_tree(
            self.late_twice,
            fold_last_again(self, fitted, rows, caps, row_keys),
            fitted,
        )
        first = jnp.argmax(rows.valid)
        subject = rows.subject[first]
        bug = (
            self.canonical_bug & rows.valid.any() & (rows.acquisition_index[first] > 0)
        )
        return dataclasses.replace(
            fitted, mean=fitted.mean.at[subject].add(jnp.where(bug, 1.0, 0.0))
        )

    def install(self, belief, fitted, subject):
        return where_tree(
            self.noop_install, belief, super().install(belief, fitted, subject)
        )


class SampleMutant(ResampleSample):
    """The statistical refit with array-valued faults, all off by default.

    - `late_twice`: a refit folds its latest record a second time.
    - `nan_refit`: a refit's posterior draw is NaN.
    - `cancelling`: a refit's draw moves by +1 on even subjects and -1 on odd
      ones, so the errors cancel when pooled over subjects.
    - `corrupt_install`: `install` adds 100 to the next subject's draw.
    - `nan_sample`: every fold's posterior draw is NaN.
    """

    late_twice: jax.Array = _flag()
    nan_refit: jax.Array = _flag()
    cancelling: jax.Array = _flag()
    corrupt_install: jax.Array = _flag()
    nan_sample: jax.Array = _flag()

    def fold(self, belief, rec, caps, key=None):
        folded = super().fold(belief, rec, caps, key=key)
        sample = jnp.where(self.nan_sample, jnp.nan, folded.sample)
        return dataclasses.replace(folded, sample=sample)

    def refit(self, belief0, rows, caps, key, row_keys):
        fitted = super().refit(belief0, rows, caps, key, row_keys)
        fitted = where_tree(
            self.late_twice,
            fold_last_again(self, fitted, rows, caps, row_keys),
            fitted,
        )
        subject = rows.subject[jnp.argmax(rows.valid)]
        shift = jnp.where(self.cancelling, jnp.where(subject % 2 == 0, 1.0, -1.0), 0.0)
        sample = fitted.sample.at[subject].add(shift)
        sample = jnp.where(self.nan_refit, jnp.nan, sample)
        return dataclasses.replace(fitted, sample=sample)

    def install(self, belief, fitted, subject):
        put = super().install(belief, fitted, subject)
        other = (subject + 1) % put.sample.shape[0]
        bump = jnp.where(self.corrupt_install, 100.0, 0.0)
        return dataclasses.replace(put, sample=put.sample.at[other].add(bump))


OBSERVERS = {
    "mean": RefitMean(),
    "sample": RefitSample(noise=NOISE),
    "ignores_half": MeanMutant(ignore_odd=True),
    "nan_sample": SampleMutant(noise=NOISE, nan_sample=True),
    "wrong_noise": RefitSample(noise=0.2),
}


def kit_study(observer):
    """The kit study run by `observer`, with its refit kinds lowered."""
    caps = toy.toy_registry().lower(Capacities(n_log=8), observer)
    return toy.survey_1(
        caps=caps,
        observer=observer,
        policies=(toy.FixedPlanPolicy(plan=PLAN),),
        channel_model=toy.ToyPhotometry(noise=jnp.full(caps.n_channels, NOISE, F64)),
        horizon_tick=jnp.asarray(4000),
        phase_until=(4000,),
    )


@functools.cache
def kit_batch():
    """`(study, run, truths, roots)` of the running mean over 2,000 worlds.

    Raises `AssertionError` if a replay of the running mean does not reproduce the
    run's beliefs, which every replayed observer relies on.
    """
    study = kit_study(OBSERVERS["mean"])
    truths, roots = make_worlds(study, N_WORLDS)
    run = CompiledExecutor().run(study, truths, roots, N_STEPS)
    replayed = replay_beliefs(study, run, roots)
    for got, want in zip(
        jax.tree_util.tree_leaves(replayed),
        jax.tree_util.tree_leaves((run.carry.belief, run.carry.pbelief)),
        strict=True,
    ):
        np.testing.assert_allclose(got, want, rtol=0, atol=GO_TOLERANCE)
    return study, run, truths, roots


@functools.cache
def kit_run(name):
    """`(study, run, truths, roots)` of a named observer over the shared batch."""
    if name == "mean":
        return kit_batch()
    _, batch, truths, roots = kit_batch()
    observer = OBSERVERS[name]
    study = kit_study(observer)
    belief, summary = replay_beliefs(study, batch, roots)
    carry = dataclasses.replace(batch.carry, belief=belief, pbelief=summary)
    run = RunResult(carry, batch.trace, observer.layout(study.caps))
    return study, run, truths, roots


def pit_of_mean(belief, truth):
    """Flat-prior Gaussian PIT of each subject's signal from the mean and count."""
    n = jnp.maximum(belief.n, 1).astype(F64)
    z = (truth.signal - belief.mean) * jnp.sqrt(n) / NOISE
    return jstats.norm.cdf(z), belief.n > 0


def pit_of_sample(noise):
    """PIT of each subject's posterior draw against its signal, under `noise`.

    A calibrated draw and the truth are two independent draws of one posterior, so
    their difference has variance twice the posterior's, 2 noise^2 / n.
    """

    def pit(belief, truth):
        n = jnp.maximum(belief.n, 1).astype(F64)
        z = (belief.sample - truth.signal) / (noise * jnp.sqrt(2.0 / n))
        return jstats.norm.cdf(z), belief.n > 0

    return pit


def first_worlds(tree, n):
    """The first `n` worlds of a batched run, or of any tree with a world axis."""
    return jax.tree_util.tree_map(lambda x: x[:n], tree)
