"""Observers and the shared batches of worlds for the observer kit (not a test file).

The kit study is the toy survey with four subjects observed twice each on channels
of one noise level, so a flat-prior Gaussian posterior with that noise is exact.
Subjects 0 and 2 are observed on channel 11 before channel 10: their release order
is not their canonical order. The log holds exactly the eight records.

Two batches of the running mean serve every observer: four worlds on the host
executor for the exact checks, and 2,000 worlds on the compiled executor for
calibration and margins (both slow tier: one executor compile alone is over the
unit budget). Other observers' beliefs over the same records come from
`replay_beliefs`: the plan reads no belief, so the records do not depend on the
observer. Each batch is checked once against a replay of its own observer.
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
    HostExecutor,
    MeanBelief,
    ObserverCapabilities,
    RunResult,
    where_tree,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance import make_worlds, replay_beliefs
from spaceodyssey.core.enums import Audience
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.core.refresh import MaskedBatchRefit, RefreshSchedule
from spaceodyssey.studies import toy
from spaceodyssey.studies.scripted import ScriptedRecords

NOISE = 0.5
N_WORLDS = 2000
HOST_WORLDS = 4
N_STEPS = 24
REFIT = ObserverCapabilities(refit=True, refit_kinds=("measurement",))
PLAN = ((0, 11), (1, 10), (2, 11), (3, 10), (0, 10), (1, 11), (2, 10), (3, 11))


def _flag(default=False):
    return eqx.field(default=default, converter=lambda x: jnp.asarray(x, bool))


class RefitMean(toy.ToyRunningMean):
    """The toy running mean, declared refit-capable on measurement rows."""

    def capabilities(self):
        return REFIT


class ReadyBelief(MeanBelief):
    """A running mean with subject-owned boolean state."""

    ready: jax.Array


class ReadyMean(RefitMean):
    """A boolean install mutant, with the fault off by default."""

    subject_leaves = ("mean", "n", "ready")
    stale_ready: jax.Array = _flag()

    def init(self, caps):
        belief = super().init(caps)
        return ReadyBelief(belief.mean, belief.n, belief.aux, belief.n > 0)

    def fold(self, belief, rec, caps, key=None):
        folded = super().fold(belief, rec, caps, key=key)
        return ReadyBelief(folded.mean, folded.n, folded.aux, folded.n > 0)

    def analyze(self, belief):
        analyzed = super().analyze(belief)
        return ReadyBelief(analyzed.mean, analyzed.n, analyzed.aux, belief.ready)

    def fold_summary(self, belief, rec, caps, key=None):
        return RefitMean().fold_summary(belief, rec, caps, key=key)

    def init_summary(self, caps):
        return RefitMean().init_summary(caps)

    def install(self, belief, fitted, subject):
        put = super().install(belief, fitted, subject)
        ready = belief.ready.at[subject].set(
            jnp.where(self.stale_ready, belief.ready[subject], fitted.ready[subject])
        )
        return ReadyBelief(put.mean, put.n, put.aux, ready)

    def layout(self, caps):
        layout = super().layout(caps)
        belief = layout.belief
        return dataclasses.replace(
            layout,
            belief=ReadyBelief(
                belief.mean,
                belief.n,
                belief.aux,
                jax.ShapeDtypeStruct(belief.n.shape, jnp.bool_),
            ),
        )


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
    stochastic_leaves = ("sample",)

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
    - `tick_aux`: a fold adds the record's `accepted_tick` to `aux[1]`, so its
      belief depends on the record's metadata as the handler presented it.
    """

    ignore_odd: jax.Array = _flag()
    late_twice: jax.Array = _flag()
    noop_install: jax.Array = _flag()
    canonical_bug: jax.Array = _flag()
    tick_aux: jax.Array = _flag()

    def fold(self, belief, rec, caps, key=None):
        folded = super().fold(belief, rec, caps, key=key)
        ignored = self.ignore_odd & (rec.acquisition_index % 2 == 1)
        mean = jnp.where(ignored, belief.mean, folded.mean)
        tick = jnp.where(self.tick_aux & rec.valid, rec.accepted_tick, 0)
        aux = folded.aux.at[1].add(tick.astype(F64))
        return MeanBelief(mean, folded.n, aux)

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
    - `canonical_mean`: a refit adds 100 to its (deterministic) mean when its first
      counting row is not its subject's first acquisition (canonical order only).
    - `wide_refit`: a refit's draw has twice the posterior's standard deviation.
    - `one_row_wide_refit`: twice the draw's standard deviation at one row only.
    - `stale_install`: `install` leaves the subject's mean as it was.
    - `leak_install`: `install` adds 100 to the next subject's mean.
    """

    late_twice: jax.Array = _flag()
    nan_refit: jax.Array = _flag()
    cancelling: jax.Array = _flag()
    corrupt_install: jax.Array = _flag()
    nan_sample: jax.Array = _flag()
    canonical_mean: jax.Array = _flag()
    wide_refit: jax.Array = _flag()
    one_row_wide_refit: jax.Array = _flag()
    stale_install: jax.Array = _flag()
    leak_install: jax.Array = _flag()

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
        first = jnp.argmax(rows.valid)
        subject = rows.subject[first]
        shift = jnp.where(self.cancelling, jnp.where(subject % 2 == 0, 1.0, -1.0), 0.0)
        spread = fitted.sample[subject] - fitted.mean[subject]
        wide = jnp.where(
            self.wide_refit | (self.one_row_wide_refit & (fitted.n[subject] == 1)),
            spread,
            0.0,
        )
        sample = fitted.sample.at[subject].add(shift + wide)
        sample = jnp.where(self.nan_refit, jnp.nan, sample)
        late = rows.valid.any() & (rows.acquisition_index[first] > 0)
        bump = jnp.where(self.canonical_mean & late, 100.0, 0.0)
        mean = fitted.mean.at[subject].add(bump)
        return dataclasses.replace(fitted, sample=sample, mean=mean)

    def install(self, belief, fitted, subject):
        put = super().install(belief, fitted, subject)
        other = (subject + 1) % put.sample.shape[0]
        bump = jnp.where(self.corrupt_install, 100.0, 0.0)
        stale = jnp.where(self.stale_install, belief.mean[subject], put.mean[subject])
        mean = put.mean.at[subject].set(stale)
        mean = mean.at[other].add(jnp.where(self.leak_install, 100.0, 0.0))
        return dataclasses.replace(
            put, sample=put.sample.at[other].add(bump), mean=mean
        )


OBSERVERS = {
    # the mutant with every fault off, so its mutants share the batch's programs
    "mean": MeanMutant(),
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


def _checked(study, run, roots):
    """`run`, after a replay of its own observer reproduces its beliefs."""
    replayed = replay_beliefs(study, run, roots)
    for got, want in zip(
        jax.tree_util.tree_leaves(replayed),
        jax.tree_util.tree_leaves((run.carry.belief, run.carry.pbelief)),
        strict=True,
    ):
        np.testing.assert_allclose(got, want, rtol=0, atol=GO_TOLERANCE)
    return run


@functools.cache
def kit_batch(source):
    """`(study, run, truths, roots)` of the running mean: "host" or "compiled".

    The host batch is four worlds, the compiled batch 2,000. Raises
    `AssertionError` if a replay of the running mean does not reproduce the run's
    beliefs, which every replayed observer relies on.
    """
    study = kit_study(OBSERVERS["mean"])
    n_worlds = HOST_WORLDS if source == "host" else N_WORLDS
    truths, roots = make_worlds(study, n_worlds)
    if source == "host":
        run = HostExecutor().run_worlds(study, truths, roots, N_STEPS)
    else:
        run = CompiledExecutor().run(study, truths, roots, N_STEPS)
    return study, _checked(study, run, roots), truths, roots


@functools.cache
def kit_run(name, source="compiled"):
    """`(study, run, truths, roots)` of a named observer over a shared batch."""
    if name == "mean":
        return kit_batch(source)
    _, batch, truths, roots = kit_batch(source)
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


class KeepsSuperseded(RefitMean):
    """The running mean whose `install` can merge into what the subject holds.

    With `merge` on, an install blends the refit into the subject's current belief,
    so a row a correction superseded stays in the belief (the fault of the delayed
    correction check). The fault is an array leaf, so both settings share a program.
    """

    merge: jax.Array = _flag()

    def install(self, belief, fitted, subject):
        put = super().install(belief, fitted, subject)
        n = belief.n[subject] + fitted.n[subject]
        blend = (
            belief.mean[subject] * belief.n[subject]
            + fitted.mean[subject] * fitted.n[subject]
        ) / jnp.maximum(n, 1)
        return MeanBelief(
            mean=jnp.where(self.merge, belief.mean.at[subject].set(blend), put.mean),
            n=jnp.where(self.merge, belief.n.at[subject].set(n), put.n),
            aux=put.aux,
        )


SUMMARY_KIND = 4  # index of the summary kind in `summary_registry`
PUBLIC = Audience.OBSERVER | Audience.POLICY
EVERY = 100


def _scripted(observer, caps, ticks, values, ids, revisions=None, kinds=None):
    process = ScriptedRecords(
        ticks=tuple(ticks),
        values=tuple(values),
        ids=tuple(ids),
        revisions=tuple(revisions or (0,) * len(ticks)),
        statuses=(0,) * len(ticks),
        kinds=tuple(kinds or ()),
    )
    schedule = RefreshSchedule(EVERY, 0, *toy.PRODUCT_CHANNELS, toy.PRODUCT_KIND)
    return toy.survey_1(
        caps,
        observer=observer,
        refresh=schedule,
        policies=(toy.NullPolicy(),),
        process=process,
    )


def correction_study(observer):
    """Subject 0 measured as 2 and 4, then acquisition 4 corrected to 9 at tick 150.

    The first checkpoint (tick 100) installs {2, 4}, the second (tick 200) the
    corrected set {9, 4}: mean 6.5 over two rows.
    """
    caps = toy.refresh_registry().lower(
        Capacities(rows_per_subject=8, refit_lanes=2), observer
    )
    return _scripted(observer, caps, (1, 2, 150), (2.0, 4.0, 9.0), (4, 5, 4), (0, 0, 1))


class SummaryMean(RefitMean):
    """The running mean of measurements, that counts a summary row like one if given.

    With `count_summary` the observer lists the summary kind for refit, which a
    registry that marks the kind derived refuses. A derived summary never reaches a
    fold or a refit, so the same fold is correct there.
    """

    count_summary: bool = eqx.field(default=False, static=True)

    def capabilities(self):
        kinds = ("measurement", "summary") if self.count_summary else ("measurement",)
        return ObserverCapabilities(refit=True, refit_kinds=kinds)

    def fold(self, belief, rec, caps, key=None):
        kind = jnp.where(rec.kind == SUMMARY_KIND, self.measurement_kind, rec.kind)
        return super().fold(belief, dataclasses.replace(rec, kind=kind), caps, key=key)


def summary_registry(derived):
    """The refresh kinds with a `summary` kind at index 4, derived or not."""
    summary = RecordKind("summary", (PayloadColumn("value", PUBLIC),), derived=derived)
    base = toy.refresh_registry()
    return KindRegistry((*(base[i] for i in range(len(base))), summary))


def summary_study(observer, *, derived, ids=(6,), revisions=None):
    """Measurements 2 and 4 of subject 0, then a summary of them (their maximum, 4).

    The summary is computed from the rows the observer folds, so its pedigree
    overlaps its revision set. Counted as a datum it would make three rows and a
    mean of 10/3 where the data are two rows and a mean of 3.
    """
    caps = summary_registry(derived).lower(
        Capacities(rows_per_subject=8, refit_lanes=2), observer
    )
    n = 2 + len(ids)
    return _scripted(
        observer,
        caps,
        range(1, n + 1),
        (2.0, 4.0) + (4.0,) * len(ids),
        (4, 5, *ids),
        (0, 0, *(revisions or (0,) * len(ids))),
        (1, 1) + (SUMMARY_KIND,) * len(ids),
    )


@functools.cache
def refresh_run(name):
    """`(study, run, roots)` of two worlds of a named scripted refresh study."""
    studies = {
        "correction": lambda: correction_study(KeepsSuperseded()),
        "keeps_superseded": lambda: correction_study(KeepsSuperseded(merge=True)),
        "summary": lambda: summary_study(SummaryMean(), derived=True),
        "summary_twice": lambda: summary_study(SummaryMean(), derived=True, ids=(6, 7)),
        "summary_not_derived": lambda: summary_study(
            SummaryMean(count_summary=True), derived=False
        ),
        "summary_reissued": lambda: summary_study(
            SummaryMean(), derived=True, ids=(6, 6), revisions=(0, 1)
        ),
    }
    study = studies[name]()
    truths, roots = make_worlds(study, 2)
    run = HostExecutor().run_worlds(study, truths, roots, 60, refit=MaskedBatchRefit())
    return study, run, roots
