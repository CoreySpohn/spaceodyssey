"""Builders and boundaries for the refresh tests (not a test file)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import Audience
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, I64, EffectKind, RefitStatus, Scope
from spaceodyssey.core.kinds import AbstractPolicy, ObserverCapabilities
from spaceodyssey.core.records import KindRegistry, PayloadColumn, RecordKind
from spaceodyssey.core.refresh import (
    REFRESH_PRODUCT_KIND,
    AbstractRefitBoundary,
    HostLoopRefit,
    MaskedBatchRefit,
    RefreshSchedule,
)
from spaceodyssey.studies import oracle, toy
from tests.core.helpers import worlds


def refresh_study(every=100, **kw):
    return toy.survey_refresh(every, **kw)


class SpyBoundary(AbstractRefitBoundary):
    """Wraps a boundary and records each request batch it is asked to answer."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []
        self.results = []

    def refit_batch(self, study, requests):
        self.calls.append(jax.tree_util.tree_map(np.asarray, requests))
        out = self.inner.refit_batch(study, requests)
        self.results.append(jax.tree_util.tree_map(np.asarray, out))
        return out


class FailingFor(AbstractRefitBoundary):
    """Answers like `inner`, but reports `status` for one audience (0 or 1)."""

    def __init__(self, inner, audience, status=RefitStatus.NO_SUPPORT):
        self.inner, self.audience, self.status = inner, audience, status

    def refit_batch(self, study, requests):
        out = self.inner.refit_batch(study, requests)
        bad = jnp.zeros_like(out.status).at[:, self.audience].set(int(self.status))
        mask = jnp.arange(2)[None, :, None] == self.audience
        return eqx.tree_at(lambda r: r.status, out, jnp.where(mask, bad, out.status))


class CorruptInstall(AbstractRefitBoundary):
    """Answers like `inner` with the mean slice shifted: a history mutant."""

    def __init__(self, inner):
        self.inner = inner

    def refit_batch(self, study, requests):
        out = self.inner.refit_batch(study, requests)
        first = out.slices[0] + 1.0
        return eqx.tree_at(lambda r: r.slices, out, (first, *out.slices[1:]))


def masked():
    return toy.toy_masked_refit()


def host_loop():
    return HostLoopRefit(toy.toy_refit_fit)


def run_pair(study, n_steps=60, n=2, chunk=7, refit=None):
    """Host and compiled runs of the first `n` worlds with the same boundary."""
    from spaceodyssey.core import CompiledExecutor, HostExecutor

    refit = masked() if refit is None else refit
    truths, roots = worlds(n)
    h = HostExecutor().run_worlds(study, truths, roots, n_steps, refit=refit)
    c = CompiledExecutor().run(study, truths, roots, n_steps, refit=refit, chunk=chunk)
    return h, c


def toy_masked_for(study):
    return MaskedBatchRefit(toy.toy_mean_kernel)


class LeastSeenPolicy(AbstractPolicy):
    """Observe the least-seen of the first two subjects by the policy belief's count.

    The policy belief of a refit-only observer changes only at checkpoints, so every
    decision here reads what a refresh installed. Drains after six decisions.
    """

    def decide(self, pstate, view, study, key):
        stop = view.n_dec >= 6
        subject = jnp.argmin(view.belief.n[:2]).astype(I64)
        none = jnp.asarray(EffectKind.NONE, I64)
        eff = toy._effects(
            study.caps,
            kind=jnp.stack(
                [jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE), none]
            ),
            subject=subject,
            channel_index=toy._chan_index(study, 10),
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=toy.Disposition.DRAIN,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


def policy_study(every=100, **kw):
    return refresh_study(every, policies=(LeastSeenPolicy(),), **kw)


class ToyRefitAnswers(toy.ToyRefitMean):
    """The refit-only mean that also refits service answers."""

    def capabilities(self):
        return ObserverCapabilities(
            incremental=False, refit=True, refit_kinds=("measurement", "answer")
        )


def oracle_refresh_study(
    every=130, *, answers_refit=False, query_at=1, plan=oracle.DEFAULT_PLAN, **overrides
):
    """The oracle study (one Query by the policy) on a refresh grid.

    Returns the study and the service that answers it. With `answers_refit` a service
    answer is itself an eligible row.
    """
    registry = toy.refresh_registry()
    kind = registry.register(oracle.ANSWER_KIND)
    observer = ToyRefitAnswers() if answers_refit else toy.ToyRefitMean()
    caps = registry.lower(Capacities(rows_per_subject=8, refit_lanes=2), observer)
    policy = oracle.QueryPolicy(
        inner=toy.FixedPlanPolicy(plan=plan),
        query_at=query_at,
        subject=0,
    )
    schedule = RefreshSchedule(every, 0, *toy.PRODUCT_CHANNELS, toy.PRODUCT_KIND)
    study = toy.survey_1(
        caps, observer=observer, refresh=schedule, policies=(policy,), **overrides
    )
    return study, oracle.ToyOracleService(kind)


def observer_only_column_study(every=100):
    """A refresh study whose measurement kind carries an observer-only fraction column.

    Both audiences accept the same rows, yet the policy's copy has the column masked,
    so the two audiences' inputs differ and cannot share one computation.
    """
    measurement = RecordKind(
        "measurement",
        (
            PayloadColumn("value", Audience.OBSERVER | Audience.POLICY),
            PayloadColumn("variate", Audience.SCORER),
            PayloadColumn("fraction", Audience.OBSERVER),
        ),
    )
    registry = KindRegistry(
        (RecordKind("none", ()), measurement, toy.AUXILIARY_KIND, REFRESH_PRODUCT_KIND)
    )
    observer = toy.ToyRefitMean()
    caps = registry.lower(Capacities(rows_per_subject=8, refit_lanes=2), observer)
    schedule = RefreshSchedule(every, 0, *toy.PRODUCT_CHANNELS, toy.PRODUCT_KIND)
    return toy.survey_1(caps, observer=observer, refresh=schedule)


def product_rows(carry):
    """The valid refresh product rows of one world as sorted tuples."""
    log = carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.kind) == toy.PRODUCT_KIND)
    cols = [
        np.asarray(x)[keep] for x in (log.channel, log.subject, log.acquisition_index)
    ]
    cols += [np.asarray(log.available_tick)[keep], np.asarray(log.payload)[keep, 0]]
    cols += [np.asarray(log.payload)[keep, 1]]
    return sorted(zip(*(c.tolist() for c in cols), strict=True))


def handled_steps(trace):
    """`(kind, tick, stage, n_written, snapshot_fp)` of each handled step."""
    h = np.asarray(trace.handled)
    pick = lambda x: np.asarray(x)[h].tolist()  # noqa: E731
    return list(
        zip(
            pick(trace.kind),
            pick(trace.tick),
            pick(trace.stage),
            pick(trace.n_written),
            pick(trace.snapshot_fp),
            strict=True,
        )
    )


def resume_host(study, carry, truth, root, first, n_steps, refit):
    """The host loop of `HostExecutor.run` from step `first` to `n_steps` on `carry`.

    Returns the carry and `(kind, tick, handled)` of each step it ran, so a state set
    by hand goes through the real compiled advance, decision, admission and refresh
    boundary. Array-form policies only.
    """
    from spaceodyssey.core.executors.host import HostExecutor, _adm, _close, _fp
    from spaceodyssey.core.lift import lift_traced
    from spaceodyssey.core.refresh import answer_refresh
    from spaceodyssey.core.snapshot import effects_none
    from spaceodyssey.core.transaction import mask_effects

    lifted = lift_traced(study)
    ex, caps = HostExecutor(), study.caps
    seen = []
    for step in range(first, n_steps):
        carry, trace, wake = ex.advance(carry, root, lifted, truth)
        if bool(wake):
            eff, pstate, _u = ex.decide(carry, root, lifted)
        else:
            eff, pstate = effects_none(caps), carry.pstate
        eff = mask_effects(eff, wake)
        fp = _fp(carry, lifted) if bool(wake) else jnp.asarray(0, jnp.int64)
        carry, _codes = _adm(carry, eff, lifted, fp)
        carry = _close(carry, eff, pstate, wake, lifted)
        carry = answer_refresh(
            carry, root, lifted, refit, can_finish=step + 1 < n_steps
        )
        seen.append((int(trace.kind), int(trace.tick), bool(trace.handled)))
    return carry, seen
