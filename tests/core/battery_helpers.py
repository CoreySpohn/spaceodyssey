"""Studies, policies and broken executors for the battery tests (not a test file).

Each broken executor changes one rule of the real handlers or transaction by cloning
the function with one name rebound, the way `mutants.py` does, so the real code under
test stays untouched.
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey import core
from spaceodyssey.core import handlers, snapshot, transaction
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    Disposition,
    EffectKind,
    RecordStatus,
    Scope,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import frozen
from tests.core.mutants import _clone, _step_with_table

PLAN = ((0, 10),) * 5
STARTS = (0, 150, 300, 450, 600)


def survey(policy=None, **overrides):
    """The one-subject, five-acquisition study the checks run on."""
    return kinds.survey_1(policies=(policy or frozen(plan=PLAN),), **overrides)


def batched_truth(signal, n_worlds, n_subjects=4):
    """`n_worlds` copies of a truth whose every signal is `signal`."""
    return kinds.ToyTruth(signal=jnp.full((n_worlds, n_subjects), float(signal), F64))


def _table(caps, kind0, kind1=EffectKind.NONE, **fields):
    """An effect table: row 0 and row 1 kinds, other columns as given."""
    eff = effects_none(caps)
    cols = {
        n: getattr(eff, n)
        for n in ("subject", "channel_index", "tick", "scope", "disposition", "group")
    }
    for name, value in fields.items():
        cols[name] = cols[name].at[:].set(value)
    return dataclasses.replace(
        eff,
        kind=jnp.stack([jnp.asarray(kind0, I64), jnp.asarray(kind1, I64)]),
        **cols,
    )


class ValueReadingPolicy(AbstractPolicy):
    """Reads the values of the records it may see and acts on their mean.

    Decision 0 observes subject 0 and waits `wait_ticks`. Later decisions do nothing
    while no record is visible, then observe subject 1 when the mean visible value
    exceeds `threshold` and drain the campaign otherwise. A policy that sees a record
    before it is released acts differently at the wait.
    """

    threshold: float = eqx.field(default=2.5)
    wait_ticks: int = eqx.field(default=110)

    def decide(self, pstate, view, study, key):
        """Choose by the mean of the visible values."""
        caps = study.caps
        first = view.n_dec == 0
        rows = view.n_rows
        mean = view.log.payload[:, 0].sum() / jnp.maximum(rows, 1)
        act = jnp.where(mean > self.threshold, EffectKind.OBSERVE, EffectKind.STOP)
        later = jnp.where(rows == 0, EffectKind.NONE, act)
        eff = _table(
            caps,
            jnp.where(first, EffectKind.OBSERVE, later),
            jnp.where(first, EffectKind.WAIT, EffectKind.NONE),
            subject=jnp.where(first, 0, 1),
            channel_index=0,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            group=view.n_dec,
        )
        eff = dataclasses.replace(
            eff, tick=eff.tick.at[1].set(view.tick + self.wait_ticks)
        )
        return eff, pstate, jnp.asarray(0.0, F64)


class AbsolutePlanPolicy(AbstractPolicy):
    """Observes subject 0 at fixed absolute ticks, waiting for the next one.

    Decisions woken by a release do nothing, so the schedule of acquisitions does not
    depend on when records are released.
    """

    starts: tuple[int, ...] = eqx.field(default=STARTS)

    def decide(self, pstate, view, study, key):
        """Observe at a planned tick and wait for the next."""
        starts = jnp.asarray(self.starts, I64)
        at_start = jnp.any(starts == view.tick)
        last = jnp.asarray(self.starts[-1], I64) == view.tick
        following = starts[
            jnp.minimum(jnp.sum(starts <= view.tick), len(self.starts) - 1)
        ]
        eff = _table(
            study.caps,
            jnp.where(at_start, EffectKind.OBSERVE, EffectKind.NONE),
            jnp.where(
                at_start,
                jnp.where(last, EffectKind.STOP, EffectKind.WAIT),
                EffectKind.NONE,
            ),
            subject=0,
            channel_index=0,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            group=view.n_dec,
        )
        eff = dataclasses.replace(eff, tick=eff.tick.at[1].set(following))
        return eff, pstate, jnp.asarray(0.0, F64)


class SelectingPolicy(AbstractPolicy):
    """Five observations of one channel, the subject and channel chosen by a rule.

    `rule` names what the choice reads, which is exactly what a metamorphic relation
    must not let matter: `"smallest_id"` picks the channel with the smallest id,
    `"late_tick"` picks subject 1 once the tick passes 300, `"asset_parity"` picks the
    subject by the parity of the asset count.
    """

    rule: str = eqx.field(static=True)

    def decide(self, pstate, view, study, key):
        """Observe five times by the rule, then drain."""
        caps = study.caps
        subject = {
            "smallest_id": jnp.asarray(0, I64),
            "late_tick": (view.tick >= 300).astype(I64),
            "asset_parity": jnp.asarray(study.caps.n_assets % 2, I64),
        }[self.rule]
        channel_index = (
            jnp.argmin(study.channels.id).astype(I64)
            if self.rule == "smallest_id"
            else jnp.asarray(0, I64)
        )
        stop = view.n_dec >= 5
        eff = _table(
            caps,
            jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE),
            subject=subject,
            channel_index=channel_index,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            group=view.n_dec,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


class StatedPlan(kinds.FixedPlanPolicy):
    """A fixed plan whose declaration is whatever the test says it is."""

    stated: Declaration | None = eqx.field(static=True, default=None)

    def declare(self):
        """The stated declaration, or the plan's own."""
        return super().declare() if self.stated is None else self.stated


class PriorBiasedMean(kinds.ToyRunningMean):
    """Running mean that adds a constant to every record: an informative prior."""

    bias: float = eqx.field(default=1.0)

    def fold(self, belief, rec, caps, key=None):
        """Fold the record with its value pushed up by the bias."""
        biased = dataclasses.replace(rec, payload=rec.payload.at[0].add(self.bias))
        return super().fold(belief, biased, caps, key=key)


# ---- broken executors ---------------------------------------------------------------


def _compiled_with(realize=None, decide=None):
    """A compiled executor whose settlement or decision stage has been broken."""
    table = handlers.HANDLERS
    if realize is not None:
        cancel = _clone(handlers._cancel_lane, _realize_rec=realize)
        table = (
            handlers.h_none,
            _clone(handlers.h_done, _realize_rec=realize),
            handlers.h_release,
            handlers.h_wait,
            _clone(handlers.h_stop, _cancel_lane=cancel),
            handlers.h_process,
            _clone(handlers.h_cancel, _cancel_lane=cancel),
            handlers.h_none,
            _clone(handlers.h_fault, _cancel_lane=cancel),
        )

    def step(carry, ev, study, truth, root):
        return _step_with_table(carry, ev, study, truth, root, table)

    advance = _clone(transaction.advance, step=step)
    swaps = {"advance": advance}
    if decide is not None:
        swaps["decide"] = decide
    moved = _clone(transaction.transition, **swaps)

    class Broken(CompiledExecutor):
        def transition(
            self, carry, i, root, study, truth, *, wake_fn=core.wake_by_audience
        ):
            return moved(carry, i, root, study, truth, wake_fn=wake_fn)

    return Broken()


def _with_draw(draw):
    """Executor whose measurement draws come from `draw(study, root, coords, ...)`."""

    def realize(c, study, truth, root, base, fraction):
        def key(root, coords, purpose, **kw):
            return draw(study, root, coords, purpose, **kw)

        return _clone(handlers._realize_rec, draw_key=key)(
            c, study, truth, root, base, fraction
        )

    return _compiled_with(realize=realize)


def draw_by_release_time_executor():
    """Measurement draws also depend on the release tick of the record."""
    return _with_draw(
        lambda study, root, coords, purpose, **kw: jr.fold_in(
            core.draw_key(root, coords, purpose, **kw), coords.available_tick
        )
    )


def branch_blind_executor():
    """Measurement draws ignore the fork's branch."""
    return _with_draw(
        lambda study, root, coords, purpose, **kw: core.draw_key(
            root, coords, purpose, **{**kw, "branch": 0}
        )
    )


def branch_everywhere_executor():
    """Measurement draws use the fork's branch for every acquisition, prefix too."""
    return _with_draw(
        lambda study, root, coords, purpose, **kw: core.draw_key(
            root, coords, purpose, **{**kw, "branch": study.fork.branch}
        )
    )


def unreleased_view_executor():
    """The policy view includes records that are written but not yet released."""

    def leaky_rows(log, audience):
        return log.valid & (log.status != RecordStatus.UNRELEASED)

    leaky_view = _clone(snapshot.snapshot_view, visible_rows=leaky_rows)
    return _compiled_with(decide=_clone(transaction.decide, snapshot_view=leaky_view))


def blind_record_key(root, rec, purpose, **kw):
    """Record key that ignores the branch and lineage of a fork."""
    return core.draw_key(root, rec, purpose, **{**kw, "branch": 0, "lineage": 0})


def blind_decision_key(root, phase, tick, ordinal, **kw):
    """Decision key that ignores the lineage of a fork."""
    return core.policy_key(root, phase, tick, ordinal, **{**kw, "lineage": 0})
