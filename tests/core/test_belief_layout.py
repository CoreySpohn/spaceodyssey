"""The observer's belief layout: refusal, the count leaf and the two belief types."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey.core import (
    Audience,
    BeliefLayout,
    MeanBelief,
    RecordStatus,
    canonical_history,
    empty_trace,
    histories_equal,
    init_carry,
    project_mean,
    snapshot_fingerprint,
    snapshot_view,
)
from spaceodyssey.core.conformance.folds import CountingObserver, plain
from spaceodyssey.core.enums import F64, I64
from spaceodyssey.core.records import record_none
from spaceodyssey.studies import toy
from tests.core.helpers import one_world


class TwoCountBelief(eqx.Module):
    """A belief with two integer leaves, so a test can tell which one is the count."""

    mean: jax.Array
    n: jax.Array
    m: jax.Array
    aux: jax.Array


class MomentSummary(eqx.Module):
    """A policy summary of a different type from the full belief."""

    mean: jax.Array
    n: jax.Array


def _shape(n=4, n_an=4):
    return MeanBelief(
        mean=jax.ShapeDtypeStruct((n,), F64),
        n=jax.ShapeDtypeStruct((n,), I64),
        aux=jax.ShapeDtypeStruct((n_an,), F64),
    )


@pytest.mark.parametrize(
    "count_leaf, summary",
    [
        (None, _shape()),
        ("count", _shape()),  # names no leaf
        ("mean", _shape()),  # a float leaf
        ("n", _shape(n=3)),  # the two types disagree on its shape
    ],
)
def test_a_layout_without_a_usable_count_leaf_is_refused(count_leaf, summary):
    """Fails if a layout is built whose count leaf the fingerprint cannot read."""
    with pytest.raises(ValueError, match="count leaf"):
        BeliefLayout(
            belief=_shape(),
            summary=summary,
            count_leaf=count_leaf,
            project=project_mean,
        )


def test_the_fingerprint_reads_the_declared_count_leaf():
    """Fails if the fingerprint reads `n` by name instead of the layout's count leaf."""
    study = toy.survey_1()
    truth, _ = one_world()
    view = snapshot_view(init_carry(study, truth), Audience.POLICY, study.caps)
    z = jnp.zeros(study.caps.n_subjects, I64)
    belief = TwoCountBelief(view.belief.mean, z, z.at[1].set(3), view.belief.aux)
    by_n, by_m = (
        BeliefLayout(belief=belief, summary=belief, count_leaf=c, project=project_mean)
        for c in ("n", "m")
    )
    view = eqx.tree_at(lambda v: v.belief, view, belief)
    fp_n = int(snapshot_fingerprint(view, by_n))
    assert int(snapshot_fingerprint(view, by_m)) != fp_n
    # a change to a leaf that is not the count leaf leaves the identity alone
    other = eqx.tree_at(lambda v: v.belief.m, view, z)
    assert int(snapshot_fingerprint(other, by_n)) == fp_n


class MomentObserver(toy.ToyRunningMean):
    """Full `MeanBelief`, `MomentSummary` for the policy."""

    def init_summary(self, caps):
        n = caps.n_subjects
        return MomentSummary(jnp.zeros(n, F64), jnp.zeros(n, I64))

    def fold_summary(self, summary, rec, caps):
        aux = jnp.zeros(caps.n_an, F64)
        full = self.fold(MeanBelief(summary.mean, summary.n, aux), rec, caps)
        return MomentSummary(full.mean, full.n)

    def layout(self, caps):
        full = jax.eval_shape(lambda: self.init(caps))
        summary = jax.eval_shape(lambda: self.init_summary(caps))
        return BeliefLayout(
            belief=full,
            summary=summary,
            count_leaf="n",
            project=project_mean,
            project_summary=lambda s: (s.mean, jnp.zeros(caps.n_an, F64)),
        )


class UndeclaredObserver(MomentObserver):
    """Initializes a `MomentSummary` but declares the `MeanBelief` layout."""

    def layout(self, caps):
        return toy.ToyRunningMean().layout(caps)


def test_the_carry_holds_the_declared_types_and_a_mismatch_is_refused():
    """Fails if `pbelief` is not the summary type, or an undeclared one is accepted."""
    truth, _ = one_world()
    carry = init_carry(toy.survey_1(observer=MomentObserver()), truth)
    assert isinstance(carry.belief, MeanBelief)
    assert isinstance(carry.pbelief, MomentSummary)
    with pytest.raises(ValueError, match="does not describe the summary"):
        toy.survey_1(observer=UndeclaredObserver())


class SpareBelief(eqx.Module):
    """A belief with an extra integer leaf and a leaf of zero capacity."""

    mean: jax.Array
    n: jax.Array
    m: jax.Array
    aux: jax.Array
    spare: jax.Array


def _two_step_trace(caps):
    row = dataclasses.replace(empty_trace(caps), handled=jnp.asarray(True))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), row, row)


def test_the_comparator_reads_the_layout_leaf_names():
    """Fails if the history names belief leaves itself, or keeps a zero-size leaf."""
    study = toy.survey_1()
    truth, _ = one_world()
    carry = init_carry(study, truth)
    b = carry.belief
    belief = SpareBelief(b.mean, b.n, b.n + 1, b.aux, jnp.zeros((4, 0), F64))
    carry = eqx.tree_at(lambda c: c.belief, carry, belief)
    layout = BeliefLayout(
        belief=belief, summary=carry.pbelief, count_leaf="n", project=project_mean
    )
    trace = _two_step_trace(study.caps)
    h = canonical_history(carry, trace, layout)
    assert {"carry.belief.m", "carry.pbelief.n"} <= set(h.ints)
    assert "carry.belief.spare" not in h.floats and "carry.pbelief.m" not in h.ints
    moved = eqx.tree_at(lambda c: c.belief.m, carry, b.n + 2)
    # the run has no records, so read the integer difference, not the verdict
    assert histories_equal(h, canonical_history(moved, trace, layout)).max_int_diff == 1


def test_the_fold_counter_wraps_any_declared_layout():
    """Fails if the conformance fold counter assumes `MeanBelief` for either type."""
    counted = CountingObserver(inner=MomentObserver())
    study = toy.survey_1(observer=counted)  # refused unless the layout matches init
    truth, _ = one_world()
    carry = init_carry(study, truth)
    assert isinstance(plain(carry.pbelief), MomentSummary)
    rec = dataclasses.replace(
        record_none(study.caps),
        kind=jnp.asarray(toy.RECORD_MEASUREMENT, I64),
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
        payload=jnp.ones(study.caps.payload_width, F64),
        valid=jnp.asarray(True),
    )
    summary = counted.fold_summary(carry.pbelief, rec, study.caps)
    layout = counted.layout(study.caps)
    assert int(summary.counts.sum()) == 1
    assert summary.mean[0] == 1.0  # a reader of the summary sees the inner fields
    assert int(layout.count(summary).sum()) == 1
    assert layout.summary_projection(summary)[0][0] == 1.0
