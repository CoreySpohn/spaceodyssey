"""The observer's belief layout: refusal, the count leaf and the two belief types."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey.core import (
    Audience,
    BeliefLayout,
    MeanBelief,
    init_carry,
    project_mean,
    snapshot_fingerprint,
    snapshot_view,
)
from spaceodyssey.core.enums import F64, I64
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
