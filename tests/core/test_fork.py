"""A fork keeps the common prefix and shares later draws only as declared."""

import dataclasses

import jax.numpy as jnp
import jax.random as jr
import pytest

from spaceodyssey.core import (
    Capacities,
    Coupling,
    Fork,
    Purpose,
    battery,
    record_none,
)
from spaceodyssey.core.keys import draw_key
from tests.core.battery_helpers import (
    blind_decision_key,
    blind_record_key,
    branch_blind_executor,
    branch_everywhere_executor,
    survey,
)
from tests.core.helpers import worlds

N_STEPS = 40
# five acquisitions of subject 0 have indices 0 to 4; fork at index 2
FORK_AT = 2
SHARE_BRANCH = Coupling(
    shared=(
        (Purpose.MEASUREMENT, ("branch", "lineage")),
        (Purpose.POLICY, ("lineage",)),
    )
)


@pytest.fixture(scope="module")
def inputs():
    return worlds(2)


def run(inputs, **kw):
    truths, roots = inputs
    return battery.fork_check(survey(), truths, roots, N_STEPS, at=FORK_AT, **kw)


def test_fork(inputs):
    """Indices 0, 1 shared; 2, 3, 4 re-drawn on the branch; trunk 0, 1, 3, 4 intact."""
    result = run(inputs)
    assert result, result.diagnostic
    # two worlds, two prefix indices and three later ones
    assert result.detail["counts"] == {"prefix": 4, "after": 6}
    assert all(not seen for seen, _ in result.detail["key_report"].values())


def test_fork_shares_later_draws_only_where_declared(inputs):
    result = run(inputs, coupling=SHARE_BRANCH)
    assert result, result.diagnostic
    report = result.detail["key_report"]
    assert report["MEASUREMENT"] == (True, True)
    assert report["POLICY"] == (True, True)
    assert report["INFERENCE"] == (False, False)


def test_fork_default_leaves_unforked_keys_unchanged():
    """A study with no fork draws exactly what it drew before forks existed."""
    root = jr.PRNGKey(3)
    rec = dataclasses.replace(
        record_none(Capacities()),
        subject=jnp.asarray(1),
        acquisition_index=jnp.asarray(4),
    )
    plain = draw_key(root, rec, Purpose.MEASUREMENT)
    traced = draw_key(
        root, rec, Purpose.MEASUREMENT, branch=Fork().branch_of(rec.acquisition_index)
    )
    assert bool(jnp.array_equal(plain, traced))


def test_fork_catches_a_branch_that_draws_ignore(inputs):
    """Negative control: forked draws equal the trunk's though no coupling says so."""
    result = run(inputs, executor=branch_blind_executor())
    assert not result
    assert "separate" in result.diagnostic


def test_fork_catches_a_branch_applied_before_the_fork(inputs):
    """Negative control: the prefix is no longer shared."""
    result = run(inputs, executor=branch_everywhere_executor())
    assert not result
    assert "prefix record" in result.diagnostic


def test_fork_catches_keys_that_ignore_the_fork_per_purpose(inputs):
    """Negative control: key functions that never separate a fork."""
    result = run(inputs, record_key=blind_record_key, decision_key=blind_decision_key)
    assert not result
    assert "MEASUREMENT" in result.diagnostic and "POLICY" in result.diagnostic


def test_fork_check_needs_records_on_both_sides(inputs):
    truths, roots = inputs
    result = battery.fork_check(survey(), truths, roots, N_STEPS, at=40)
    assert not result
    assert "vacuous" in result.diagnostic
