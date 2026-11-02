"""Reserved fields exist everywhere a value travels and no rule reads them yet."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey import core
from spaceodyssey.core.enums import (
    AdmissionCode,
    ChargeRule,
    RemainderRule,
)
from spaceodyssey.core.executors.base import (
    canonical_history,
    histories_equal,
    index_world,
)
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy as kinds
from tests.core.helpers import CAPS, worlds

N_STEPS = 40


@pytest.fixture(scope="module")
def runs():
    study = kinds.survey_1()
    truths, roots = worlds(1)
    cs, ts = CompiledExecutor().run(study, truths, roots, N_STEPS)
    scan = (index_world(cs, 0), index_world(ts, 0))
    host = HostExecutor().run(study, index_world(truths, 0), roots[0], N_STEPS)
    return study, scan, tuple(host)


def histories(runs):
    _, scan, host = runs
    return canonical_history(*scan), canonical_history(*host)


# carry-side field: (history key, expected value as a function of the capacities)
LEDGER_FIELDS = {
    "Ledger.program_charge_rule": (
        "program_charge_rule",
        lambda c: np.full(c.n_programs, ChargeRule.RECEIPT),
    ),
    "Ledger.valid_from_tick": ("valid_from_tick", lambda c: np.zeros(c.n_balance_rows)),
    "Ledger.valid_until_tick": (
        "valid_until_tick",
        lambda c: np.full(c.n_balance_rows, c.max_tick),
    ),
    "Ledger.remainder_rule": (
        "remainder_rule",
        lambda c: np.full(c.n_balance_rows, RemainderRule.NONE),
    ),
    "Ledger.disruptive_used": ("disruptive_used", lambda c: np.zeros(c.n_assets)),
}


@pytest.mark.parametrize("name", LEDGER_FIELDS)
def test_reserved_ledger_field_roundtrips_through_both_executors(runs, name):
    key, expected = LEDGER_FIELDS[name]
    scan, host = histories(runs)
    want = expected(CAPS)
    assert np.array_equal(scan.ints[key], want)
    assert np.array_equal(host.ints[key], want)
    assert histories_equal(scan, host).ok


def test_reserved_record_refund_of_roundtrips(runs):
    scan, host = histories(runs)
    for h in (scan, host):
        assert h.n_records == 5
        assert np.array_equal(h.ints["refund_of"], np.full(5, -1))


def test_reserved_record_charged_equals_cost_in_both_executors(runs):
    scan, host = histories(runs)
    for h in (scan, host):
        assert np.array_equal(h.floats["charged"], h.floats["cost"])
        assert h.floats["charged"].shape == (5, CAPS.n_resources)


def test_reserved_channel_charge_equals_cost_and_survives_a_run(runs):
    study, _, _ = runs
    assert np.array_equal(study.channels.charge, study.channels.cost)


def test_reserved_asset_fields_default_to_none(runs):
    study, _, _ = runs
    assert np.array_equal(study.assets.freeze_ticks, np.zeros(CAPS.n_assets))
    assert np.array_equal(study.assets.disruptive_resource, -np.ones(CAPS.n_assets))


def test_reserved_effect_disruptive_roundtrips(runs):
    scan, host = histories(runs)
    for h in (scan, host):
        assert h.ints["eff_disruptive"].shape[1] == CAPS.n_effects
        assert not h.ints["eff_disruptive"].any()


def test_reserved_admission_codes_are_declared():
    names = {c.name for c in AdmissionCode}
    assert {"OUTSIDE_VALIDITY", "FROZEN_HORIZON", "FORBIDDEN", "UNSUPPORTED"} <= names
    assert len({int(c) for c in AdmissionCode}) == len(AdmissionCode)


def perturbed(study, name):
    """The study with one reserved table field set far from its default."""
    if name == "ChannelTable.charge":
        return eqx.tree_at(
            lambda s: s.channels.charge, study, study.channels.charge * 7
        )
    if name == "AssetTable.freeze_ticks":
        return eqx.tree_at(
            lambda s: s.assets.freeze_ticks, study, jnp.full(CAPS.n_assets, 500)
        )
    return eqx.tree_at(
        lambda s: s.assets.disruptive_resource, study, jnp.full(CAPS.n_assets, 1)
    )


@pytest.mark.parametrize(
    "name",
    [
        "ChannelTable.charge",
        "AssetTable.freeze_ticks",
        "AssetTable.disruptive_resource",
    ],
)
def test_reserved_table_field_has_no_behaviour(runs, name):
    study, scan, _ = runs
    truths, roots = worlds(1)
    cs, ts = CompiledExecutor().run(perturbed(study, name), truths, roots, N_STEPS)
    changed = canonical_history(index_world(cs, 0), index_world(ts, 0))
    baseline = canonical_history(*scan)
    assert histories_equal(baseline, changed).ok
    assert changed.events == baseline.events


def test_effect_table_carries_the_reserved_flag():
    e = core.effects_none(CAPS)
    assert e.disruptive.dtype == jnp.bool_ and not bool(e.disruptive.any())
    assert dataclasses.replace(e, disruptive=e.disruptive.at[0].set(True)).disruptive[0]
