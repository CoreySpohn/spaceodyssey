"""Charge components: the variable debit, the pricing modes and the fixed surcharge.

With refunds off an admission debits the channel cost, as before. With refunds on it
debits `charge` plus the surcharge of a SURCHARGE channel, and a cancellation keeps
only the part the pricing mode says is spent. Every case below cancels halfway.
"""

import jax
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor, handlers
from spaceodyssey.core.enums import ChargePricing
from spaceodyssey.core.ledger import charge_parts
from tests.core.helpers import one_world
from tests.core.program_helpers import cancel, observe, wait
from tests.core.refund_helpers import mutate, refund_study

pytestmark = pytest.mark.medium  # A cold host run compiles the executor.

CANCELLED = (observe(0, 0, 10, 0), wait(0, 60), cancel(1, 0))
MATRIX = {
    # name: (study columns, balance after a halfway cancel, retained, fixed)
    "occupancy": (dict(charge=10.0), 95.0, 5.0, 0.0),
    "statistical": (
        dict(charge=4.0, pricing=ChargePricing.STATISTICAL),
        98.0,
        2.0,
        0.0,
    ),
    "flat": (dict(charge=4.0, pricing=ChargePricing.FLAT_FRACTION), 96.0, 4.0, 0.0),
    "surcharge": (
        dict(charge=10.0, pricing=ChargePricing.SURCHARGE, surcharge=2.0),
        93.0,
        7.0,
        2.0,
    ),
}


def _run(study):
    truth, root = one_world(caps=study.caps)
    return HostExecutor().run(study, truth, root, 80)


class _Runs(dict):
    """The runs by pricing mode, each made on first use."""

    def __missing__(self, name):
        self[name] = _run(refund_study(CANCELLED, **MATRIX[name][0]))
        return self[name]


@pytest.fixture(scope="module")
def runs():
    return _Runs()


def _balance(run):
    return float(np.asarray(run.carry.ledger.balance)[0, 0])


@pytest.mark.parametrize("name", MATRIX)
def test_a_halfway_cancel_retains_what_the_pricing_mode_spends(runs, name):
    """Occupancy and statistical keep half, flat keeps all, a surcharge stays fixed."""
    _, balance, retained, fixed = MATRIX[name]
    run = runs[name]
    assert _balance(run) == pytest.approx(balance)
    log = run.carry.log
    assert float(log.charged[0, 0]) == pytest.approx(retained)
    assert float(log.nonrefundable[0, 0]) == pytest.approx(fixed)


@pytest.mark.medium  # the refunds-off study compiles on its own: 6.7 s on a laptop
def test_without_refunds_the_cost_is_debited_and_no_column_exists():
    """The default path: charge is not read, the refund columns are absent."""
    study = refund_study(CANCELLED, charge=3.0, refunds=False, failure=None)
    run = _run(study)
    assert _balance(run) == pytest.approx(95.0)  # cost 10, half returned
    assert run.carry.log.nonrefundable is None
    assert run.carry.ledger.refund_ignored is None


def test_charge_parts_reads_the_cost_when_refunds_are_off():
    """Fails if the legacy path reads charge: the default program would change."""
    study = refund_study(charge=3.0, refunds=False, failure=None)
    charge, fixed, pricing = charge_parts(study, 0)
    np.testing.assert_allclose(charge, 10.0)
    np.testing.assert_allclose(fixed, 0.0)
    assert (np.asarray(pricing) == ChargePricing.OCCUPANCY).all()


def test_charge_parts_adds_the_surcharge_only_to_a_surcharge_channel():
    """Fails if a surcharge column counts under another pricing mode."""
    kw = dict(charge=10.0, surcharge=2.0)
    on = charge_parts(refund_study(pricing=ChargePricing.SURCHARGE, **kw), 0)
    np.testing.assert_allclose(on[0], 12.0)
    np.testing.assert_allclose(on[1], 2.0)
    off = charge_parts(refund_study(pricing=ChargePricing.OCCUPANCY, **kw), 0)
    np.testing.assert_allclose(off[0], 10.0)
    np.testing.assert_allclose(off[1], 0.0)


def test_the_default_charge_is_the_cost():
    """A study that sets no charge debits its cost with refunds on."""
    study = refund_study(CANCELLED)
    assert _balance(_run(study)) == pytest.approx(95.0)


@pytest.mark.medium  # a mutant clears the caches and recompiles: 7.2 s on a laptop
def test_mutant_a_cancel_that_credits_the_surcharge_is_caught(monkeypatch):
    """Mutant: the cancel credit ignores the fixed part, so 93 becomes 95.5 or more."""
    old = "kept = (charge - fixed) * part + fixed"
    new = "kept = (charge - fixed) * part"
    monkeypatch.setattr(
        handlers, "_cancel_lane", mutate(handlers._cancel_lane, old, new)
    )
    jax.clear_caches()
    try:
        kw = MATRIX["surcharge"][0]
        assert _balance(_run(refund_study(CANCELLED, **kw))) != pytest.approx(93.0)
    finally:
        jax.clear_caches()
