"""Tests for the named-resource ledger and the reference cost model."""

import jax.numpy as jnp
import pytest
from planit_py.vocabulary import (
    Action,
    FixedDuration,
    Mode,
    ObservingContext,
    StopOnCertificate,
)

from spaceodyssey.account import ConstantOverheads, Ledger
from spaceodyssey.registry import REGISTRY


def make_ctx() -> ObservingContext:
    """A context with two targets and distinct slew costs."""
    return ObservingContext(
        time_d=0.0,
        target_ids=jnp.asarray([1, 2]),
        observable=jnp.asarray([True, True]),
        slew_d=jnp.asarray([0.1, 0.4]),
    )


def make_action(integration) -> Action:
    """An action on target 2 with the given integration policy."""
    return Action(
        target_id=2,
        mode=Mode.IMAGING,
        bands=("500nm",),
        integration=integration,
        start_time_d=0.0,
    )


class TestLedger:
    """Functional charging against declared budgets."""

    def test_charge_is_functional(self):
        """Charging returns a new ledger and leaves the original alone."""
        ledger = Ledger.from_budgets((("time_d", 10.0),))
        charged = ledger.charge({"time_d": 4.0})
        assert ledger.spent["time_d"] == 0.0
        assert charged.spent["time_d"] == 4.0
        assert charged.remaining("time_d") == 6.0

    def test_exhaustion_flips_at_cap(self):
        """The ledger reports exhaustion exactly when a cap is reached."""
        ledger = Ledger.from_budgets((("time_d", 10.0),))
        assert not ledger.charge({"time_d": 9.9}).exhausted
        assert ledger.charge({"time_d": 10.0}).exhausted

    def test_undeclared_resource_rejected(self):
        """Spending an unbudgeted resource is a loud error."""
        ledger = Ledger.from_budgets((("time_d", 10.0),))
        with pytest.raises(KeyError, match="fuel_kg"):
            ledger.charge({"fuel_kg": 1.0})

    def test_multiple_resources(self):
        """Named resources beyond time are ordinary ledger keys."""
        ledger = Ledger.from_budgets((("time_d", 10.0), ("fuel_kg", 2.0)))
        charged = ledger.charge({"time_d": 1.0, "fuel_kg": 2.0})
        assert charged.exhausted
        assert charged.remaining("time_d") == 9.0


class TestConstantOverheads:
    """Expected and realized pricing of one shared cost model."""

    def test_expected_bounds_fixed_duration(self):
        """A fixed-duration action prices integration + overhead + slew."""
        cost = ConstantOverheads(overhead_d=0.5)
        action = make_action(FixedDuration(duration_d=2.0))
        assert cost.expected(action, make_ctx()) == {
            "time_d": pytest.approx(2.0 + 0.5 + 0.4)
        }

    def test_expected_bounds_adaptive_by_max(self):
        """An adaptive integration is priced by its maximum duration."""
        cost = ConstantOverheads(overhead_d=0.5)
        action = make_action(StopOnCertificate(threshold=7.0, max_duration_d=14.0))
        assert cost.expected(action, make_ctx()) == {
            "time_d": pytest.approx(14.0 + 0.5 + 0.4)
        }

    def test_realized_uses_reported_duration(self):
        """Realized pricing differs from expected only in the duration term."""
        cost = ConstantOverheads(overhead_d=0.5)
        action = make_action(StopOnCertificate(threshold=7.0, max_duration_d=14.0))
        assert cost.realized(action, make_ctx(), duration_d=3.0) == {
            "time_d": pytest.approx(3.0 + 0.5 + 0.4)
        }

    def test_unknown_target_rejected(self):
        """Pricing a target missing from the context is a loud error."""
        cost = ConstantOverheads()
        action = Action(
            target_id=99,
            mode=Mode.IMAGING,
            bands=(),
            integration=FixedDuration(duration_d=1.0),
            start_time_d=0.0,
        )
        with pytest.raises(KeyError, match="99"):
            cost.expected(action, make_ctx())

    def test_registered_by_default(self):
        """The reference cost model is discoverable in the registry."""
        assert REGISTRY.get("cost", "constant_overheads").cls is ConstantOverheads
