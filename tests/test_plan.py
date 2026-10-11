"""Tests for the plan-level contracts: PlanSummary, PlanPredictive, outcomes."""

import jax.numpy as jnp
import pytest

import spaceodyssey
from spaceodyssey import contracts
from spaceodyssey.contracts import resolve_contract
from spaceodyssey.errors import OutOfRangeError
from spaceodyssey.plan import CertificationOutcome, PlanPredictive, PlanSummary


def make_plan(n_planets: int = 3, n_bins: int = 2, n_visits: int = 4, **overrides):
    """A plan summary with P planets, B bins and V visits."""
    fields = {
        "snr": jnp.full((n_planets, n_bins), 5.0),
        "spectral_resolution": jnp.full((n_planets,), 70.0),
        "visit_epochs_d": jnp.tile(jnp.arange(n_visits) * 90.0, (n_planets, 1)),
        "visit_success": jnp.ones((n_planets, n_visits), dtype=bool),
        "astrometric_precision_mas": jnp.full((n_planets, n_visits), 2.0),
        "mass_measured": jnp.zeros((n_planets,), dtype=bool),
    }
    fields.update(overrides)
    return PlanSummary(**fields)


def make_outcome(n_planets: int = 3, **overrides):
    """A valid outcome with every planet in range."""
    zeros = jnp.zeros((n_planets,))
    fields = {
        "p_pos": jnp.full((n_planets,), 0.6),
        "p_neg": jnp.full((n_planets,), 0.1),
        "p_wrong": zeros,
        "p_neg_wrong": zeros,
        "p_pos_se": zeros,
        "p_neg_se": zeros,
        "p_wrong_se": zeros,
        "p_neg_wrong_se": zeros,
        "truth_positive": jnp.ones((n_planets,), dtype=bool),
        "in_range": jnp.ones((n_planets,), dtype=bool),
    }
    fields.update(overrides)
    return CertificationOutcome(**fields)


class TestPlanSummary:
    """One canonical planet axis across every field."""

    def test_builds_with_consistent_shapes(self):
        """A consistent plan exposes its batch size."""
        plan = make_plan(n_planets=3, n_bins=5, n_visits=2)
        assert plan.n_planets == 3
        assert plan.snr.shape == (3, 5)
        assert plan.visit_epochs_d.shape == (3, 2)

    def test_masks_are_boolean(self):
        """Masks given as integers are stored as booleans."""
        plan = make_plan(
            visit_success=[[1, 0, 1, 1]] * 3,
            mass_measured=[1, 0, 0],
        )
        assert plan.visit_success.dtype == jnp.bool_
        assert plan.mass_measured.dtype == jnp.bool_
        assert not bool(plan.visit_success[0, 1])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("snr", jnp.ones((3,))),
            ("spectral_resolution", jnp.ones((2,))),
            ("visit_epochs_d", jnp.ones((2, 4))),
            ("visit_success", jnp.ones((3, 3), dtype=bool)),
            ("astrometric_precision_mas", jnp.ones((3, 5))),
            ("mass_measured", jnp.ones((3, 1), dtype=bool)),
        ],
    )
    def test_inconsistent_shape_rejected(self, field, value):
        """Any field off the planet or visit axis is rejected at construction."""
        with pytest.raises(ValueError, match=field.split("_")[0]):
            make_plan(**{field: value})


class TestPlanPredictive:
    """The truth pytree must share the plan's planet axis."""

    def test_accepts_matching_pytree(self):
        """A dict of per-planet arrays is a valid truth."""
        truth = {"theta": jnp.zeros(3), "radius_rearth": jnp.ones(3)}
        predictive = PlanPredictive(plan=make_plan(), truth=truth)
        assert predictive.truth["theta"].shape == (3,)

    def test_rejects_wrong_planet_axis(self):
        """A truth leaf with a different leading length is rejected."""
        with pytest.raises(ValueError, match="planet axis"):
            PlanPredictive(plan=make_plan(), truth=jnp.zeros(4))

    def test_rejects_scalar_truth(self):
        """A scalar truth would broadcast silently, so it is rejected."""
        with pytest.raises(ValueError, match="planet axis"):
            PlanPredictive(plan=make_plan(), truth=jnp.asarray(1.0))


class TestCertificationOutcome:
    """Shapes, flags, and the host-side range check."""

    def test_rejects_mismatched_shapes(self):
        """Every field must be one value per planet."""
        with pytest.raises(ValueError, match="p_wrong"):
            make_outcome(p_wrong=jnp.zeros(4))

    def test_rejects_batched_probabilities(self):
        """The outcome carries exactly one axis."""
        with pytest.raises(ValueError, match="p_pos"):
            make_outcome(p_pos=jnp.zeros((3, 1)))

    def test_in_range_check_passes_when_all_in_range(self):
        """No planet out of range, no exception."""
        make_outcome().raise_if_out_of_range()

    def test_in_range_check_names_offending_planets(self):
        """Out-of-range planets raise on the host with their indices."""
        outcome = make_outcome(in_range=jnp.array([True, False, False]))
        with pytest.raises(OutOfRangeError, match=r"\[1, 2\]"):
            outcome.raise_if_out_of_range()


class TestContractRegistration:
    """The new contracts resolve through the contract namespace."""

    @pytest.mark.parametrize("cls", [PlanSummary, PlanPredictive, CertificationOutcome])
    def test_resolves_by_public_name(self, cls):
        """'spaceodyssey.<Name>' resolves to the class and contracts re-exports it."""
        assert resolve_contract(f"spaceodyssey.{cls.__name__}") is cls
        assert getattr(contracts, cls.__name__) is cls
        assert cls.__name__ in contracts.__all__
        assert getattr(spaceodyssey, cls.__name__) is cls
