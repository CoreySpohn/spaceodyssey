"""Tests for the plan-level contracts: PlanSummary, PlanPredictive, outcomes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spaceodyssey
from spaceodyssey import contracts
from spaceodyssey.contracts import resolve_contract
from spaceodyssey.errors import OutOfRangeError
from spaceodyssey.plan import CertificationOutcome, PlanPredictive, PlanSummary


def make_plan(n_planets: int = 3, n_bins: int = 2, n_visits: int = 4, **overrides):
    """A plan summary with P planets, B bins and V visits."""
    fields = {
        "noise_sigma": jnp.full((n_planets, n_bins), 0.2),
        "bin_wavelength_nm": jnp.linspace(500.0, 900.0, n_bins),
        "noise_known": jnp.ones(n_planets, dtype=bool),
        "visit_photometric_snr": jnp.full((n_planets, n_visits), 5.0),
        "spectrum_visit": jnp.full(n_planets, 0 if n_visits else -1),
        "spectral_resolution": jnp.full((n_planets,), 70.0),
        "visit_epochs_d": jnp.tile(jnp.arange(n_visits) * 90.0, (n_planets, 1)),
        "visit_success": jnp.ones((n_planets, n_visits), dtype=bool),
        "astrometric_precision_mas": jnp.full((n_planets, n_visits), 2.0),
        "k_precision_ms": jnp.full((n_planets,), jnp.inf),
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
        assert plan.noise_sigma.shape == (3, 5)
        assert plan.visit_epochs_d.shape == (3, 2)

    def test_masks_are_boolean(self):
        """Masks given as integers are stored as booleans."""
        plan = make_plan(
            visit_success=[[1, 0, 1, 1]] * 3,
            noise_known=[1, 0, 0],
            k_precision_ms=[1.0, jnp.inf, jnp.inf],
        )
        assert plan.visit_success.dtype == jnp.bool_
        assert plan.mass_measured.dtype == jnp.bool_
        assert plan.noise_known.dtype == jnp.bool_
        np.testing.assert_array_equal(plan.mass_measured, [True, False, False])
        assert not bool(plan.visit_success[0, 1])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("noise_sigma", jnp.ones((3,))),
            ("bin_wavelength_nm", jnp.ones((3,))),
            ("noise_corr_factors", jnp.ones((3, 3, 1))),
            ("noise_corr_factors", jnp.ones((3, 2))),
            ("noise_known", jnp.ones((2,))),
            ("spectrum_visit", jnp.zeros((3, 1), dtype=int)),
            ("visit_photometric_snr", jnp.ones((2, 4))),
            ("spectral_resolution", jnp.ones((2,))),
            ("visit_epochs_d", jnp.ones((2, 4))),
            ("visit_success", jnp.ones((3, 3), dtype=bool)),
            ("astrometric_precision_mas", jnp.ones((3, 5))),
            ("k_precision_ms", jnp.ones((3, 1))),
            ("k_precision_assumed_ms", jnp.ones((3, 1))),
            ("k_precision_assumed_ms", jnp.ones(2)),
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

    @pytest.mark.parametrize("field", ["p_pos", "p_neg", "p_wrong", "p_neg_wrong"])
    @pytest.mark.parametrize("value", [-1e-7, 1.0 + 1e-7, jnp.nan, jnp.inf, -jnp.inf])
    def test_rejects_invalid_probability_values(self, field, value):
        """Every channel has exact finite bounds on in-range planets."""
        values = jnp.zeros(3).at[1].set(value)
        with pytest.raises(ValueError, match=field) as error:
            make_outcome(**{field: values})
        assert "planet" in str(error.value)
        assert "[1]" in str(error.value)

    @pytest.mark.parametrize("field", ["p_pos", "p_neg", "p_pos_se"])
    @pytest.mark.parametrize(
        "value", [jnp.ones(3, dtype=bool), jnp.full(3, 0.1 + 0.1j)]
    )
    def test_rejects_non_real_probability_dtypes(self, field, value):
        """Boolean or complex channels cannot satisfy the real-valued contract."""
        with pytest.raises(ValueError, match=field):
            make_outcome(**{field: value})

    @pytest.mark.parametrize(
        ("dtype", "p_pos"),
        [(jnp.bfloat16, 0.50390625), (jnp.float16, 0.50048828125)],
    )
    def test_rejects_overlap_hidden_by_low_precision_rounding(self, dtype, p_pos):
        """The relations are checked in at least float32, not the channel dtype."""
        with pytest.raises(ValueError, match=r"p_pos.*p_neg"):
            make_outcome(
                p_pos=jnp.full(3, p_pos, dtype=dtype),
                p_neg=jnp.full(3, 0.5, dtype=dtype),
            )

    def test_rejects_overlapping_verdicts(self):
        """Positive and negative verdicts cannot overlap beyond the slack."""
        with pytest.raises(ValueError, match=r"p_pos.*p_neg") as error:
            make_outcome(p_neg=jnp.array([0.1, 0.5, 0.5]))
        assert "planet" in str(error.value)
        assert "[1, 2]" in str(error.value)

    def test_rejects_wrong_positive_exceeding_positive(self):
        """Wrong positive certifications are a subset of positive ones."""
        with pytest.raises(ValueError, match=r"p_wrong.*p_pos") as error:
            make_outcome(p_wrong=jnp.array([0.0, 0.7, 0.0]))
        assert "planet" in str(error.value)
        assert "[1]" in str(error.value)

    def test_rejects_wrong_negative_exceeding_negative(self):
        """Wrong negative certifications are a subset of negative ones."""
        with pytest.raises(ValueError, match=r"p_neg_wrong.*p_neg") as error:
            make_outcome(p_neg_wrong=jnp.array([0.0, 0.2, 0.0]))
        assert "planet" in str(error.value)
        assert "[1]" in str(error.value)

    @pytest.mark.parametrize(
        "field", ["p_pos_se", "p_neg_se", "p_wrong_se", "p_neg_wrong_se"]
    )
    @pytest.mark.parametrize("value", [-1e-7, jnp.nan, -jnp.inf])
    def test_rejects_invalid_standard_errors(self, field, value):
        """Standard errors must be nonnegative, excluding NaN."""
        values = jnp.zeros(3).at[1].set(value)
        with pytest.raises(ValueError, match=field) as error:
            make_outcome(**{field: values})
        assert "planet" in str(error.value)
        assert "[1]" in str(error.value)

    @pytest.mark.parametrize(
        "changes",
        [
            {"p_pos": 0.6, "p_neg": 0.4 + 1e-7},
            {"p_wrong": 0.6 + 1e-7},
            {"p_neg_wrong": 0.1 + 1e-7},
        ],
    )
    def test_accepts_probability_relations_inside_slack(self, changes):
        """Each probability relation permits rounding within 1e-6."""
        outcome = make_outcome(
            **{name: jnp.full(3, value) for name, value in changes.items()}
        )
        for name, value in changes.items():
            np.testing.assert_array_equal(getattr(outcome, name), jnp.full(3, value))

    @pytest.mark.parametrize("positive", [0.0, 1.0])
    def test_accepts_probability_endpoints(self, positive):
        """Exact probability endpoints and subset equality are valid."""
        make_outcome(
            p_pos=jnp.full(3, positive),
            p_neg=jnp.full(3, 1 - positive),
            p_wrong=jnp.full(3, positive),
            p_neg_wrong=jnp.full(3, 1 - positive),
        )

    def test_accepts_positive_infinite_standard_errors(self):
        """The standard error contract requires nonnegativity, not finiteness."""
        make_outcome(
            **{
                name: jnp.full(3, jnp.inf)
                for name in ("p_pos_se", "p_neg_se", "p_wrong_se", "p_neg_wrong_se")
            }
        )

    @pytest.mark.parametrize("value", [-1.0, 2.0, jnp.nan, jnp.inf, 0.9])
    def test_accepts_invalid_out_of_range_values(self, value):
        """Untrusted planets are excluded from every main channel value check."""
        fields = {
            name: jnp.array([0.0, value, 0.0])
            for name in (
                "p_pos",
                "p_neg",
                "p_wrong",
                "p_neg_wrong",
                "p_pos_se",
                "p_neg_se",
                "p_wrong_se",
                "p_neg_wrong_se",
            )
        }
        outcome = make_outcome(in_range=[True, False, True], **fields)
        np.testing.assert_array_equal(outcome.p_pos, fields["p_pos"])

    def test_constructs_from_traced_arrays_under_jit(self):
        """Valid outcome construction accepts traced probabilities and flags."""

        @jax.jit
        def build(positive, in_range):
            return make_outcome(
                p_pos=positive,
                p_neg=1 - positive,
                p_wrong=positive / 2,
                p_neg_wrong=(1 - positive) / 2,
                p_pos_se=positive / 10,
                p_neg_se=positive / 10,
                p_wrong_se=positive / 10,
                p_neg_wrong_se=positive / 10,
                in_range=in_range,
            )

        positive = jnp.array([0.0, 0.6, 1.0])
        outcome = build(positive, jnp.ones(3, dtype=bool))
        np.testing.assert_array_equal(outcome.p_pos, positive)
        np.testing.assert_allclose(outcome.p_neg_wrong, (1 - positive) / 2)

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


@pytest.mark.parametrize("value", [0.0, -1.0, jnp.inf, jnp.nan])
def test_invalid_noise_sigma(value):
    with pytest.raises(ValueError, match="noise_sigma must be finite and > 0"):
        make_plan(noise_sigma=jnp.full((3, 2), value))


@pytest.mark.parametrize("value", [0.0, -1.0, -jnp.inf, jnp.nan])
def test_invalid_k_precision(value):
    with pytest.raises(ValueError, match="k_precision_ms must be > 0"):
        make_plan(k_precision_ms=jnp.full(3, value))


@pytest.mark.parametrize("value", [-2, 4, 1.5])
def test_invalid_spectrum_visit(value):
    with pytest.raises(ValueError, match="spectrum_visit"):
        make_plan(spectrum_visit=jnp.full(3, value))


@pytest.mark.parametrize("value", [-1.0, jnp.nan])
def test_invalid_photometric_snr(value):
    with pytest.raises(ValueError, match="visit_photometric_snr"):
        make_plan(visit_photometric_snr=jnp.full((3, 4), value))


def test_nonfinite_correlated_noise():
    with pytest.raises(ValueError, match="noise_corr_factors must be finite"):
        make_plan(noise_corr_factors=jnp.full((3, 2, 1), jnp.nan))


def test_version_sentinels_and_pytree():
    from spaceodyssey.plan import CONTRACT_VERSION

    plan = make_plan(spectrum_visit=[-1, 0, 3])
    assert CONTRACT_VERSION == PlanSummary.CONTRACT_VERSION == 3
    assert not hasattr(plan, "snr")
    assert not bool(jnp.any(plan.mass_measured))
    leaves, structure = jax.tree_util.tree_flatten(plan)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    np.testing.assert_array_equal(restored.spectrum_visit, [-1, 0, 3])
    assert restored.noise_corr_factors is None
    assert restored.k_precision_assumed_ms is None


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_per_planet_wavelengths_and_correlation_rank(rank):
    plan = make_plan(
        bin_wavelength_nm=jnp.full((3, 2), 550.0),
        noise_corr_factors=jnp.zeros((3, 2, rank)),
    )
    assert plan.noise_corr_factors.shape == (3, 2, rank)


def test_empty_visits_require_sentinel():
    assert make_plan(n_visits=0).visit_success.shape == (3, 0)
    with pytest.raises(ValueError, match="spectrum_visit"):
        make_plan(n_visits=0, spectrum_visit=jnp.zeros(3, dtype=int))


def test_construction_under_jit():
    plan = jax.jit(lambda noise: make_plan(noise_sigma=noise))(jnp.ones((3, 2)))
    np.testing.assert_array_equal(plan.noise_sigma, np.ones((3, 2)))


@pytest.mark.parametrize(
    "assumed", [None, [0.1, jnp.inf, 0.5], [0.2, jnp.inf, 1.0], [0.4, jnp.inf, 2.0]]
)
def test_k_precisions_round_trip(assumed):
    """Both precisions survive construction, pytree round trips and jitted identity.

    Reported precision may be smaller, equal, or larger than sampling precision.
    """
    true = [0.2, jnp.inf, 1.0]
    plan = make_plan(k_precision_ms=true, k_precision_assumed_ms=assumed)
    np.testing.assert_array_equal(plan.k_precision_ms, jnp.asarray(true))
    np.testing.assert_array_equal(plan.mass_measured, [True, False, True])
    reported = true if assumed is None else assumed
    np.testing.assert_array_equal(
        plan.k_precision_for_certification_ms, jnp.asarray(reported)
    )
    if assumed is None:
        assert plan.k_precision_assumed_ms is None
    else:
        np.testing.assert_array_equal(plan.k_precision_assumed_ms, jnp.asarray(assumed))
    leaves, structure = jax.tree_util.tree_flatten(plan)
    restored = jax.tree_util.tree_unflatten(structure, leaves)
    for result in (restored, jax.jit(lambda value: value)(plan)):
        result_leaves, result_structure = jax.tree_util.tree_flatten(result)
        assert result_structure == structure
        for expected, actual in zip(leaves, result_leaves, strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert (result.k_precision_assumed_ms is None) == (assumed is None)


@pytest.mark.parametrize("value", [0.0, -1.0, -jnp.inf, jnp.nan])
def test_invalid_k_precision_assumed(value):
    """Reported precision rejects nonpositive and NaN values."""
    with pytest.raises(ValueError, match="k_precision_assumed_ms must be > 0"):
        make_plan(k_precision_ms=jnp.ones(3), k_precision_assumed_ms=jnp.full(3, value))


@pytest.mark.parametrize("assumed", [[0.1, 0.2, 0.3], [jnp.inf, jnp.inf, 0.3]])
def test_k_precision_inf_pattern_mismatch(assumed):
    """Missing and spurious infinite reported precisions are both rejected."""
    with pytest.raises(ValueError, match="inf exactly where"):
        make_plan(k_precision_ms=[0.2, jnp.inf, 1.0], k_precision_assumed_ms=assumed)


@pytest.mark.parametrize("field", ["k_precision_ms", "k_precision_assumed_ms"])
def test_k_precision_shape_checked_under_jit(field):
    """Tracing retains static shape validation for both precision fields."""
    with pytest.raises(ValueError, match=field + " must have shape"):
        jax.jit(lambda value: make_plan(**{field: value}))(jnp.ones((3, 1)))


def test_k_precisions_construct_under_jit():
    """Both precision leaves can be supplied during traced construction."""
    true = jnp.array([0.2, jnp.inf, 1.0])
    assumed = jnp.array([0.4, jnp.inf, 0.5])
    plan = jax.jit(
        lambda k, reported: make_plan(k_precision_ms=k, k_precision_assumed_ms=reported)
    )(true, assumed)
    np.testing.assert_array_equal(plan.k_precision_ms, true)
    np.testing.assert_array_equal(plan.k_precision_assumed_ms, assumed)


def test_version_2_consumer_refuses_version_3():
    """A version-pinned consumer fails closed after the contract upgrade."""

    def require_pinned_version(pinned):
        if PlanSummary.CONTRACT_VERSION != pinned:
            raise ValueError(f"Unsupported plan contract; expected {pinned}")

    with pytest.raises(ValueError, match="expected 2"):
        require_pinned_version(2)
    require_pinned_version(3)


def test_old_precision_name_refused():
    """The retired field is neither a constructor argument nor a property."""
    assert not hasattr(make_plan(), "rv_precision_ms")
    with pytest.raises(TypeError, match="rv_precision_ms"):
        make_plan(rv_precision_ms=jnp.ones(3))


@pytest.mark.parametrize("field", ["k_precision_ms", "k_precision_assumed_ms"])
@pytest.mark.parametrize("traced", [False, True], ids=["concrete", "jit"])
def test_complex_k_precision_rejected(field, traced):
    """Complex precision dtypes are invalid on the host and during tracing."""

    def construct(value):
        fields = {"k_precision_ms": jnp.ones(3), field: value}
        return make_plan(**fields)

    with pytest.raises(ValueError, match=field + " must have a real dtype"):
        (jax.jit(construct) if traced else construct)(jnp.full(3, 1 + 2j))
