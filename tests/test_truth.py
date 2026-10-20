"""Truth-model validation, compatibility, and Gaussian sampling contracts."""

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from photomancy.priors import Uniform

import spaceodyssey
from spaceodyssey import Claim, Edge, Profile, Purpose, stream
from spaceodyssey.errors import UnsupportedProfile
from spaceodyssey.testing import LinearGaussianTruthModel
from spaceodyssey.truth import (
    AbstractTruthModel,
    ObservedData,
    PriorSet,
    Truth,
    check_compatible,
)
from tests.test_plan import make_plan


def make_truth(n=3, **kwargs):
    fields = dict(
        theta=jnp.tile(jnp.array([2.0, 3.0]), (n, 1)),
        context={"offset": jnp.full(n, 0.5)},
        model_id=LinearGaussianTruthModel.model_id,
    )
    fields.update(kwargs)
    return Truth(**fields)


def make_profile(**kwargs):
    fields = dict(
        name="test",
        claims=(Claim("x_positive", (Edge("x", 0.0, "above"),), "positive"),),
        alpha=0.05,
        support_rule="joint",
        prior_set_id="uniform@1",
        evidence_rule="every_claim",
    )
    fields.update(kwargs)
    return Profile(**fields)


@pytest.mark.parametrize("method", ["expected_data", "simulate", "quantities"])
@pytest.mark.parametrize("cause", ["D", "context"])
def test_wrong_d_or_missing_context_raises(method, cause):
    model = LinearGaussianTruthModel()
    truth = make_truth(
        **({"theta": jnp.ones((3, 3))} if cause == "D" else {"context": {}})
    )
    with pytest.raises(ValueError, match=cause):
        if method == "quantities":
            model.quantities(truth.theta, truth.context)
        elif method == "simulate":
            model.simulate(stream(0, 1, 0, Purpose.MEASUREMENT), truth, make_plan())
        else:
            model.expected_data(truth, make_plan())


@pytest.mark.parametrize(
    ("kwargs", "cause"),
    [
        ({"theta": jnp.ones(3)}, "theta"),
        ({"context": {"offset": jnp.ones(2)}}, "offset"),
        ({"context": {"": jnp.ones(3)}}, "context keys"),
    ],
)
def test_truth_shape_and_context_keys(kwargs, cause):
    with pytest.raises(ValueError, match=cause):
        make_truth(**kwargs)


@pytest.mark.parametrize("cause", ["quantity", "prior", "flux"])
def test_unsupported_profile_names_cause(cause):
    model, profile, plan = LinearGaussianTruthModel(), make_profile(), make_plan()
    if cause == "quantity":
        profile = dataclasses.replace(
            profile,
            claims=(Claim("missing", (Edge("missing_q", 0, "above"),), "either"),),
        )
        message = "missing quantities.*missing_q"
    elif cause == "prior":
        profile = dataclasses.replace(profile, prior_set_id="unknown@1")
        message = "unknown prior set.*unknown@1"
    else:
        plan = dataclasses.replace(plan, flux_reference="different_flux")
        message = "flux reference.*different_flux.*unit_flux"
    with pytest.raises(UnsupportedProfile, match=message):
        check_compatible(profile, model, plan)


@pytest.mark.parametrize("reference", ["", "unit_flux"])
def test_compatible_and_optional_flux_metadata(reference):
    model = LinearGaussianTruthModel()
    plan = make_plan(flux_reference=reference)
    check_compatible(make_profile(), model)
    check_compatible(make_profile(), model, plan)
    assert plan.CONTRACT_VERSION == 3
    assert reference not in [
        x for x in jax.tree_util.tree_leaves(plan) if isinstance(x, str)
    ]
    assert model.prior_set("uniform@1") is model.prior_sets[0]


def flatten_data(data):
    return np.concatenate(
        [
            np.asarray(data.spectrum).reshape(-1),
            np.asarray(data.photometry).reshape(-1),
            np.asarray(data.astrometry_mas).reshape(-1),
            np.asarray(data.rv_semi_amplitude_ms).reshape(-1),
        ]
    )


def test_20000_simulated_draws_match_expected_data_and_plan_covariance_within_5_se():
    n = 20_000
    model = LinearGaussianTruthModel(spectral_matrix=[[1.0, 0.2], [-0.3, 0.8]])
    truth = make_truth(n=2, theta=jnp.array([[2.0, 3.0], [1.0, 4.0]]))
    factors = np.array([[[0.3], [-0.2]], [[0.1], [0.4]]])
    plan = make_plan(
        n_planets=2,
        n_visits=2,
        noise_sigma=[[0.2, 0.4], [0.3, 0.5]],
        noise_corr_factors=factors,
        k_precision_ms=[0.8, 1.2],
        k_precision_assumed_ms=[0.4, 2.4],
        astrometric_precision_mas=[[0.7, 0.9], [1.1, 1.3]],
        visit_photometric_snr=[[4.0, 6.0], [3.0, 5.0]],
    )
    expected = model.expected_data(truth, plan)
    np.testing.assert_allclose(expected.spectrum, [[3.1, 2.3], [2.3, 3.4]], rtol=1e-6)
    np.testing.assert_allclose(expected.photometry, [[2.5, 2.5], [1.5, 1.5]])
    np.testing.assert_allclose(
        expected.astrometry_mas, [[[2.5, 3.5]] * 2, [[1.5, 4.5]] * 2]
    )
    np.testing.assert_allclose(expected.rv_semi_amplitude_ms, [3.5, 4.5])

    # Same coordinates as stream, vectorized over stable realization ids.
    planet_key = jax.random.fold_in(jax.random.PRNGKey(123), 41)
    keys = jax.vmap(
        lambda i: jax.random.fold_in(
            jax.random.fold_in(planet_key, i), int(Purpose.MEASUREMENT)
        )
    )(jnp.arange(n))
    samples = jax.jit(jax.vmap(lambda key: model.simulate(key, truth, plan)))(keys)
    values = np.concatenate(
        [
            np.asarray(getattr(samples, name)).reshape(n, -1)
            for name in (
                "spectrum",
                "photometry",
                "astrometry_mas",
                "rv_semi_amplitude_ms",
            )
        ],
        axis=1,
    )
    covariance = np.diag(
        np.concatenate(
            [
                np.asarray(plan.noise_sigma).reshape(-1) ** 2,
                (
                    np.asarray(expected.photometry)
                    / np.asarray(plan.visit_photometric_snr)
                ).reshape(-1)
                ** 2,
                np.repeat(
                    np.asarray(plan.astrometric_precision_mas).reshape(-1) ** 2, 2
                ),
                np.asarray(plan.k_precision_ms) ** 2,
            ]
        )
    )
    for p in range(2):
        covariance[2 * p : 2 * p + 2, 2 * p : 2 * p + 2] += factors[p] @ factors[p].T
    mean_se = np.sqrt(np.diag(covariance) / n)
    assert np.all(np.abs(values.mean(0) - flatten_data(expected)) <= 5 * mean_se)
    covariance_se = np.sqrt(
        (np.outer(np.diag(covariance), np.diag(covariance)) + covariance**2) / (n - 1)
    )
    assert np.all(
        np.abs(np.cov(values, rowvar=False) - covariance) <= 5 * covariance_se
    )


def test_failed_visits_and_missing_spectra_give_false_masks_and_no_nan():
    model, truth = LinearGaussianTruthModel(), make_truth()
    plan = make_plan(
        spectrum_visit=[0, -1, 1],
        visit_success=[
            [False, True, True, True],
            [True, False, True, True],
            [True, False, False, True],
        ],
        astrometric_precision_mas=[
            [jnp.inf, 2, 2, 2],
            [2, jnp.inf, 2, 2],
            [2, jnp.inf, jnp.inf, 2],
        ],
        visit_photometric_snr=[[0, 5, 0, 5], [5, 0, 5, 5], [5, 0, 0, 5]],
    )
    for data in (
        model.expected_data(truth, plan),
        model.simulate(stream(2, 5, 0, Purpose.MEASUREMENT), truth, plan),
    ):
        np.testing.assert_array_equal(data.has_spectrum, [False, False, False])
        np.testing.assert_array_equal(data.visit_usable, plan.visit_success)
        assert not np.any(data.spectrum)
        assert not np.any(data.photometry[~plan.visit_success])
        assert not np.any(data.astrometry_mas[~plan.visit_success])
        assert not np.any(data.rv_semi_amplitude_ms)
        assert np.all(np.isfinite(flatten_data(data)))


def test_empty_visits_and_zero_rank_covariance():
    plan = make_plan(n_visits=0, noise_corr_factors=jnp.zeros((3, 2, 0)))
    model = LinearGaussianTruthModel()
    data = model.simulate(stream(1, 0, 0, Purpose.MEASUREMENT), make_truth(), plan)
    assert data.photometry.shape == (3, 0)
    assert not np.any(data.has_spectrum)
    assert np.all(np.isfinite(flatten_data(data)))


def test_per_planet_stream_keys_preserve_draws_when_reordered():
    model, truth, plan = LinearGaussianTruthModel(), make_truth(), make_plan()
    keys = jnp.stack([stream(0, i, 7, Purpose.MEASUREMENT) for i in (32, 1, 99)])
    data = model.simulate(keys, truth, plan)
    perm = jnp.array([2, 0, 1])
    reordered = model.simulate(keys[perm], truth, plan)
    np.testing.assert_array_equal(reordered.spectrum, data.spectrum[perm])
    np.testing.assert_array_equal(reordered.astrometry_mas, data.astrometry_mas[perm])


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("labels", ("same", "same"), "unique"),
        ("labels", ("one",), "one label"),
        ("nominal_index", -1, "nominal_index"),
        ("nominal_index", 2, "nominal_index"),
        ("members", (), "nonempty"),
        (
            "members",
            (Uniform(jnp.zeros(2), jnp.ones(2)), Uniform(jnp.zeros(3), jnp.ones(3))),
            "ndim",
        ),
    ],
)
def test_prior_set_invariants(field, value, message):
    prior = Uniform(jnp.zeros(2), jnp.ones(2))
    fields = dict(
        members=(prior, prior), id="test@1", labels=("a", "b"), nominal_index=0
    )
    fields[field] = value
    with pytest.raises(ValueError, match=message):
        PriorSet(**fields)


@pytest.mark.parametrize("cause", ["dimension", "nominal", "units"])
def test_model_prior_invariants(cause):
    class InvalidModel(LinearGaussianTruthModel):
        param_units: ClassVar[tuple[str, ...]] = (
            ("1",) if cause == "units" else ("1", "1")
        )

        def __init__(self):
            super().__init__()
            if cause == "dimension":
                member = Uniform(jnp.zeros(3), jnp.ones(3))
                self.prior_sets = (PriorSet((member,), "bad@1", ("nominal",), 0),)
            elif cause == "nominal":
                member = Uniform(jnp.zeros(2), jnp.ones(2))
                self.prior_sets = (PriorSet((member,), "bad@1", ("nominal",), 0),)

    message = {
        "dimension": "prior set.*wrong D",
        "nominal": "nominal member differs",
        "units": "param_names, param_units and prior.ndim must agree on D",
    }[cause]
    with pytest.raises(ValueError, match=message):
        InvalidModel()


def test_uniform_prior_quantities_and_public_exports():
    model = LinearGaussianTruthModel(low=[-2, -3], high=[4, 5])
    np.testing.assert_array_equal(model.prior.forward(jnp.array([0.0, 1.0])), [-2, 5])
    assert float(model.prior.log_prob(jnp.zeros(2))) == pytest.approx(-np.log(48))
    truth = make_truth()
    quantities = model.quantities(truth.theta, truth.context)
    for i, name in enumerate(model.param_names):
        np.testing.assert_array_equal(quantities[name], truth.theta[:, i])
    for cls in (Truth, ObservedData, PriorSet, AbstractTruthModel, UnsupportedProfile):
        assert getattr(spaceodyssey, cls.__name__) is cls
    assert spaceodyssey.check_compatible is check_compatible
    assert eqx.tree_equal(model.prior, model.prior_sets[0].members[0])


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_simulate_rejects_non_threefry_keys(implementation):
    with pytest.raises(ValueError, match="threefry2x32"):
        LinearGaussianTruthModel().simulate(
            jax.random.key(0, impl=implementation), make_truth(), make_plan()
        )


@pytest.mark.parametrize("key_factory", [jax.random.key, jax.random.PRNGKey])
def test_simulate_accepts_typed_and_raw_threefry_keys(key_factory, monkeypatch):
    class KeyImplementation:
        def __str__(self):
            return "threefry2x32"

    implementation = KeyImplementation()
    assert implementation != "threefry2x32"
    monkeypatch.setattr(jax.random, "key_impl", lambda key: implementation)
    model = LinearGaussianTruthModel()
    data = model.simulate(key_factory(0), make_truth(), make_plan())
    assert data.spectrum.shape == (3, 2)
    assert bool(jnp.all(jnp.isfinite(data.spectrum)))


def test_deprecated_truth_model_alias():
    import spaceodyssey.testing as testing

    with pytest.warns(DeprecationWarning, match=r"LinearGaussianTruthModel.*0\.1\.0"):
        old = testing.LinearGaussianTruth
    assert old is testing.LinearGaussianTruthModel
