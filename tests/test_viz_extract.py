"""Known numerical answers for visualization, without rendering."""

import sys
from dataclasses import replace

import numpy as np
import pytest

from spaceodyssey.profiles import ALPHA_BAND, EXOEARTH_CANDIDATE, WATER_ONLY
from spaceodyssey.reporting import PairedDifference, certified_yield
from spaceodyssey.viz import QuantityDisplay, _extract
from tests.test_plan import make_outcome, make_plan


@pytest.fixture(autouse=True)
def block_eyepiece(monkeypatch):
    """Every numerical regression runs with eyepiece unavailable."""
    monkeypatch.setitem(sys.modules, "eyepiece", None)


def test_plane_region_known_points():
    mask, skipped = _extract.plane_region_mask(
        EXOEARTH_CANDIDATE, [1.0, 0.5, 1.0], [1.0, 1.0, 1.5]
    )
    np.testing.assert_array_equal(mask, [True, False, False])
    assert skipped == ("water",)


@pytest.mark.parametrize("a_eeid, expected", [(1.67, 0.6191), (0.95, 0.8208)])
def test_plane_lower_radius_edge(a_eeid, expected):
    rp_re = np.geomspace(0.5, 1.5, 10001)
    mask, _ = _extract.plane_region_mask(EXOEARTH_CANDIDATE, a_eeid, rp_re)
    first = np.flatnonzero(mask)[0]
    bound = 0.8 / np.sqrt(a_eeid)
    assert rp_re[first - 1] < bound <= rp_re[first]
    assert rp_re[first] == pytest.approx(expected, abs=0.0002)


def test_plane_grid_broadcast_and_skipped_claims():
    a_eeid = np.geomspace(0.5, 2, 400)[None, :]
    rp_re = np.geomspace(0.5, 2, 400)[:, None]
    mask, skipped = _extract.plane_region_mask(EXOEARTH_CANDIDATE, a_eeid, rp_re)
    assert mask.shape == (400, 400)
    assert skipped == ("water",)
    mask, skipped = _extract.plane_region_mask(WATER_ONLY, a_eeid, rp_re)
    assert mask.all()
    assert skipped == ("water",)


@pytest.mark.parametrize("a_eeid, rp_re", [(0, 1), (-1, 1), (1, np.nan), (np.inf, 1)])
def test_plane_invalid_coordinates(a_eeid, rp_re):
    with pytest.raises(ValueError, match="finite and positive"):
        _extract.plane_region_mask(EXOEARTH_CANDIDATE, a_eeid, rp_re)


def test_edges_and_panel_order():
    grouped = _extract.edges_by_quantity(EXOEARTH_CANDIDATE)
    assert tuple(grouped) == (
        "ln_a_eeid",
        "ln_rp_re",
        "ln_rp_sqrt_a_eeid",
        "log10_h2o_vmr",
    )
    expected = {
        "ln_a_eeid": [(np.log(0.95), "above"), (np.log(1.67), "below")],
        "ln_rp_re": [(np.log(1.4), "below")],
        "ln_rp_sqrt_a_eeid": [(np.log(0.8), "above")],
        "log10_h2o_vmr": [(-4, "above")],
    }
    for quantity, edges in grouped.items():
        np.testing.assert_allclose(
            [e.threshold for e in edges], [e[0] for e in expected[quantity]]
        )
        assert [e.side for e in edges] == [e[1] for e in expected[quantity]]
    reversed_profile = replace(
        EXOEARTH_CANDIDATE, claims=EXOEARTH_CANDIDATE.claims[::-1]
    )
    assert next(iter(_extract.edges_by_quantity(reversed_profile))) == "log10_h2o_vmr"


@pytest.mark.parametrize("quantity", ["ln_a_eeid", "ln_rp_re", "ln_rp_sqrt_a_eeid"])
def test_log_quantity_display(quantity):
    display = _extract.quantity_display(quantity)
    assert (display.transform, display.scale) == ("exp", "log")
    np.testing.assert_allclose(
        _extract.transform_quantity(np.log([0.8, 1.4]), quantity), [0.8, 1.4]
    )


def test_identity_display_and_overrides():
    display = _extract.quantity_display("log10_h2o_vmr")
    assert (display.transform, display.scale) == ("identity", "linear")
    np.testing.assert_array_equal(
        _extract.transform_quantity([-4, -3], "log10_h2o_vmr"), [-4, -3]
    )
    assert _extract.quantity_display("custom") == QuantityDisplay(
        "custom", "identity", "linear"
    )
    overrides = {"ln_a_eeid": QuantityDisplay("stored distance", "identity", "linear")}
    np.testing.assert_array_equal(
        _extract.transform_quantity([1, 2], "ln_a_eeid", overrides), [1, 2]
    )
    assert _extract.quantity_display("ln_rp_re", overrides).transform == "exp"


@pytest.fixture
def samples():
    return {
        "ln_a_eeid": np.log([1.0, 0.5, 1.2, 1.1]),
        "ln_rp_re": np.log([1.0, 1.0, 1.5, 0.7]),
        "ln_rp_sqrt_a_eeid": np.log([1.0, 0.7, 1.6, 0.7]),
        "log10_h2o_vmr": np.array([-3, -5, -4, -2]),
    }


def test_claim_mass_weighted_and_uniform(samples):
    truth = np.asarray(EXOEARTH_CANDIDATE.claim_truth(samples))
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(
        _extract.claim_mass(samples, EXOEARTH_CANDIDATE, weights),
        np.average(truth, weights=weights, axis=0),
    )
    np.testing.assert_allclose(
        _extract.claim_mass(samples, EXOEARTH_CANDIDATE), truth.mean(axis=0)
    )
    np.testing.assert_allclose(
        _extract.claim_mass(samples, EXOEARTH_CANDIDATE, np.ones(4)), truth.mean(axis=0)
    )


def test_claim_mass_missing_key_names_received_keys(samples):
    samples.pop("ln_rp_re")
    with pytest.raises(
        ValueError, match=r"missing quantity 'ln_rp_re'.*received keys.*ln_a_eeid"
    ):
        _extract.claim_mass(samples, EXOEARTH_CANDIDATE)


@pytest.mark.parametrize("weights", [[0] * 4, [-1, 1, 1, 1], [np.nan] * 4, [1, 2]])
def test_claim_mass_invalid_weights(samples, weights):
    with pytest.raises(ValueError, match="weights"):
        _extract.claim_mass(samples, EXOEARTH_CANDIDATE, weights)


def test_claim_mass_inclusive_edges():
    values = {"log10_h2o_vmr": [-5.0, -4.0, -3.0]}
    np.testing.assert_allclose(_extract.claim_mass(values, WATER_ONLY), [2 / 3])


def test_report_intervals_are_read_exactly():
    difference = PairedDifference(
        yields={"y_cert_pos": 2.0},
        replicate_yields={},
        standard_errors={"y_cert_pos": 99.0},
        intervals={"y_cert_pos": (-7.5, 11.25)},
        interval_kind="replicate-batch",
        replicate_key="example",
        confidence=0.95,
    )
    assert _extract.report_interval(difference) == (2.0, -7.5, 11.25)
    assert _extract.report_interval(difference, interval="none") == (2.0, None, None)
    report = certified_yield(make_outcome(), [1, 2, 3], certificate="example")
    report = replace(
        report,
        intervals={"y_cert_pos": (-8, 12)},
        total_intervals={"y_cert_pos": (-9, 15)},
    )
    assert _extract.report_interval(report) == (report.y_cert_pos, -8, 12)
    assert _extract.report_interval(report, interval="total") == (
        report.y_cert_pos,
        -9,
        15,
    )


def test_empty_report_intervals_name_kind_and_remedy():
    report = certified_yield(make_outcome(), [1, 2, 3], certificate="example")
    report = replace(report, intervals={})
    with pytest.raises(ValueError, match=r'independent-lower-bound.*interval="none"'):
        _extract.report_interval(report)
    assert _extract.report_interval(report, interval="none") == (
        report.y_cert_pos,
        None,
        None,
    )


def test_bare_report_intervals():
    assert _extract.report_interval(2) == (2, None, None)
    assert _extract.report_interval((2, -1, 8)) == (2, -1, 8)
    assert _extract.report_interval((2, -1, 8), interval="none") == (2, None, None)
    assert np.isnan(_extract.report_interval(None)[0])
    with pytest.raises(ValueError, match="interval must"):
        _extract.report_interval(2, interval="invalid")


def test_yield_comparison_extraction_keeps_missing_rows_and_signed_gap():
    panels, gap = _extract.yield_comparison_data(
        {"first": (2, -8, 12), "pending": None, "last": 1},
        references={"pending": 3},
        margin={"pending": 0.5, "y_wrong": 0.125},
        difference={"y_wrong": (0, -2, 7)},
        difference_channels=("y_wrong",),
        gap=("first", "last"),
    )
    assert tuple(panels[0]) == ("first", "pending", "last")
    assert panels[0]["first"] == (2, -8, 12, None, None)
    assert np.isnan(panels[0]["pending"][0])
    assert panels[0]["pending"][1:] == (None, None, 3, 0.5)
    assert panels[1]["y_wrong"] == (0, -2, 7, 0, 0.125)
    assert gap == (2, 1, -1)


def test_yield_comparison_extraction_accepts_device_arrays():
    import jax.numpy as jnp

    panels, _ = _extract.yield_comparison_data(
        {"plan": jnp.array([2, -8, 12]), "prediction": jnp.array(3)},
        references={"plan": jnp.array(4)},
        difference={"y_wrong": jnp.array([0, -1, 2])},
        difference_channels=("y_wrong",),
    )
    assert panels[0]["plan"] == (2, -8, 12, 4, None)
    assert panels[0]["prediction"] == (3, None, None, None, None)
    assert panels[1]["y_wrong"] == (0, -1, 2, 0, None)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"rows": {}}, "nonempty mapping"),
        ({"channel": "unknown"}, "unknown yield channel"),
        ({"interval": "unknown"}, "interval must"),
        ({"references": {"unknown": 1}}, "row names"),
        ({"references": {"plan": np.inf}}, "finite"),
        ({"predicted": ("unknown",)}, "row names"),
        ({"margin": -1}, "finite and nonnegative"),
        ({"margin": np.nan}, "finite and nonnegative"),
        ({"margin": [1, 2]}, "finite and nonnegative"),
        ({"margin": {"unknown": 1}}, "margin keys"),
        ({"difference": {}, "difference_channels": ()}, "nonempty subset"),
        ({"difference": {}, "difference_channels": ("unknown",)}, "nonempty subset"),
        ({"difference": {}}, "missing channel"),
        ({"gap": ("plan", "unknown")}, "two row names"),
        ({"rows": {"plan": None}, "gap": ("plan", "plan")}, "finite point"),
    ],
)
def test_yield_comparison_extraction_rejects_invalid_inputs(options, message):
    with pytest.raises(ValueError, match=message):
        _extract.yield_comparison_data(**{"rows": {"plan": 2}, **options})


def test_planet_arrays_ignore_replicates_and_retain_out_of_range():
    outcome = make_outcome(
        in_range=[True, False, True],
        p_pos_rep=np.full((5, 3), 0.6),
        p_neg_rep=np.full((5, 3), 0.1),
        p_wrong_rep=np.zeros((5, 3)),
        p_neg_wrong_rep=np.zeros((5, 3)),
        replicate_kind="bootstrap",
        replicate_key="example",
    )
    series, in_range = _extract.verdict_series(outcome, ("pos", "neg", "wrong"))
    assert all(values.shape == (3,) for values in series.values())
    np.testing.assert_array_equal(in_range, [True, False, True])
    data = _extract.planet_plane_data(
        [1, 1.1, 1.2], [0.9, 1, 1.1], values=outcome, active=[True, False, True]
    )
    assert len(data["values"]) == 3
    assert (~data["in_range"]).sum() == 1
    assert (~data["active"]).sum() == 1
    bare = _extract.planet_plane_data(
        data["a_eeid"],
        data["rp_re"],
        values=series["pos"],
        active=data["active"],
        in_range=in_range,
    )
    for key in data:
        np.testing.assert_array_equal(data[key], bare[key])
    bare_series, bare_range = _extract.verdict_series(
        {**series, "in_range": in_range}, tuple(series)
    )
    for key in series:
        np.testing.assert_array_equal(series[key], bare_series[key])
    np.testing.assert_array_equal(in_range, bare_range)


def test_plane_range_override_and_cloud_defaults():
    data = _extract.planet_plane_data(
        [1, 2, 3], [1, 1, 1], values=make_outcome(), in_range=[False] * 3
    )
    assert not data["in_range"].any()
    cloud = _extract.planet_plane_data([1], [1])
    assert cloud["values"] is None
    assert cloud["active"].all() and cloud["in_range"].all()
    assert not cloud["marked"].any()


def test_plane_region_grid_physical_and_stored_display():
    x, y, mask, skipped = _extract.plane_region_grid(
        EXOEARTH_CANDIDATE, (0.4, 2.5), (0.4, 2)
    )
    assert x.shape == y.shape == (400,)
    assert mask.shape == (400, 400) and skipped == ("water",)
    np.testing.assert_allclose(x[[0, -1]], [0.4, 2.5])
    np.testing.assert_allclose(y[[0, -1]], [0.4, 2])
    np.testing.assert_allclose(np.diff(np.log(x)), np.log(2.5 / 0.4) / 399)
    np.testing.assert_allclose(np.diff(np.log(y)), np.log(2 / 0.4) / 399)
    expected = (
        (x[None, :] >= 0.95)
        & (x[None, :] <= 1.67)
        & (y[:, None] <= 1.4)
        & (y[:, None] * np.sqrt(x[None, :]) >= 0.8)
    )
    np.testing.assert_array_equal(mask, expected)
    display = {
        name: QuantityDisplay(name, "identity", "linear")
        for name in ("ln_a_eeid", "ln_rp_re")
    }
    lx, ly, stored_mask, _ = _extract.plane_region_grid(
        EXOEARTH_CANDIDATE, np.log([0.4, 2.5]), np.log([0.4, 2]), display
    )
    np.testing.assert_allclose(np.exp(lx), x)
    np.testing.assert_allclose(np.exp(ly), y)
    np.testing.assert_array_equal(stored_mask, mask)


def test_plane_device_arrays():
    import jax.numpy as jnp

    data = _extract.planet_plane_data(
        jnp.array([1, 2]),
        jnp.array([0.5, 1]),
        values=jnp.array([0, 1]),
        active=jnp.array([True, False]),
        in_range=jnp.array([False, True]),
    )
    assert all(isinstance(value, np.ndarray) for value in data.values())
    np.testing.assert_array_equal(data["values"], [0, 1])
    np.testing.assert_array_equal(data["in_range"], [False, True])


@pytest.mark.parametrize(
    "options, match",
    [
        ({"a_eeid": [0]}, "finite and positive"),
        ({"rp_re": [-1]}, "finite and positive"),
        ({"rp_re": [np.nan]}, "finite and positive"),
        ({"a_eeid": [np.inf]}, "finite and positive"),
        ({"a_eeid": [[1]]}, "expected shape"),
        ({"rp_re": [1, 2]}, "expected shape"),
        ({"values": [np.nan]}, "finite probabilities"),
        ({"values": [1.1]}, "finite probabilities"),
        ({"values": [-0.1]}, "finite probabilities"),
        ({"values": [0, 1]}, "expected shape"),
        ({"in_range": [False, True]}, "expected shape"),
    ],
)
def test_plane_invalid_arrays(options, match):
    with pytest.raises(ValueError, match=match):
        _extract.planet_plane_data(**{"a_eeid": [1], "rp_re": [1], **options})


def test_out_of_range_midpoint_spans():
    spans = _extract.out_of_range_spans([0, 1, 2, 3], [True, False, False, True])
    np.testing.assert_array_equal(spans, [[0.5, 2.5]])
    np.testing.assert_array_equal(
        _extract.out_of_range_spans([0, 1, 2, 3], [False, True, True, False]),
        [[-0.5, 0.5], [2.5, 3.5]],
    )
    assert _extract.out_of_range_spans([0, 1], [True, True]).shape == (0, 2)
    assert _extract.out_of_range_spans([], []).shape == (0, 2)
    np.testing.assert_allclose(
        _extract.out_of_range_spans(
            [1, 4, 16, 64], [True, False, False, True], log=True
        ),
        [[2, 32]],
    )


def test_certification_curve_extraction_device_arrays_and_intervals():
    import jax.numpy as jnp

    x, series, mask, xr, points, indicator, threshold = (
        _extract.certification_curve_data(
            jnp.array([1, 2]),
            {"pos": jnp.array([0.25, 0.75]), "in_range": [True, False]},
            channels=("pos",),
            reference={"pos": jnp.array([0, 0.5, 1]), "pos_se": [0.1, 0.1, 0.1]},
            x_reference=jnp.array([1, 1.5, 2]),
            indicator=jnp.array([0.5, 1]),
            threshold=jnp.array(1.5),
        )
    )
    assert all(isinstance(a, np.ndarray) for a in (x, series["pos"], xr, indicator))
    np.testing.assert_array_equal(mask, [True, False])
    np.testing.assert_array_equal(points["pos"][0], [0, 0.5, 1])
    np.testing.assert_allclose(points["pos"][1], [0, 0.304003601546, 0.804003601546])
    np.testing.assert_allclose(points["pos"][2], [0.195996398454, 0.695996398454, 1])
    assert threshold == 1.5


@pytest.mark.parametrize(
    "options, message",
    [
        ({"x": []}, "nonempty shape"),
        ({"x": [1, 1]}, "strictly increasing"),
        ({"x": [1, np.nan]}, "finite"),
        ({"x": [0, 1], "log_x": True}, "positive"),
        ({"outcome": {"pos": [0.1]}}, "expected shape"),
        ({"outcome": {"pos": [0.1, np.nan]}}, "probabilities"),
        ({"channels": ()}, "must not be empty"),
        ({"channels": ("unknown",)}, "unknown verdict channel"),
        ({"confidence": 1}, "confidence"),
        ({"confidence": np.nan}, "confidence"),
        ({"indicator": [0, 2]}, "probabilities"),
        ({"threshold": [1, 2]}, "expected shape"),
        ({"threshold": 0, "log_x": True}, "positive"),
        ({"x_reference": [1, 2]}, "require reference"),
        ({"reference": {"pos": [0.1]}}, "expected shape"),
        ({"reference": {"pos": [0, 1]}, "x_reference": [1, np.inf]}, "finite"),
        ({"reference": {"pos": [0, 1], "pos_se": [-1, 0]}}, "nonnegative"),
        (
            {
                "reference": {"pos": [0, 1]},
                "reference_intervals": {"pos": ([1, 1], [0, 0])},
            },
            "ordered",
        ),
    ],
)
def test_certification_curve_extraction_errors(options, message):
    with pytest.raises(ValueError, match=message):
        _extract.certification_curve_data(
            **{
                "x": [1, 2],
                "outcome": {"pos": [0.2, 0.8]},
                "channels": ("pos",),
                **options,
            }
        )


@pytest.mark.parametrize(
    "x, log", [([1], False), ([1, 1], False), ([2, 1], False), ([0, 1], True)]
)
def test_invalid_span_coordinates(x, log):
    with pytest.raises(ValueError):
        _extract.out_of_range_spans(x, [True] * len(x), log=log)


def test_profile_bound_known_answers():
    reports = [
        certified_yield(make_outcome(), [1, 2, 3], certificate="example")
        for _ in ALPHA_BAND
    ]
    np.testing.assert_allclose(
        _extract.profile_bounds(ALPHA_BAND, reports),
        [
            profile.alpha * report.y_issued
            for profile, report in zip(ALPHA_BAND, reports, strict=True)
        ],
    )
    separate = [replace(profile, support_rule="separate") for profile in ALPHA_BAND]
    np.testing.assert_allclose(
        _extract.profile_bounds(separate, reports),
        [
            profile.alpha * len(profile.claims) * report.y_issued
            for profile, report in zip(separate, reports, strict=True)
        ],
    )
    with pytest.raises(ValueError, match="same length"):
        _extract.profile_bounds(ALPHA_BAND, reports[:1])


def test_zero_floor_retains_original_zero_mask():
    displayed, zeros = _extract.zero_floor_values([0, 0.001, 2], 0.01)
    np.testing.assert_array_equal(displayed, [0.01, 0.001, 2])
    np.testing.assert_array_equal(zeros, [True, False, False])


def test_yield_sweep_extraction_bare_device_arrays():
    import jax.numpy as jnp

    x, labels, profiles, series, bounds = _extract.yield_sweep_data(
        jnp.array([1, 2]),
        {"example": {"y_wrong": jnp.array([[0, -0.1, 0.2], [0.1, 0, 0.3]])}},
        ("y_wrong",),
        bound=jnp.array([0.2, 0.4]),
    )
    np.testing.assert_array_equal(x, [1, 2])
    assert labels is None and not profiles
    point, lo, hi = series["example"]["y_wrong"]
    np.testing.assert_allclose(point, [0, 0.1])
    np.testing.assert_allclose(lo, [-0.1, 0])
    np.testing.assert_allclose(hi, [0.2, 0.3])
    np.testing.assert_allclose(bounds["example"], [0.2, 0.4])


def test_sweep_floor_clips_only_interval_endpoints():
    point, lo, hi, zeros = _extract.sweep_floor_data(
        [0, 0.001, 2], [-1, 0, 1], [0, 0.005, 3], 0.01
    )
    np.testing.assert_array_equal(point, [0.01, 0.001, 2])
    np.testing.assert_array_equal(lo, [0.01, 0.01, 1])
    np.testing.assert_array_equal(hi, [0.01, 0.01, 3])
    np.testing.assert_array_equal(zeros, [True, False, False])


@pytest.mark.parametrize(
    "x, reports, bound, message",
    [
        ([], [], None, "nonempty shape"),
        ([[1, 2]], [], None, "nonempty shape"),
        ([1, np.nan], [], None, "finite"),
        ([1, 2], [{"y_wrong": 0}], None, "expected length 2"),
        ([1, 2], {"y_wrong": [0]}, None, "expected shape"),
        ([1, 2], {"y_wrong": [-1, 2]}, None, "nonnegative"),
        ([1, 2], {"y_wrong": [0, 2]}, [-1, 2], "nonnegative"),
    ],
)
def test_yield_sweep_extraction_errors(x, reports, bound, message):
    with pytest.raises(ValueError, match=message):
        _extract.yield_sweep_data(x, reports, ("y_wrong",), bound, "none")


def test_plan_visits_skip_padding_and_extract_spectrum_flags():
    plan = make_plan(
        n_visits=3,
        visit_epochs_d=[[0, 10, np.nan], [2, 12, 22], [4, np.nan, np.nan]],
        visit_success=[[True, False, False], [False, True, True], [True, False, False]],
        spectrum_visit=[0, 2, -1],
        k_precision_ms=[1, np.inf, 2],
    )
    data = _extract.plan_visit_data(plan)
    np.testing.assert_array_equal(data["epoch_d"], [0, 10, 2, 12, 22, 4])
    assert data["spectrum"].sum() == (np.asarray(plan.spectrum_visit) >= 0).sum()
    np.testing.assert_array_equal(data["mass_measured"], [True, False, True])
    mapping = {
        name: np.asarray(getattr(plan, name))
        for name in (
            "visit_epochs_d",
            "visit_success",
            "visit_photometric_snr",
            "spectrum_visit",
            "k_precision_ms",
        )
    }
    bare = _extract.plan_visit_data(mapping)
    for key in data:
        np.testing.assert_array_equal(data[key], bare[key])
    selected = _extract.plan_visit_data(plan, [2, 0])
    np.testing.assert_array_equal(selected["epoch_d"], [4, 0, 10])
    np.testing.assert_array_equal(selected["row"], [0, 1, 1])
    assert selected["spectrum"].sum() == 1


@pytest.fixture
def claim_outcome():
    return make_outcome(
        claim_names=("zone", "water"),
        claim_p_pos=np.full((3, 2), 0.6),
        claim_p_neg=np.full((3, 2), 0.1),
        claim_p_wrong=np.zeros((3, 2)),
        claim_p_neg_wrong=np.zeros((3, 2)),
        claim_truth_positive=np.ones((3, 2), dtype=bool),
        claim_in_range=[[True, False], [False, False], [True, True]],
    )


def test_tier_agreement_hollow_count_and_bare_arrays(claim_outcome):
    names, series, in_range = _extract.tier_agreement_series(
        claim_outcome, claim_outcome
    )
    assert names == ("zone", "water")
    assert (~in_range).sum() * len(series) == 6
    bare = {
        "claim_names": names,
        "pos": np.asarray(claim_outcome.claim_p_pos).T,
        "neg": np.asarray(claim_outcome.claim_p_neg).T,
        "in_range": in_range,
    }
    bare_names, bare_series, bare_range = _extract.tier_agreement_series(bare, bare)
    assert bare_names == names
    np.testing.assert_array_equal(bare_range, in_range)
    for verdict, (x, y) in series.items():
        assert x.shape == y.shape == (2, 3)
        np.testing.assert_array_equal(x, bare_series[verdict][0])
        np.testing.assert_array_equal(y, bare_series[verdict][1])
    names, _, mask = _extract.tier_agreement_series(
        bare, bare, claims=("water",), verdicts=("pos",)
    )
    assert names == ("water",)
    assert (~mask).sum() == 2


def test_tier_agreement_claim_name_mismatch(claim_outcome):
    bare = {
        "claim_names": ("water", "zone"),
        "pos": np.full((2, 3), 0.6),
        "neg": np.full((2, 3), 0.1),
        "in_range": np.ones((2, 3), dtype=bool),
    }
    with pytest.raises(ValueError, match="claim_names disagree"):
        _extract.tier_agreement_series(claim_outcome, bare)


def test_tier_agreement_missing_claim_fields(claim_outcome):
    with pytest.raises(ValueError, match="claim fields"):
        _extract.tier_agreement_series(make_outcome(), claim_outcome)
    with pytest.raises(ValueError, match="claim fields"):
        _extract.tier_agreement_series(claim_outcome, make_outcome())


def test_claim_mass_keeps_sample_dtype_at_thresholds():
    import jax.numpy as jnp

    from spaceodyssey.profiles import EXOEARTH_CANDIDATE

    samples = {
        edge.quantity: np.full(3, edge.threshold, dtype=np.float32)
        for claim in EXOEARTH_CANDIDATE.claims
        for edge in claim.edges
    }
    direct = np.asarray(
        EXOEARTH_CANDIDATE.claim_truth({k: jnp.asarray(v) for k, v in samples.items()})
    ).mean(axis=0)
    np.testing.assert_allclose(_extract.claim_mass(samples, EXOEARTH_CANDIDATE), direct)


@pytest.fixture
def observed_arrays():
    data = {
        "spectrum": [[1, 2]],
        "astrometry_mas": [[[10, -5], [np.nan, np.nan], [20, 15]]],
        "visit_usable": [[True, False, True]],
    }
    plan = {
        "noise_sigma": [[0.1, 0.25]],
        "bin_wavelength_nm": [500, 900],
        "astrometric_precision_mas": [[2, np.nan, 3]],
    }
    return data, plan


def test_observed_extraction_covariance_and_units(observed_arrays):
    data, plan = observed_arrays
    result = _extract.observed_data_arrays(
        data, plan, astrometric_corr=-0.5, expected=[0.9, 1.8]
    )
    np.testing.assert_array_equal(result["sky"], [[10, -5], [20, 15]])
    np.testing.assert_array_equal(
        result["covariance"], [[[4, -2], [-2, 4]], [[9, -4.5], [-4.5, 9]]]
    )
    np.testing.assert_array_equal(result["wavelength_um"], [0.5, 0.9])
    np.testing.assert_array_equal(result["lo"], [0.9, 1.75])
    np.testing.assert_array_equal(result["hi"], [1.1, 2.25])
    np.testing.assert_array_equal(result["expected"], [0.9, 1.8])
    assert result["has_spectrum"]


def test_observed_extraction_device_arrays_and_absent_data(observed_arrays):
    import jax.numpy as jnp

    data, plan = observed_arrays
    data = {k: jnp.asarray(v) for k, v in data.items()}
    plan = {k: jnp.asarray(v) for k, v in plan.items()}
    result = _extract.observed_data_arrays(data, plan, planet=jnp.asarray(0))
    assert isinstance(result["sky"], np.ndarray)
    np.testing.assert_allclose(result["hi"], [1.1, 2.25])
    data.update(has_spectrum=[False], visit_usable=[[False] * 3])
    result = _extract.observed_data_arrays(data, plan)
    assert result["sky"].shape == (0, 2)
    assert result["covariance"].shape == (0, 2, 2)
    assert result["spectrum"].size == result["lo"].size == result["hi"].size == 0


@pytest.mark.parametrize(
    "options, message",
    [
        ({"planet": -1}, "integer index"),
        ({"planet": 1}, "integer index"),
        ({"planet": 0.5}, "integer index"),
        ({"astrometric_corr": 1.1}, "in "),
        ({"astrometric_corr": np.nan}, "finite"),
        ({"expected": [1]}, "expected shape"),
    ],
)
def test_observed_extraction_invalid_arguments(observed_arrays, options, message):
    with pytest.raises(ValueError, match=message):
        _extract.observed_data_arrays(*observed_arrays, **options)


@pytest.mark.parametrize(
    "source, field, value, message",
    [
        (0, "spectrum", [1, 2], "expected shape"),
        (0, "visit_usable", [[True]], "expected shape"),
        (0, "astrometry_mas", [[[np.nan, 1]] * 3], "usable astrometry"),
        (1, "astrometric_precision_mas", [[0, 1, 2]], "positive precision"),
        (1, "noise_sigma", [[0, 1]], "positive noise_sigma"),
        (1, "bin_wavelength_nm", [500], "expected shape"),
    ],
)
def test_observed_extraction_invalid_fields(
    observed_arrays, source, field, value, message
):
    observed_arrays[source][field] = value
    with pytest.raises(ValueError, match=message):
        _extract.observed_data_arrays(*observed_arrays)
