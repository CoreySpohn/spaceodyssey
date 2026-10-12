"""Plot contracts and numerical artist regressions, without image baselines."""

from dataclasses import replace

import eyepiece as ep
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from spaceodyssey.profiles import ALPHA_BAND, EXOEARTH_CANDIDATE, WATER_ONLY
from spaceodyssey.reporting import PairedDifference, certified_yield
from spaceodyssey.viz import (
    plot_certification_curve,
    plot_planet_plane,
    plot_yield_comparison,
    plot_yield_sweep,
)
from tests.test_plan import make_outcome


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def curve_outcome():
    return make_outcome(
        n_planets=4,
        p_pos=[0, 0.2, 0.5, 0.9],
        p_neg=[0.8, 0.6, 0.3, 0],
        p_wrong=[0, 0.02, 0.01, 0.03],
        p_pos_se=[0.1, 0.05, 0.1, 0.2],
        p_neg_se=[0.2, 0.1, 0.05, 0.1],
        in_range=[True, False, False, True],
        p_pos_rep=np.full((7, 4), 0.55),
        p_neg_rep=np.full((7, 4), 0.15),
        p_wrong_rep=np.full((7, 4), 0.05),
        p_neg_wrong_rep=np.zeros((7, 4)),
        replicate_kind="bootstrap",
        replicate_key="example",
    )


def test_plot_certification_curve_smoke(curve_outcome, save_viz_png):
    result = plot_certification_curve(
        [0, 1, 2, 3],
        curve_outcome,
        reference=curve_outcome,
        indicator=[0.1, 0.4, 0.8, 1],
        threshold=1.5,
        x_label="signal-to-noise ratio (dimensionless)",
    )
    assert isinstance(result, ep.PlotResult)
    assert result.update is None and result.insets == ()
    assert set(result.artists) == {
        "line",
        "scatter",
        "lines",
        "fill",
        "indicator",
        "threshold",
        "text",
    }
    assert len(result.artists["line"]) == len(result.artists["scatter"]) == 2
    assert len(result.artists["lines"]) == 8
    assert result.ax.get_ylim() == (0, 1)
    assert "dimensionless" in result.ax.get_xlabel()
    assert result.artists["indicator"][0].get_linestyle() == "--"
    np.testing.assert_array_equal(
        result.artists["threshold"][0].get_xdata(), [1.5, 1.5]
    )
    assert result.artists["text"][0].get_transform() == result.ax.get_xaxis_transform()
    result.fig.set_size_inches(6, 3)
    result.fig.canvas.draw()
    save_viz_png(result, "plot_certification_curve")


@pytest.mark.parametrize(
    "x, log_x, expected",
    [([0, 1, 2, 3], False, [0.5, 2.5]), ([1, 4, 16, 64], True, [2, 32])],
)
def test_certification_curve_span_known_answer(curve_outcome, x, log_x, expected):
    result = plot_certification_curve(x, curve_outcome, log_x=log_x)
    (span,) = result.artists["fill"]
    vertices = span.get_patch_transform().transform(span.get_path().vertices)
    np.testing.assert_allclose([vertices[:, 0].min(), vertices[:, 0].max()], expected)
    np.testing.assert_allclose(span.get_facecolor(), to_rgba(ep.neutral(0.12)))
    assert result.ax.get_xscale() == ("log" if log_x else "linear")


def test_certification_curve_replicates_are_not_planets(curve_outcome):
    channels = ("pos", "neg", "wrong")
    result = plot_certification_curve(
        [0, 1, 2, 3], curve_outcome, reference=curve_outcome, channels=channels
    )
    assert len(result.artists["line"]) == len(result.artists["scatter"]) == 3
    for index, channel in enumerate(channels):
        expected = np.asarray(getattr(curve_outcome, f"p_{channel}"))
        line, points = result.artists["line"][index], result.artists["scatter"][index]
        assert len(line.get_xdata()) == len(points.get_offsets()) == 4
        np.testing.assert_array_equal(line.get_ydata(), expected)
        np.testing.assert_array_equal(points.get_offsets()[:, 1], expected)
        np.testing.assert_array_equal(points.get_facecolors()[:, 3] == 0, expected == 0)


def test_certification_curve_bare_arrays_and_reference_intervals(curve_outcome):
    channels = ("pos", "neg", "wrong")
    bare = {
        key: np.asarray(getattr(curve_outcome, f"p_{key}"))
        for ch in channels
        for key in (ch, f"{ch}_se")
    }
    bare["in_range"] = np.asarray(curve_outcome.in_range)
    options = dict(
        channels=channels,
        x_reference=[0.1, 0.9, 1.9, 2.9],
        confidence=0.9,
        reference_intervals={"wrong": ([0.001] * 4, [0.123] * 4)},
    )
    typed = plot_certification_curve(
        [0, 1, 2, 3], curve_outcome, reference=curve_outcome, **options
    )
    arrays = plot_certification_curve([0, 1, 2, 3], bare, reference=bare, **options)
    for key in ("line", "lines"):
        for a, b in zip(typed.artists[key], arrays.artists[key], strict=True):
            np.testing.assert_array_equal(a.get_xydata(), b.get_xydata())
    for a, b in zip(typed.artists["scatter"], arrays.artists["scatter"], strict=True):
        np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
    z = 1.6448536269514722
    for index, channel in enumerate(channels[:2]):
        expected = bare[channel]
        se = bare[f"{channel}_se"]
        for j, line in enumerate(typed.artists["lines"][index * 4 : (index + 1) * 4]):
            np.testing.assert_allclose(
                line.get_ydata(),
                np.clip([expected[j] - z * se[j], expected[j] + z * se[j]], 0, 1),
            )
    for line in typed.artists["lines"][8:]:
        np.testing.assert_array_equal(line.get_ydata(), [0.001, 0.123])


@pytest.mark.parametrize("layout", [None, "constrained"])
def test_certification_curve_preserves_caller_axes(curve_outcome, layout):
    fig, axes = plt.subplots(1, 2, layout=layout)
    engine = fig.get_layout_engine()
    positions = [ax.get_position(original=True).bounds for ax in axes]
    result = plot_certification_curve([0, 1, 2, 3], curve_outcome, ax=axes[0])
    assert result.ax is axes[0] and fig.get_layout_engine() is engine
    for ax, position in zip(axes, positions, strict=True):
        np.testing.assert_array_equal(ax.get_position(original=True).bounds, position)
        assert ax.get_shared_x_axes().get_siblings(ax) == [ax]
        assert ax.get_shared_y_axes().get_siblings(ax) == [ax]
    assert not axes[1].lines and not axes[1].collections and not axes[1].patches


def test_certification_curve_axes_shape_error(curve_outcome):
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match=r"expected shape \(\), received \(2,\)"):
        plot_certification_curve([0, 1, 2, 3], curve_outcome, ax=axes)


def test_certification_curve_styles_overlays_and_short_reference(curve_outcome):
    x = [0, 1, 2, 3]
    first = plot_certification_curve(x, curve_outcome, channels=("wrong", "neg", "pos"))
    palette = ep.SourceStyles(["positive", "negative"])
    for line, color in zip(
        first.artists["line"],
        [ep.role("alert"), palette["negative"]["color"], palette["positive"]["color"]],
        strict=True,
    ):
        assert to_rgba(line.get_color()) == to_rgba(color)
    line_kw, point_kw = {"lw": 3}, {"s": 70}
    interval_kw, span_kw, ref_kw = {"lw": 4}, {"alpha": 0.25}, {"lw": 2}
    second = plot_certification_curve(
        x,
        curve_outcome,
        channels=("pos",),
        ax=first.ax,
        reference={"pos": [0, 0.8]},
        x_reference=[0.5, 2.5],
        reference_intervals={"pos": ([0, 0.7], [0.2, 0.9])},
        colors={"pos": "black"},
        label="trial",
        indicator=[0.1, 0.2, 0.3, 0.4],
        line_kw=line_kw,
        point_kw=point_kw,
        interval_kw=interval_kw,
        span_kw=span_kw,
        threshold=2,
        ref_kw=ref_kw,
    )
    assert second.artists["line"][0].get_linewidth() == 3
    assert second.artists["scatter"][0].get_sizes()[0] == 70
    assert second.artists["lines"][0].get_linewidth() == 4
    assert second.artists["fill"][0].get_alpha() == 0.25
    assert second.artists["threshold"][0].get_linewidth() == 2
    assert to_rgba(second.artists["indicator"][0].get_color()) == to_rgba("black")
    assert len(second.artists["scatter"][0].get_offsets()) == 2
    assert "trial: positive (reference)" in [
        t.get_text() for t in first.ax.legend_.texts
    ]
    assert all(line in first.ax.lines for line in first.artists["line"])
    assert line_kw == {"lw": 3} and point_kw == {"s": 70}
    assert (
        interval_kw == {"lw": 4} and span_kw == {"alpha": 0.25} and ref_kw == {"lw": 2}
    )


def test_certification_curve_reference_without_errors():
    result = plot_certification_curve(
        [1, 2], {"pos": [0.2, 0.8]}, reference={"pos": [0, 1]}, channels=("pos",)
    )
    assert result.artists["lines"] == result.artists["fill"] == []
    np.testing.assert_array_equal(
        result.artists["scatter"][0].get_offsets(), [[1, 0], [2, 1]]
    )


@pytest.fixture
def sweep_reports():
    base = certified_yield(make_outcome(), [1, 2, 3], certificate="example")
    return [
        replace(
            base,
            y_issued=issued,
            y_wrong=wrong,
            intervals={
                "y_issued": (issued - 0.25, issued + 0.5),
                "y_wrong": (0, wrong + 0.02),
            },
            total_intervals={
                "y_issued": (issued - 1, issued + 2),
                "y_wrong": (-0.01, wrong + 0.1),
            },
        )
        for issued, wrong in zip([2, 4, 3], [0, 0.03, 0.01], strict=True)
    ]


def test_plot_yield_sweep_smoke(sweep_reports, save_viz_png):
    result = plot_yield_sweep(
        ALPHA_BAND, sweep_reports, bound="profile", zero_floor=0.001
    )
    assert isinstance(result, ep.MosaicResult)
    assert result.axes.shape == (2,)
    assert result.update is None
    assert set(result.artists) == {"line", "lines", "scatter", "bound", "zero_floor"}
    assert len(result.artists["scatter"]) == 2
    assert len(result.artists["line"]) == 2
    assert len(result.artists["lines"]) == 6
    assert len(result.artists["zero_floor"]) == 2
    assert "dimensionless" in result.axes[0].get_xlabel()
    result.fig.set_size_inches(7, 3)
    result.fig.canvas.draw()
    save_viz_png(result, "plot_yield_sweep")


@pytest.mark.parametrize("support_rule", ["joint", "separate"])
def test_yield_sweep_profile_bound(sweep_reports, support_rule):
    profiles = [replace(p, support_rule=support_rule) for p in ALPHA_BAND]
    result = plot_yield_sweep(profiles, sweep_reports, bound="profile")
    (curve,) = result.artists["bound"]
    expected = [
        p.alpha * (len(p.claims) if support_rule == "separate" else 1) * r.y_issued
        for p, r in zip(profiles, sweep_reports, strict=True)
    ]
    np.testing.assert_allclose(curve.get_ydata(), expected)
    np.testing.assert_allclose(curve.get_xdata(), [p.alpha for p in profiles])
    assert curve.axes is result.axes[1]
    assert curve.get_linestyle() == "--"
    assert to_rgba(curve.get_color()) == to_rgba(ep.neutral(0.6))
    assert "stated bound" in [t.get_text() for t in result.axes[1].legend_.texts]


def test_yield_sweep_zero_floor(sweep_reports):
    result = plot_yield_sweep(
        [1, 2, 3], sweep_reports, zero_floor=0.001, interval="total"
    )
    ax = result.axes[1]
    points = result.artists["scatter"][1]
    np.testing.assert_allclose(points.get_offsets()[:, 1], [0.001, 0.03, 0.01])
    np.testing.assert_allclose(
        result.artists["line"][1].get_ydata(), [0.001, 0.03, 0.01]
    )
    np.testing.assert_array_equal(points.get_facecolors()[:, 3], [0, 1, 1])
    np.testing.assert_allclose(result.artists["lines"][3].get_ydata(), [0.001, 0.1])
    assert ax.get_yscale() == "log"
    assert result.axes[0].get_yscale() == "linear"
    result.fig.canvas.draw()
    ticks = dict(
        zip(ax.get_yticks(), [t.get_text() for t in ax.get_yticklabels()], strict=True)
    )
    assert ticks[0.001] == "0"
    assert all(
        line.get_transform() == ax.get_yaxis_transform()
        for line in result.artists["zero_floor"]
    )


@pytest.mark.parametrize("with_bound", [False, True])
def test_yield_sweep_categorical_has_no_connecting_line(sweep_reports, with_bound):
    result = plot_yield_sweep(
        ["joint", "split", "separate"],
        sweep_reports,
        bound=[0.1, 0.2, 0.3] if with_bound else None,
        line_kw={"ls": "-", "marker": "s"},
    )
    assert result.artists["line"] == []
    for ax in result.axes:
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "joint",
            "split",
            "separate",
        ]
        for line in ax.lines:
            if line.get_linestyle() != "None":
                assert np.ptp(line.get_xdata()) == 0  # intervals only
    np.testing.assert_array_equal(
        result.artists["scatter"][0].get_offsets()[:, 0], [0, 1, 2]
    )


@pytest.mark.parametrize("layout", [None, "constrained"])
def test_yield_sweep_preserves_caller_axes(sweep_reports, layout):
    fig, axes = plt.subplots(1, 3, layout=layout)
    engine = fig.get_layout_engine()
    positions = [ax.get_position(original=True).bounds for ax in axes]
    result = plot_yield_sweep([1, 2, 3], sweep_reports, axes=axes[:2])
    assert all(a is b for a, b in zip(result.axes, axes[:2], strict=True))
    assert fig.get_layout_engine() is engine
    for ax, position in zip(axes, positions, strict=True):
        np.testing.assert_array_equal(ax.get_position(original=True).bounds, position)
        assert ax.get_shared_x_axes().get_siblings(ax) == [ax]
        assert ax.get_shared_y_axes().get_siblings(ax) == [ax]
    assert not axes[2].lines and not axes[2].collections


@pytest.mark.parametrize("shape", [(1,), (1, 2), ()])
def test_yield_sweep_axes_shape_error(shape):
    _, ax = plt.subplots()
    axes = np.full(shape, ax, dtype=object)
    with pytest.raises(ValueError) as error:
        plot_yield_sweep([1], {"y_issued": [2], "y_wrong": [0]}, axes=axes)
    assert "expected shape (2,)" in str(error.value)
    assert f"received {shape}" in str(error.value)


@pytest.mark.parametrize("interval", ["random", "total", "none"])
def test_yield_sweep_bare_arrays_and_stored_intervals(sweep_reports, interval):
    channels = ("y_wrong", "y_issued")
    field = "total_intervals" if interval == "total" else "intervals"
    bare = {
        ch: np.array([[getattr(r, ch), *getattr(r, field)[ch]] for r in sweep_reports])
        for ch in channels
    }
    typed = plot_yield_sweep(
        ALPHA_BAND, sweep_reports, channels=channels, bound="profile", interval=interval
    )
    arrays = plot_yield_sweep(
        ALPHA_BAND, bare, channels=channels, bound="profile", interval=interval
    )
    for key in ("line", "lines", "bound"):
        for a, b in zip(typed.artists[key], arrays.artists[key], strict=True):
            np.testing.assert_allclose(a.get_xydata(), b.get_xydata())
    for a, b in zip(typed.artists["scatter"], arrays.artists["scatter"], strict=True):
        np.testing.assert_allclose(a.get_offsets(), b.get_offsets())
    if interval != "none":
        for line, report in zip(typed.artists["lines"][:3], sweep_reports, strict=True):
            np.testing.assert_array_equal(
                line.get_ydata(), getattr(report, field)["y_wrong"]
            )
    else:
        assert typed.artists["lines"] == []


def test_yield_sweep_series_styles_and_per_series_bounds(sweep_reports):
    doubled = [replace(r, y_issued=r.y_issued * 2) for r in sweep_reports]
    styles = ep.SourceStyles(["second", "first"])
    result = plot_yield_sweep(
        ALPHA_BAND,
        {"first": sweep_reports, "second": doubled},
        styles=styles,
        bound="profile",
        interval="none",
    )
    for index, name in enumerate(["first", "second", "first", "second"]):
        expected_color = styles[name]["color"] if index < 2 else ep.role("alert")
        assert to_rgba(result.artists["line"][index].get_color()) == to_rgba(
            expected_color
        )
        np.testing.assert_allclose(
            result.artists["scatter"][index].get_edgecolors()[0],
            to_rgba(expected_color),
        )
    np.testing.assert_allclose(
        result.artists["bound"][1].get_ydata(),
        2 * result.artists["bound"][0].get_ydata(),
    )


def test_yield_sweep_single_panel_explicit_bound_and_overrides():
    line_kw, ref_kw, interval_kw = {"c": "black", "lw": 3}, {"lw": 2}, {"lw": 4}
    result = plot_yield_sweep(
        [1, 10],
        {"trial": {"y_wrong": [[0.1, 0.05, 0.2], [0.2, 0.1, 0.3]]}},
        channels=("y_wrong",),
        bound=[0.3, 0.4],
        log_x=True,
        log_y=True,
        x_label="exposure (d)",
        line_kw=line_kw,
        ref_kw=ref_kw,
        interval_kw=interval_kw,
    )
    assert result.axes.shape == (1,)
    assert result.axes[0].get_xscale() == result.axes[0].get_yscale() == "log"
    assert result.axes[0].get_xlabel() == "exposure (d)"
    np.testing.assert_array_equal(result.artists["bound"][0].get_ydata(), [0.3, 0.4])
    assert result.artists["bound"][0].get_linewidth() == 2
    assert result.artists["line"][0].get_linewidth() == 3
    assert result.artists["lines"][0].get_linewidth() == 4
    assert line_kw == {"c": "black", "lw": 3}
    assert ref_kw == {"lw": 2} and interval_kw == {"lw": 4}


def test_yield_sweep_refuses_missing_stored_intervals(sweep_reports):
    report = replace(sweep_reports[0], intervals={})
    with pytest.raises(ValueError, match=r'independent-lower-bound.*interval="none"'):
        plot_yield_sweep([1], [report])


@pytest.mark.parametrize(
    "options, match",
    [
        ({"bound": "profile"}, "Profiles"),
        ({"bound": [0.1]}, "expected shape"),
        ({"zero_floor": 0}, "positive"),
        ({"log_x": True}, "positive numeric"),
        ({"channels": ()}, "nonempty subset"),
        ({"channels": ("unknown",)}, "nonempty subset"),
        ({"interval": "unknown"}, "interval must"),
    ],
)
def test_yield_sweep_invalid_inputs(options, match):
    with pytest.raises(ValueError, match=match):
        plot_yield_sweep([0, 1], {"y_issued": [2, 3], "y_wrong": [0, 0.1]}, **options)


@pytest.fixture
def comparison_report():
    base = certified_yield(make_outcome(), [1, 2, 3], certificate="example")
    return replace(
        base,
        y_cert_pos=2.0,
        intervals={"y_cert_pos": (-8.125, 12.5)},
        total_intervals={"y_cert_pos": (-9.75, 15.25)},
        standard_errors={"y_cert_pos": 999.0},
    )


@pytest.fixture
def comparison_difference():
    return PairedDifference(
        yields={"y_cert_pos": 2.0, "y_wrong": 0.0, "y_neg": -1.0},
        replicate_yields={},
        standard_errors=dict.fromkeys(("y_cert_pos", "y_wrong", "y_neg"), 999.0),
        intervals={
            "y_cert_pos": (-7.5, 11.25),
            "y_wrong": (-0.125, 0.75),
            "y_neg": (-9.5, 3.25),
        },
        interval_kind="replicate-batch",
        replicate_key="example",
        confidence=0.95,
    )


def test_plot_yield_comparison_smoke(comparison_difference, save_viz_png):
    result = plot_yield_comparison(
        {"scored": (3, 2.5, 3.5), "prediction": 4, "pending": None},
        references={"scored": 3.2, "pending": 2},
        margin=0.5,
        predicted=("prediction",),
        difference=comparison_difference,
        gap=("scored", "prediction"),
    )
    assert isinstance(result, ep.MosaicResult)
    assert result.axes.shape == (2,)
    assert result.update is None
    assert set(result.artists) == {
        "scatter",
        "lines",
        "fill",
        "reference",
        "line",
        "bracket",
        "text",
    }
    assert len(result.artists["scatter"]) == 5
    assert len(result.artists["fill"]) == 5
    assert result.artists["text"][0].get_text() == "gap = 1"
    np.testing.assert_array_equal(
        result.artists["bracket"][0].get_xdata(), [3, 3, 4, 4]
    )
    result.fig.set_size_inches(8, 3)
    result.fig.canvas.draw()
    save_viz_png(result, "plot_yield_comparison")


@pytest.mark.parametrize("interval", ["random", "total", "none"])
def test_yield_comparison_reads_report_intervals(comparison_report, interval):
    field = "total_intervals" if interval == "total" else "intervals"
    report = comparison_report
    bare = np.array([report.y_cert_pos, *getattr(report, field)["y_cert_pos"]])
    typed = plot_yield_comparison(
        {"plan": report}, references={"plan": report}, interval=interval
    )
    arrays = plot_yield_comparison(
        {"plan": bare}, references={"plan": report.y_cert_pos}, interval=interval
    )
    np.testing.assert_array_equal(
        typed.artists["scatter"][0].get_offsets(),
        arrays.artists["scatter"][0].get_offsets(),
    )
    for key in ("lines", "reference"):
        for a, b in zip(typed.artists[key], arrays.artists[key], strict=True):
            np.testing.assert_array_equal(a.get_xydata(), b.get_xydata())
    if interval == "none":
        assert typed.artists["lines"] == []
    else:
        (line,) = typed.artists["lines"]
        np.testing.assert_array_equal(
            line.get_xdata(), getattr(report, field)["y_cert_pos"]
        )
        np.testing.assert_array_equal(line.get_ydata(), [0, 0])


@pytest.mark.parametrize("interval", ["random", "none"])
def test_yield_comparison_reads_paired_intervals(comparison_difference, interval):
    diff = comparison_difference
    channels = tuple(diff.yields)
    typed = plot_yield_comparison({"plan": 3}, difference=diff, interval=interval)
    assert len(typed.artists["lines"]) == (3 if interval == "random" else 0)
    bare = plot_yield_comparison(
        {"plan": 3},
        difference={
            name: (diff.yields[name], *diff.intervals[name]) for name in channels
        },
        interval=interval,
    )
    for a, b in zip(typed.artists["scatter"], bare.artists["scatter"], strict=True):
        np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
    for index, (a, b) in enumerate(
        zip(typed.artists["lines"], bare.artists["lines"], strict=True)
    ):
        np.testing.assert_array_equal(a.get_xydata(), b.get_xydata())
        np.testing.assert_array_equal(a.get_xdata(), diff.intervals[channels[index]])
    if interval == "none":
        assert typed.artists["lines"] == []
    (zero,) = typed.artists["line"]
    np.testing.assert_array_equal(zero.get_xdata(), [0, 0])
    assert zero.get_transform() == typed.axes[1].get_xaxis_transform()
    assert [t.get_text() for t in typed.axes[1].get_yticklabels()] == [
        "certified positive",
        "wrong among issued",
        "certified negative",
    ]


def test_yield_comparison_refuses_empty_intervals(
    comparison_report, comparison_difference
):
    report = replace(comparison_report, intervals={})
    with pytest.raises(ValueError, match=r'independent-lower-bound.*interval="none"'):
        plot_yield_comparison({"plan": report})
    assert (
        plot_yield_comparison({"plan": report}, interval="none").artists["lines"] == []
    )
    diff = replace(comparison_difference, intervals={})
    with pytest.raises(ValueError, match=r'replicate-batch.*interval="none"'):
        plot_yield_comparison({"plan": 3}, difference=diff)
    with pytest.raises(
        ValueError, match=r"total interval unavailable.*replicate-batch"
    ):
        plot_yield_comparison(
            {"plan": 3}, difference=comparison_difference, interval="total"
        )


@pytest.mark.parametrize("with_difference", [False, True])
@pytest.mark.parametrize("layout", [None, "constrained"])
def test_yield_comparison_panel_count_and_caller_geometry(
    with_difference, layout, comparison_difference
):
    diff = comparison_difference if with_difference else None
    count = 2 if with_difference else 1
    own = plot_yield_comparison({"plan": 3}, difference=diff)
    assert own.axes.shape == (count,)
    assert len(own.fig.axes) == count
    fig, axes = plt.subplots(1, 3, layout=layout)
    engine = fig.get_layout_engine()
    positions = [ax.get_position(original=True).bounds for ax in axes]
    result = plot_yield_comparison({"plan": 3}, difference=diff, axes=axes[:count])
    assert all(a is b for a, b in zip(result.axes, axes[:count], strict=True))
    assert fig.get_layout_engine() is engine
    for ax, position in zip(axes, positions, strict=True):
        np.testing.assert_array_equal(ax.get_position(original=True).bounds, position)
        assert ax.get_shared_x_axes().get_siblings(ax) == [ax]
        assert ax.get_shared_y_axes().get_siblings(ax) == [ax]
    assert not axes[2].collections and not axes[2].lines


@pytest.mark.parametrize(
    "with_difference, shape", [(False, (2,)), (False, ()), (True, (1,)), (True, (1, 2))]
)
def test_yield_comparison_axes_shape_error(
    with_difference, shape, comparison_difference
):
    _, ax = plt.subplots()
    with pytest.raises(ValueError) as error:
        plot_yield_comparison(
            {"plan": 3},
            difference=comparison_difference if with_difference else None,
            axes=np.full(shape, ax, dtype=object),
        )
    assert f"expected shape {(2,) if with_difference else (1,)}" in str(error.value)
    assert f"received {shape}" in str(error.value)


def test_yield_comparison_rows_bands_and_roles(comparison_difference):
    result = plot_yield_comparison(
        {"prediction": 4, "pending": None, "scored": 0},
        references={"prediction": 3, "pending": 2},
        predicted=("prediction",),
        margin={"prediction": 0.5, "pending": 0.25, "y_wrong": 0.125},
        difference=comparison_difference,
    )
    assert [t.get_text() for t in result.axes[0].get_yticklabels()] == [
        "prediction",
        "pending",
        "scored",
    ]
    assert result.axes[0].yaxis_inverted()
    points = result.artists["scatter"]
    np.testing.assert_array_equal(points[0].get_offsets(), [[4, 0]])
    np.testing.assert_array_equal(points[1].get_offsets(), [[0, 2]])
    for point, role in zip(
        points, ["model", "data", "data", "alert", "data"], strict=True
    ):
        np.testing.assert_allclose(point.get_edgecolors()[0], to_rgba(ep.role(role)))
    assert points[1].get_facecolors()[0, 3] == points[3].get_facecolors()[0, 3] == 0
    for tick, x, row in zip(result.artists["reference"], [3, 2], [0, 1], strict=True):
        np.testing.assert_array_equal(tick.get_xydata(), [[x, row]])
        assert to_rgba(tick.get_color()) == to_rgba(ep.neutral(0.6))
    for band, bounds in zip(
        result.artists["fill"], [(2.5, 3.5), (1.75, 2.25), (-0.125, 0.125)], strict=True
    ):
        vertices = band.get_paths()[0].vertices
        np.testing.assert_array_equal(
            [vertices[:, 0].min(), vertices[:, 0].max()], bounds
        )
        np.testing.assert_allclose(band.get_facecolors()[0], to_rgba(ep.neutral(0.15)))


def test_yield_comparison_routed_styles_are_not_mutated():
    point_kw = {"s": 70, "marker": "s"}
    interval_kw, ref_kw, band_kw = {"lw": 4}, {"lw": 3}, {"alpha": 0.25}
    result = plot_yield_comparison(
        {"plan": (2, 1, 3)},
        references={"plan": 2.5},
        margin=0.5,
        difference={"y_wrong": (0, -0.1, 0.2)},
        difference_channels=("y_wrong",),
        point_kw=point_kw,
        interval_kw=interval_kw,
        ref_kw=ref_kw,
        band_kw=band_kw,
    )
    assert result.artists["scatter"][0].get_sizes()[0] == 70
    assert result.artists["lines"][0].get_linewidth() == 4
    assert (
        result.artists["reference"][0].get_linewidth()
        == result.artists["line"][0].get_linewidth()
        == 3
    )
    assert result.artists["fill"][0].get_alpha() == 0.25
    assert point_kw == {"s": 70, "marker": "s"}
    assert (
        interval_kw == {"lw": 4} and ref_kw == {"lw": 3} and band_kw == {"alpha": 0.25}
    )


@pytest.fixture
def visit_plan():
    from tests.test_plan import make_plan

    return make_plan(
        n_visits=3,
        visit_epochs_d=[[0, 10, np.nan], [2, 12, 22], [4, np.nan, np.nan]],
        visit_success=[[True, False, True], [False, True, True], [True, False, False]],
        spectrum_visit=[1, 2, -1],
        visit_photometric_snr=[[3, 0, 99], [0, 6, 9], [12, 0, 0]],
        k_precision_ms=[1, np.inf, 2],
        astrometric_precision_mas=[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
    )


@pytest.fixture
def observed_data():
    from spaceodyssey.truth import ObservedData

    return ObservedData(
        spectrum=[[0, 1], [2, 3], [0, 0]],
        has_spectrum=[True, True, False],
        photometry=np.ones((3, 3)),
        astrometry_mas=[
            [[10, 5], [0, 0], [20, -5]],
            [[30, 2], [40, 4], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ],
        visit_usable=[[True, False, True], [True, True, False], [False] * 3],
        rv_semi_amplitude_ms=np.zeros(3),
    )


def test_plot_plan_visits_smoke(visit_plan, save_viz_png):
    from matplotlib.markers import MarkerStyle

    from spaceodyssey.viz import plot_plan_visits

    result = plot_plan_visits(visit_plan)
    assert isinstance(result, ep.PlotResult)
    assert result.update is None
    assert set(result.artists) == {"scatter", "cbar", "text"}
    good, failed = result.artists["scatter"]
    np.testing.assert_array_equal(
        good.get_offsets(), [[0, 0], [12, 1], [22, 1], [4, 2]]
    )
    np.testing.assert_array_equal(failed.get_offsets(), [[10, 0], [2, 1]])
    np.testing.assert_array_equal(good.get_array(), [3, 6, 9, 12])
    square = MarkerStyle("s")
    vertices = square.get_path().transformed(square.get_transform()).vertices
    count = sum(
        np.array_equal(path.vertices, vertices)
        for scatter in (good, failed)
        for path in scatter.get_paths()
    )
    assert count == (np.asarray(visit_plan.spectrum_visit) >= 0).sum() == 2
    assert (
        sum(len(s.get_offsets()) for s in (good, failed))
        == np.isfinite(visit_plan.visit_epochs_d).sum()
    )
    assert failed.get_facecolors().size == 0
    np.testing.assert_allclose(failed.get_edgecolors()[0], to_rgba(ep.neutral(0.6)))
    assert [t.get_position() for t in result.artists["text"]] == [(0.98, 0), (0.98, 2)]
    assert result.ax.yaxis_inverted()
    assert [t.get_text() for t in result.ax.get_yticklabels()] == [
        "planet 0",
        "planet 1",
        "planet 2",
    ]
    assert len(result.insets) == 1
    result.fig.set_size_inches(5, 3)
    result.fig.canvas.draw()
    assert result.insets[0].get_position().x0 > result.ax.get_position().x1
    save_viz_png(result, "plot_plan_visits")


def test_plan_visits_selection_bare_arrays_and_styles(visit_plan):
    from spaceodyssey.viz import plot_plan_visits

    bare = {
        name: np.asarray(getattr(visit_plan, name))
        for name in (
            "visit_epochs_d",
            "visit_success",
            "visit_photometric_snr",
            "spectrum_visit",
            "k_precision_ms",
        )
    }
    style = {"s": 70, "alpha": 0.8}
    options = dict(
        planets=[2, 0],
        labels=["outer", "inner"],
        snr_vlim=(0, 20),
        scatter_kw=style,
        cbar_kw={"label": "SNR"},
    )
    typed = plot_plan_visits(visit_plan, **options)
    arrays = plot_plan_visits(bare, **options)
    for a, b in zip(typed.artists["scatter"], arrays.artists["scatter"], strict=True):
        np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
        for pa, pb in zip(a.get_paths(), b.get_paths(), strict=True):
            np.testing.assert_array_equal(pa.vertices, pb.vertices)
        assert a.get_sizes()[0] == 70
    assert typed.artists["scatter"][0].get_clim() == (0, 20)
    assert typed.artists["cbar"].ax.get_ylabel() == "SNR"
    assert [t.get_text() for t in typed.ax.get_yticklabels()] == ["outer", "inner"]
    assert style == {"s": 70, "alpha": 0.8}


@pytest.mark.parametrize(
    "epochs, success", [([[np.nan]], [[True]]), ([[1]], [[False]])]
)
def test_plan_visits_empty_success_and_padded_spectrum(epochs, success):
    from spaceodyssey.viz import plot_plan_visits

    result = plot_plan_visits(
        dict(
            visit_epochs_d=epochs,
            visit_success=success,
            visit_photometric_snr=[[0]],
            spectrum_visit=[0],
        )
    )
    assert len(result.artists["scatter"][0].get_offsets()) == 0
    assert len(result.artists["scatter"][1].get_offsets()) == int(
        np.isfinite(epochs[0][0])
    )
    result.fig.canvas.draw()


def test_plot_observed_data_smoke(observed_data, visit_plan, save_viz_png):
    from spaceodyssey.viz import plot_observed_data

    result = plot_observed_data(
        observed_data, visit_plan, expected=[0.1, 0.9], astrometric_corr=0.5, n_sigma=2
    )
    assert isinstance(result, ep.MosaicResult)
    assert result.axes.shape == (2,) and result.update is None
    assert set(result.artists) == {
        "scatter",
        "ellipse",
        "star",
        "lines",
        "line",
        "text",
    }
    usable = np.asarray(observed_data.visit_usable[0])
    positions = np.asarray(observed_data.astrometry_mas[0])[usable]
    np.testing.assert_array_equal(result.artists["scatter"][0].get_offsets(), positions)
    assert len(result.artists["ellipse"]) == usable.sum() == 2
    for ellipse, mean, sigma in zip(
        result.artists["ellipse"], positions, [1, 3], strict=True
    ):
        np.testing.assert_array_equal(ellipse.center, mean)
        # Recover the covariance from the actual ellipse geometry.
        angle = np.deg2rad(ellipse.angle)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        covariance = (
            rotation
            @ np.diag((np.array([ellipse.width, ellipse.height]) / 4) ** 2)
            @ rotation.T
        )
        np.testing.assert_allclose(
            covariance, sigma**2 * np.array([[1, 0.5], [0.5, 1]])
        )
        np.testing.assert_allclose(ellipse.get_edgecolor(), to_rgba(ep.role("data")))
    assert result.axes[0].get_aspect() == 1
    np.testing.assert_array_equal(result.artists["star"].get_offsets(), [[0, 0]])
    np.testing.assert_allclose(
        result.artists["star"].get_facecolors()[0], to_rgba(ep.role("star"))
    )
    np.testing.assert_allclose(
        result.artists["scatter"][1].get_offsets(), [[0.5, 0], [0.9, 1]]
    )
    for line, wave, flux in zip(
        result.artists["lines"], [0.5, 0.9], [0, 1], strict=True
    ):
        np.testing.assert_allclose(
            line.get_xydata(), [[wave, flux - 0.2], [wave, flux + 0.2]]
        )
    for scatter in result.artists["scatter"]:
        np.testing.assert_allclose(
            scatter.get_facecolors()[0], to_rgba(ep.role("data"))
        )
    np.testing.assert_allclose(
        result.artists["line"][0].get_xydata(), [[0.5, 0.1], [0.9, 0.9]]
    )
    assert to_rgba(result.artists["line"][0].get_color()) == to_rgba(ep.neutral(0.6))
    result.fig.set_size_inches(7, 3)
    result.fig.canvas.draw()
    save_viz_png(result, "plot_observed_data")


def test_observed_data_bare_and_per_planet_wavelengths(observed_data, visit_plan):
    from spaceodyssey.viz import plot_observed_data

    bare = {
        name: np.asarray(getattr(observed_data, name))
        for name in ("astrometry_mas", "visit_usable", "spectrum", "has_spectrum")
    }
    plan = {
        name: np.asarray(getattr(visit_plan, name))
        for name in ("astrometric_precision_mas", "noise_sigma", "bin_wavelength_nm")
    }
    typed = plot_observed_data(observed_data, visit_plan, planet=1)
    plan["bin_wavelength_nm"] = np.tile(plan["bin_wavelength_nm"], (3, 1))
    arrays = plot_observed_data(bare, plan, planet=1)
    for key in ("scatter", "lines"):
        for a, b in zip(typed.artists[key], arrays.artists[key], strict=True):
            method = "get_offsets" if key == "scatter" else "get_xydata"
            np.testing.assert_array_equal(getattr(a, method)(), getattr(b, method)())
    for a, b in zip(typed.artists["ellipse"], arrays.artists["ellipse"], strict=True):
        np.testing.assert_array_equal(a.center, b.center)
        assert (a.width, a.height, a.angle) == (b.width, b.height, b.angle)
    plan["bin_wavelength_nm"][1] = [600, 800]
    changed = plot_observed_data(bare, plan, planet=1)
    np.testing.assert_allclose(
        changed.artists["scatter"][1].get_offsets()[:, 0], [0.6, 0.8]
    )


def test_observed_data_empty_and_routed_styles(observed_data, visit_plan):
    from spaceodyssey.viz import plot_observed_data

    empty = plot_observed_data(observed_data, visit_plan, planet=2)
    assert not empty.artists["ellipse"] and not empty.artists["lines"]
    assert all(len(s.get_offsets()) == 0 for s in empty.artists["scatter"])
    assert empty.artists["text"][0].get_text() == "no spectrum"
    point_kw, ellipse_kw = {"s": 70, "c": "black"}, {"lw": 3}
    interval_kw, line_kw = {"lw": 4}, {"c": "black", "lw": 2}
    result = plot_observed_data(
        observed_data,
        visit_plan,
        expected=[0, 1],
        point_kw=point_kw,
        ellipse_kw=ellipse_kw,
        interval_kw=interval_kw,
        line_kw=line_kw,
    )
    assert all(s.get_sizes()[0] == 70 for s in result.artists["scatter"])
    assert result.artists["ellipse"][0].get_linewidth() == 3
    assert result.artists["lines"][0].get_linewidth() == 4
    assert result.artists["line"][0].get_linewidth() == 2
    assert point_kw == {"s": 70, "c": "black"} and ellipse_kw == {"lw": 3}
    assert interval_kw == {"lw": 4} and line_kw == {"c": "black", "lw": 2}


@pytest.mark.parametrize("plot", ["plan", "observed"])
@pytest.mark.parametrize("layout", [None, "constrained"])
def test_plan_plots_preserve_caller_geometry(plot, layout, observed_data, visit_plan):
    from spaceodyssey.viz import plot_observed_data, plot_plan_visits

    fig, axes = plt.subplots(1, 3, layout=layout)
    engine = fig.get_layout_engine()
    positions = [ax.get_position(original=True).bounds for ax in axes]
    if plot == "plan":
        result = plot_plan_visits(visit_plan, ax=axes[0])
        assert result.ax is axes[0]
    else:
        result = plot_observed_data(observed_data, visit_plan, axes=axes[:2])
        assert all(a is b for a, b in zip(result.axes, axes[:2], strict=True))
    assert fig.get_layout_engine() is engine
    for ax, position in zip(axes, positions, strict=True):
        np.testing.assert_array_equal(ax.get_position(original=True).bounds, position)
        assert ax.get_shared_x_axes().get_siblings(ax) == [ax]
        assert ax.get_shared_y_axes().get_siblings(ax) == [ax]
    assert not axes[2].collections and not axes[2].lines


@pytest.mark.parametrize("shape", [(), (1,), (1, 2)])
def test_observed_axes_shape_errors(shape, observed_data, visit_plan):
    from spaceodyssey.viz import plot_observed_data

    _, ax = plt.subplots()
    with pytest.raises(ValueError) as error:
        plot_observed_data(
            observed_data, visit_plan, axes=np.full(shape, ax, dtype=object)
        )
    assert "expected shape (2,)" in str(error.value)
    assert f"received {shape}" in str(error.value)


def test_plan_axes_shape_error(visit_plan):
    from spaceodyssey.viz import plot_plan_visits

    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match=r"expected shape \(\), received \(2,\)"):
        plot_plan_visits(visit_plan, ax=axes)


def test_plot_planet_plane_smoke(curve_outcome, save_viz_png):
    result = plot_planet_plane(
        [0.5, 1, 1.5, 2],
        [0.5, 1, 1.5, 2],
        values=curve_outcome,
        profile=EXOEARTH_CANDIDATE,
        marked=[False, True, False, False],
        truth=(1.2, 1.1),
    )
    assert isinstance(result, ep.PlotResult) and result.update is None
    assert set(result.artists) == {
        "scatter",
        "collection",
        "cbar",
        "region_skipped",
        "marked",
        "truth",
    }
    assert result.artists["region_skipped"] == ("water",)
    assert result.ax.get_xscale() == result.ax.get_yscale() == "log"
    assert result.artists["cbar"].ax.get_ylabel() == "P(certified positive)"
    assert len(result.insets) == 1
    assert result.insets[0] in result.ax.child_axes
    result.fig.set_size_inches(5, 3)
    result.fig.canvas.draw()
    assert result.insets[0].get_position().x0 > result.ax.get_position().x1
    np.testing.assert_array_equal(result.artists["marked"].get_offsets(), [[1, 1]])
    np.testing.assert_allclose(result.artists["truth"].get_offsets(), [[1.2, 1.1]])
    np.testing.assert_allclose(
        result.artists["marked"].get_edgecolors()[0], to_rgba(ep.neutral(1))
    )
    np.testing.assert_allclose(
        result.artists["truth"].get_facecolors()[0], to_rgba(ep.neutral(0.6))
    )
    save_viz_png(result, "plot_planet_plane")


@pytest.mark.parametrize(
    "active", [[True] * 4, [True, False, True, False], [False] * 4]
)
@pytest.mark.parametrize("cloud", [False, True])
def test_plane_retains_every_planet_and_triangle(curve_outcome, active, cloud):
    from matplotlib.markers import MarkerStyle

    xy = np.array([[0.5, 0.8], [1, 1], [1.5, 1.2], [2, 1.4]])
    result = plot_planet_plane(
        *xy.T,
        values=None if cloud else curve_outcome,
        active=active,
        in_range=curve_outcome.in_range,
        marked=[True] * 4,
    )
    active = np.asarray(active)
    triangle = MarkerStyle("^")
    vertices = triangle.get_path().transformed(triangle.get_transform()).vertices
    count = 0
    total = 0
    for collection, selected in zip(
        result.artists["scatter"], (~active, active), strict=True
    ):
        np.testing.assert_allclose(collection.get_offsets(), xy[selected])
        paths = collection.get_paths()
        triangles = [np.array_equal(p.vertices, vertices) for p in paths]
        np.testing.assert_array_equal(
            triangles, ~np.asarray(curve_outcome.in_range)[selected]
        )
        total += len(collection.get_offsets())
        count += sum(triangles)
    assert total == 4 and count == 2  # Seven replicates never add planets.
    inactive, points = result.artists["scatter"]
    np.testing.assert_allclose(inactive.get_facecolors()[0], to_rgba(ep.neutral(0.3)))
    assert inactive.get_zorder() < points.get_zorder()
    if cloud:
        assert result.insets == () and result.artists["cbar"] is None
        np.testing.assert_allclose(
            points.get_facecolors()[0], to_rgba(ep.role("model"), alpha=0.2)
        )
    else:
        np.testing.assert_array_equal(
            points.get_array(), np.asarray(curve_outcome.p_pos)[active]
        )
        assert points.get_clim() == (0, 1)
    result.fig.canvas.draw()


def test_plane_region_contour_known_answers():
    result = plot_planet_plane([0.4, 2.5], [0.4, 2], profile=EXOEARTH_CANDIDATE)
    contour = result.artists["collection"]
    (path,) = contour.get_paths()
    np.testing.assert_array_equal(
        path.contains_points([[1, 1], [0.5, 1], [1, 1.5]]), [True, False, False]
    )
    vertices = path.vertices
    np.testing.assert_allclose(
        [vertices[:, 0].min(), vertices[:, 0].max()], [0.95, 1.67], rtol=0.006
    )
    assert vertices[:, 1].max() == pytest.approx(1.4, rel=0.006)
    for distance in (0.95, 1.67):
        nearby = vertices[np.abs(vertices[:, 0] - distance) < 0.012]
        assert nearby[:, 1].min() == pytest.approx(0.8 / np.sqrt(distance), abs=0.006)
    np.testing.assert_allclose(contour.get_edgecolors()[0], to_rgba(ep.neutral(0.6)))


def test_plane_bare_arrays_override_and_routed_styles(curve_outcome):
    x, y = [0.5, 1, 1.5, 2], [0.5, 1, 1.5, 2]
    scatter_kw, region_kw, cbar_kw = (
        {"s": 70, "marker": "s"},
        {"linewidths": 3},
        {"label": "score"},
    )
    options = dict(
        in_range=[True, True, False, True],
        vlim=(0.1, 0.9),
        scatter_kw=scatter_kw,
        region_kw=region_kw,
        cbar_kw=cbar_kw,
        profile=EXOEARTH_CANDIDATE,
    )
    typed = plot_planet_plane(x, y, values=curve_outcome, **options)
    arrays = plot_planet_plane(x, y, values=np.asarray(curve_outcome.p_pos), **options)
    for a, b in zip(typed.artists["scatter"], arrays.artists["scatter"], strict=True):
        np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
        for pa, pb in zip(a.get_paths(), b.get_paths(), strict=True):
            np.testing.assert_array_equal(pa.vertices, pb.vertices)
    assert typed.artists["scatter"][1].get_clim() == (0.1, 0.9)
    assert typed.artists["scatter"][1].get_sizes()[0] == 70
    assert typed.artists["collection"].get_linewidths()[0] == 3
    assert typed.artists["cbar"].ax.get_ylabel() == "score"
    assert scatter_kw == {"s": 70, "marker": "s"}
    assert region_kw == {"linewidths": 3} and cbar_kw == {"label": "score"}


@pytest.mark.parametrize("layout", [None, "constrained"])
def test_plane_preserves_caller_geometry(layout):
    fig, axes = plt.subplots(1, 2, layout=layout)
    positions = [a.get_position(original=True).bounds for a in axes]
    engine = fig.get_layout_engine()
    axes[0].set_xlim(0.4, 3)
    axes[0].set_ylim(0.4, 2)
    result = plot_planet_plane(
        [0.5, 2], [0.5, 2], values=[0, 1], profile=EXOEARTH_CANDIDATE, ax=axes[0]
    )
    assert result.ax is axes[0] and fig.get_layout_engine() is engine
    assert axes[0].get_xlim() == (0.4, 3) and axes[0].get_ylim() == (0.4, 2)
    for ax, position in zip(axes, positions, strict=True):
        np.testing.assert_array_equal(ax.get_position(original=True).bounds, position)
        assert ax.get_shared_x_axes().get_siblings(ax) == [ax]
        assert ax.get_shared_y_axes().get_siblings(ax) == [ax]
    assert not axes[1].collections and not axes[1].child_axes


def test_plane_axes_shape_error():
    _, axes = plt.subplots(1, 2)
    with pytest.raises(ValueError, match=r"expected shape \(\), received \(2,\)"):
        plot_planet_plane([1], [1], ax=axes)


@pytest.mark.parametrize("profile", [None, WATER_ONLY, EXOEARTH_CANDIDATE])
def test_plane_empty_and_uniform_regions(profile):
    result = plot_planet_plane([], [], values=[], profile=profile)
    assert sum(len(s.get_offsets()) for s in result.artists["scatter"]) == 0
    if profile is not EXOEARTH_CANDIDATE:
        assert result.artists["collection"] is None
    assert result.artists["region_skipped"] == (() if profile is None else ("water",))
    result.fig.canvas.draw()


def test_plane_region_absent_when_view_is_outside():
    _, ax = plt.subplots()
    ax.set_xlim(3, 4)
    ax.set_ylim(2, 3)
    result = plot_planet_plane([3.5], [2.5], profile=EXOEARTH_CANDIDATE, ax=ax)
    assert result.artists["collection"] is None
    assert result.artists["region_skipped"] == ("water",)


def test_plane_display_override():
    from spaceodyssey.viz import QuantityDisplay

    display = {
        "ln_a_eeid": QuantityDisplay("log distance ratio", "identity", "linear"),
        "ln_rp_re": QuantityDisplay("log radius", "identity", "linear"),
    }
    result = plot_planet_plane(
        [0.5, 1, 2],
        [0.5, 1, 2],
        truth=(1.2, 1.1),
        display=display,
        profile=EXOEARTH_CANDIDATE,
    )
    np.testing.assert_allclose(
        result.artists["scatter"][1].get_offsets(), np.log([[0.5, 0.5], [1, 1], [2, 2]])
    )
    np.testing.assert_allclose(
        result.artists["truth"].get_offsets(), np.log([[1.2, 1.1]])
    )
    assert result.ax.get_xscale() == result.ax.get_yscale() == "linear"
    assert result.ax.get_xlabel() == "log distance ratio"
    assert result.artists["collection"].get_paths()[0].contains_point((0, 0))


@pytest.mark.parametrize(
    "options, match",
    [
        ({"truth": (0, 1)}, "finite and positive"),
        ({"truth": (1,)}, "expected shape"),
        ({"vlim": (1, 0)}, "finite and increasing"),
        ({"vlim": (0, np.inf)}, "finite and increasing"),
    ],
)
def test_plane_invalid_plot_arguments(options, match):
    with pytest.raises(ValueError, match=match):
        plot_planet_plane([1], [1], **options)
