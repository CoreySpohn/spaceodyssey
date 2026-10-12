"""Plot contracts and numerical artist regressions, without image baselines."""

from dataclasses import replace

import eyepiece as ep
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from spaceodyssey.profiles import ALPHA_BAND
from spaceodyssey.reporting import certified_yield
from spaceodyssey.viz import plot_yield_sweep
from tests.test_plan import make_outcome


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


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
