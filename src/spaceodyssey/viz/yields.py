"""Expected planet yield comparisons and parameter sweeps."""

import numpy as np

from spaceodyssey.viz import _extract
from spaceodyssey.viz._axes import decimal_log_ticks
from spaceodyssey.viz._legend import scatter_key
from spaceodyssey.viz._require import eyepiece
from spaceodyssey.viz.labels import CHANNEL_LABELS

SCENERY_LEVEL = 0.6
WRONG_YIELD_ROLE = "alert"
COMPARISON_ROLES = {"predicted": "model", "scored": "data"}
MARGIN_SCENERY_LEVEL = 0.15


def plot_yield_comparison(
    rows,
    *,
    channel="y_cert_pos",
    references=None,
    margin=None,
    predicted=(),
    difference=None,
    difference_channels=("y_cert_pos", "y_wrong", "y_neg"),
    interval="random",
    gap=None,
    axes=None,
    point_kw=None,
    interval_kw=None,
    band_kw=None,
    ref_kw=None,
):
    """Compare expected planet counts with stored uncertainty and reference bands.

    Args:
        rows: Nonempty mapping of plan names to CertifiedYield reports, bare
            floats, (point, lo, hi) arrays, or None. Insertion order runs from
            top to bottom; None retains its label without an estimate.
        channel: Yield channel shown in the first panel.
        references: Mapping of row names to reference floats or CertifiedYield
            reports. Reference intervals are ignored. Missing keys omit ticks.
        margin: Nonnegative half-width of equivalence bands. A scalar applies
            to all references and difference channels. A mapping uses row names
            in panel one and channel names in panel two; missing keys omit bands.
            No equivalence verdict is computed.
        predicted: Row names drawn in the model role; scored rows use data.
        difference: PairedDifference, or a bare mapping of channels to floats or
            (point, lo, hi) arrays. Adds a second panel only when supplied.
            Wrong channels use alert; other differences use data.
        difference_channels: Channels shown in the second panel, in row order.
        interval: ``random`` reads intervals, ``total`` reads total_intervals,
            and ``none`` omits intervals. Endpoints are never recomputed. Missing
            intervals raise with interval_kind and the remedy interval="none".
            PairedDifference has no total_intervals, so total raises for it.
        gap: Optional (row_a, row_b) names. A bracket below the first panel's
            rows labels the signed point difference row_b minus row_a.
        axes: Axes array shaped (1,), or (2,) when difference is supplied.
            None creates a row of panels. Caller layout and sharing are retained.
        point_kw: Overrides passed to interval_points for ax.scatter.
        interval_kw: Overrides passed to interval_points for interval lines.
        band_kw: Overrides passed to ax.fill_betweenx for per-row margin bands.
        ref_kw: Line2D overrides for reference ticks and the difference zero line.

    Returns:
        MosaicResult with update=None. Artists contain ``scatter`` (one estimate
        collection per nonmissing row), ``lines`` (stored intervals), ``fill``
        (margin bands), ``reference`` (ticks), ``line`` (the zero line),
        ``bracket`` (optional gap line), and ``text`` (gap label). Lists follow
        panel then row order; missing estimates contribute no point or interval.

    Raises:
        ValueError: If axes shapes, channels, keys, margins, or intervals are
            invalid. Axes errors name both expected and received shapes.

    Example::

        from spaceodyssey.viz import plot_yield_comparison
        result = plot_yield_comparison(
            {"plan A": (3, 2.5, 3.6), "plan B": 4},
            references={"plan A": 3.2}, margin=0.5, predicted=("plan B",),
        )
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D

    predicted = tuple(predicted)
    panels, gap_values = _extract.yield_comparison_data(
        rows,
        channel,
        references=references,
        margin=margin,
        predicted=predicted,
        difference=difference,
        difference_channels=difference_channels,
        interval=interval,
        gap=gap,
    )
    shape = (len(panels),)
    if axes is None:
        _, axes = plt.subplots(1, len(panels), squeeze=False, layout="constrained")
        axes = axes[0]
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != shape:
            raise ValueError(f"axes: expected shape {shape}, received {axes.shape}")
    artists = {
        key: []
        for key in ("scatter", "lines", "fill", "reference", "line", "bracket", "text")
    }
    ref_options = normalize_kwargs(ref_kw, Line2D)
    for panel_index, (ax, panel) in enumerate(zip(axes, panels, strict=True)):
        for row, (name, (point, lo, hi, ref, width)) in enumerate(panel.items()):
            if ref is not None and width is not None:
                artists["fill"].append(
                    ax.fill_betweenx(
                        [row - 0.35, row + 0.35],
                        ref - width,
                        ref + width,
                        **{
                            "color": ep.neutral(MARGIN_SCENERY_LEVEL),
                            "zorder": 0,
                            **(band_kw or {}),
                        },
                    )
                )
            if panel_index == 0 and ref is not None:
                artists["reference"].extend(
                    ax.plot(
                        [ref],
                        [row],
                        **{
                            "color": ep.neutral(SCENERY_LEVEL),
                            "marker": "|",
                            "markersize": 12,
                            "linestyle": "none",
                            **ref_options,
                        },
                    )
                )
            if np.isnan(point):
                continue
            role = COMPARISON_ROLES[
                "predicted" if panel_index == 0 and name in predicted else "scored"
            ]
            if panel_index == 1 and name in ("y_wrong", "y_neg_wrong"):
                role = WRONG_YIELD_ROLE
            result = ep.interval_points(
                [row],
                [point],
                lo,
                hi,
                ax=ax,
                orient="h",
                color=ep.role(role),
                point_kw=point_kw,
                interval_kw=interval_kw,
            )
            artists["scatter"].append(result.artists["scatter"])
            artists["lines"].extend(result.artists["lines"])
        if panel_index == 1:
            result = ep.ref_line(
                ax, 0, orient="v", color=ep.neutral(SCENERY_LEVEL), line_kw=ref_kw
            )
            artists["line"].append(result.artists["line"])
        ax.set_yticks(
            range(len(panel)),
            list(panel) if panel_index == 0 else [CHANNEL_LABELS[n] for n in panel],
        )
        ax.set_ylim(len(panel) - 0.5, -0.5)
        ax.set_xlabel(
            f"Expected planets ({CHANNEL_LABELS[channel]})"
            if panel_index == 0
            else "Difference in expected planets"
        )
    if gap_values is not None:
        left, right, delta = gap_values
        result = ep.bracket(
            axes[0],
            left,
            right,
            len(rows),
            f"gap = {delta:g}",
            depth=0.15,
            color=ep.neutral(SCENERY_LEVEL),
        )
        artists["bracket"].append(result.artists["line"])
        artists["text"].append(result.artists["text"])
        axes[0].set_ylim(len(rows) + 0.5, -0.5)
    return ep.MosaicResult(axes, artists, update=None)


def plot_yield_sweep(
    x,
    reports,
    *,
    channels=("y_issued", "y_wrong"),
    bound=None,
    interval="random",
    styles=None,
    zero_floor=None,
    log_x=False,
    log_y=False,
    x_label=None,
    axes=None,
    line_kw=None,
    interval_kw=None,
    ref_kw=None,
):
    """Draw yield channels and their stored intervals across a sweep.

    Args:
        x: Nonempty (S,) numeric coordinates, Profiles (using alpha), or
            strings (categorical positions with no connecting data line).
        reports: Sequence of S CertifiedYield reports, or a mapping of series
            names to such sequences. Bare arrays use a mapping of channel
            names to (S,) points or (S, 3) point/lower/upper triples; a mapping
            of series names to these column mappings is also accepted.
        channels: Yield channels in panel order, default issued and wrong.
        bound: ``profile`` uses each profile's issued_error_bound() times
            that series' issued yield. Alternatively, supply an (S,) bound
            shared by all series. Bounds appear only on the y_wrong panel.
        interval: ``random`` reads report.intervals, ``total`` reads
            report.total_intervals, and ``none`` omits intervals. Missing
            stored intervals raise; bare points require no intervals.
        styles: SourceStyles for series names; None builds one at call time.
            An unnamed sequence or channel mapping uses the name ``yield``.
            Series retain their palette colors, except wrong-yield channels
            use the alert role. Series markers distinguish those overlays.
        zero_floor: Positive display position for exact zero y_wrong values.
            Makes that panel logarithmic, with a zero tick and spine break.
            Interval endpoints on that panel are clipped to the floor;
            positive point values are unchanged. Zero points remain hollow.
        log_x: Use a logarithmic numeric x axis; invalid for categories.
        log_y: Use logarithmic y axes for all channels.
        x_label: Sweep label including its units. Profiles default to alpha
            (dimensionless); callers should label other physical coordinates.
        axes: Axes array of shape (C,), C = len(channels). None creates a row
            of panels. Supplied axes retain their layout and sharing.
        line_kw: Overrides passed to ax.plot for connecting data lines.
            Color and marker overrides also apply to the interval points.
        interval_kw: Overrides passed to interval_points for interval lines.
        ref_kw: Overrides passed to ax.plot for dashed neutral bound curves.
            Categorical bounds use unconnected underscore markers.

    Returns:
        MosaicResult with axes shaped (C,) and update=None. Artists contain
        ``line`` (connecting data lines), ``scatter`` (point collections),
        and ``lines`` (interval lines), flattened in channel then series
        order. Domain keys ``bound`` and ``zero_floor`` contain bound curves
        and spine-break strokes respectively. Categorical data have no
        connecting lines; the bound legend reads "stated bound".

    Raises:
        ValueError: If axes or data shapes disagree, intervals are unavailable,
            or coordinates, channels, bounds, or the zero floor are invalid.

    Example::

        from spaceodyssey.viz import plot_yield_sweep
        result = plot_yield_sweep(
            [0.01, 0.05], {"y_issued": [2, 3], "y_wrong": [0, 0.02]},
            zero_floor=0.001, x_label="alpha (dimensionless)",
        )
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D

    channels = tuple(channels)
    x, categories, profiles, series, bounds = _extract.yield_sweep_data(
        x, reports, channels, bound, interval
    )
    if log_x and (categories is not None or np.any(x <= 0)):
        raise ValueError("log_x requires positive numeric coordinates")
    if zero_floor is not None:
        _extract.zero_floor_values([], zero_floor)
    if axes is None:
        _, axes = plt.subplots(1, len(channels), squeeze=False, layout="constrained")
        axes = axes[0]
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != (len(channels),):
            raise ValueError(
                f"axes: expected shape {(len(channels),)}, received {axes.shape}"
            )
    if styles is None:
        styles = ep.SourceStyles(list(series))
    line_options = normalize_kwargs(line_kw, Line2D)
    ref_options = normalize_kwargs(ref_kw, Line2D)
    artists = {key: [] for key in ("line", "scatter", "lines", "bound", "zero_floor")}
    for ax, channel in zip(axes, channels, strict=True):
        floor = zero_floor if channel == "y_wrong" else None
        legend_handles, legend_labels = [], []
        for name, values in series.items():
            point, lo, hi = values[channel]
            zeros = point == 0
            if floor is not None:
                point, lo, hi, zeros = _extract.sweep_floor_data(point, lo, hi, floor)
            options = dict(styles[name])
            if channel in ("y_wrong", "y_neg_wrong"):
                options["color"] = ep.role(WRONG_YIELD_ROLE)
            options.update(line_options)
            if categories is None:
                artists["line"].extend(
                    ax.plot(x, point, **{"color": options["color"], **line_options})
                )
            result = ep.interval_points(
                x,
                point,
                lo,
                hi,
                ax=ax,
                color=options["color"],
                marker=options["marker"],
                label=name,
                interval_kw=interval_kw,
            )
            scatter = result.artists["scatter"]
            if floor is not None and zeros.any():
                faces = np.broadcast_to(scatter.get_facecolors(), (len(x), 4)).copy()
                edges = scatter.get_edgecolors().copy()
                faces[zeros, 3] = 0
                scatter.set_facecolors(faces)
                scatter.set_edgecolors(edges)
            artists["scatter"].append(scatter)
            artists["lines"].extend(result.artists["lines"])
            filled = np.flatnonzero(scatter.get_facecolors()[:, 3] > 0)
            legend_handles.append(scatter_key(scatter, filled[0] if len(filled) else 0))
            legend_labels.append(name)
        if channel == "y_wrong":
            for index, values in enumerate(bounds.values()):
                if floor is not None:
                    values, _ = _extract.zero_floor_values(values, floor)
                options = {
                    "color": ep.neutral(SCENERY_LEVEL),
                    "linestyle": "--",
                    "label": "stated bound" if index == 0 else "_nolegend_",
                    **ref_options,
                }
                if categories is not None:
                    options.update(linestyle="none", marker="_")
                artists["bound"].extend(ax.plot(x, values, **options))
                if index == 0:
                    legend_handles.append(artists["bound"][-1])
                    legend_labels.append(options["label"])
        if log_x:
            ax.set_xscale("log")
        if log_y or floor is not None:
            ax.set_yscale("log")
        if floor is not None:
            result = ep.log_zero_floor(ax, floor)
            artists["zero_floor"].extend(result.artists["lines"])
        decimal_log_ticks(ax.xaxis)
        decimal_log_ticks(ax.yaxis, zero_floor=floor)
        if categories is not None:
            ax.set_xticks(x, categories)
        ax.set_xlabel(
            x_label
            if x_label is not None
            else (r"$\alpha$ (dimensionless)" if profiles else "sweep parameter")
        )
        ax.set_ylabel(f"Expected planets\n({CHANNEL_LABELS[channel]})")
        if len(series) > 1 or (channel == "y_wrong" and bounds):
            ax.legend(legend_handles, legend_labels)
    return ep.MosaicResult(axes, artists, update=None)
