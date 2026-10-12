"""Expected planet yields against statistical or physical sweep coordinates."""

import numpy as np

from spaceodyssey.viz import _extract
from spaceodyssey.viz._require import eyepiece
from spaceodyssey.viz.labels import CHANNEL_LABELS

SCENERY_LEVEL = 0.6
WRONG_YIELD_ROLE = "alert"


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
        if log_x:
            ax.set_xscale("log")
        if log_y or floor is not None:
            ax.set_yscale("log")
        if floor is not None:
            result = ep.log_zero_floor(ax, floor)
            artists["zero_floor"].extend(result.artists["lines"])
        if categories is not None:
            ax.set_xticks(x, categories)
        ax.set_xlabel(
            x_label
            if x_label is not None
            else (r"$\alpha$ (dimensionless)" if profiles else "sweep parameter")
        )
        ax.set_ylabel(f"Expected planets ({CHANNEL_LABELS[channel]})")
        if len(series) > 1 or (channel == "y_wrong" and bounds):
            ax.legend()
    return ep.MosaicResult(axes, artists, update=None)
