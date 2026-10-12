"""Claim regions and planetary populations in physical coordinates."""

import numpy as np

from spaceodyssey.viz import _extract
from spaceodyssey.viz._axes import decimal_log_ticks
from spaceodyssey.viz._legend import legend_below, scatter_key
from spaceodyssey.viz._require import eyepiece
from spaceodyssey.viz.labels import CLAIM_LABELS

# Resolve style roles at call time; eyepiece owns their fallback policy.
PLANE_CMAP_ROLE = "probability"
POSTERIOR_ROLE = "model"
SCENERY_LEVEL = 0.6
INACTIVE_LEVEL = 0.3
CLOUD_ALPHA = 0.2
OUTCOME_CBAR_LABEL = "P(certified positive)"
PLANE_QUANTITIES = ("ln_a_eeid", "ln_rp_re")
# Interim panel policy: retain every profile quantity, including coupled radius.
POSTERIOR_OMITTED_QUANTITIES = ()
COUNTED_REGION_LEVEL = 0.15


def plot_claim_posterior(
    samples,
    profile,
    *,
    weights=None,
    truth=None,
    quantities=None,
    display=None,
    bins=40,
    risk=None,
    verdict=None,
    show_mass=True,
    axes=None,
    hist_kw=None,
    edge_kw=None,
    truth_kw=None,
    region_kw=None,
    text_kw=None,
):
    """Draw posterior marginals against the exact edges of a claim profile.

    Args:
        samples: Mapping from profile quantity names to bare NumPy, device, or
            array-like samples shaped (N,). All profile quantities are required.
        profile: Profile supplying claim edges and names.
        weights: Optional nonnegative finite importance weights shaped (N,).
            Histograms sum these weights; claim masses use normalized weights.
        truth: Optional mapping of quantity names to scalar stored truth values.
        quantities: Nonempty unique selection of profile quantities in panel
            order. Defaults to all quantities in first edge appearance order.
        display: Optional QuantityDisplay overrides. Unknown names use their raw
            name on a linear identity axis.
        bins: Positive count, or explicit increasing edges in displayed units.
            Automatic bins are geometric for exponential transforms or log axes.
        risk: Scalar or (K,) certifier risk in profile claim order. Printed using
            str without rounding, recalculation, or comparison with claim mass.
        verdict: Optional string printed verbatim beside the supplied risk.
        show_mass: Print each claim's joint posterior mass, to two decimals, on
            its first displayed quantity. A multi-quantity claim is evaluated
            jointly, even though each panel shades only its marginal edges.
        axes: Array of Axes shaped (Q,), one per selected quantity, or None to
            create a row. Caller layout, geometry, sharing, and limits survive.
        hist_kw: Bar styling overrides routed to eyepiece.hist_fill bar_kw.
        edge_kw: Line overrides routed to eyepiece.ref_line for each threshold.
        truth_kw: Line overrides for truth references, solid by default.
        region_kw: Fill overrides routed to eyepiece.spans, one intersection
            per claim on the displayed quantity.
        text_kw: Text overrides for titles and the risk/verdict block.

    Returns:
        MosaicResult with update=None and artists ``hist`` (BarContainers),
        ``fill`` (lists of region patches), ``lines`` (lists of edge lines),
        ``truth`` (line or None), and ``title`` (Text), each in panel order.
        ``text`` is a list containing the optional risk/verdict block, located
        at (1.04, 0.5) in the last panel's axes coordinates.

    Raises:
        ValueError: If samples, weights, bins, truth, risk, or axes shapes are
            invalid. Missing sample errors name the quantity and received keys.

    Example::

        from spaceodyssey.profiles import WATER_ONLY
        from spaceodyssey.viz import plot_claim_posterior
        result = plot_claim_posterior(
            {"log10_h2o_vmr": [-5, -3, -2]}, WATER_ONLY,
            risk=0.02, verdict="certified",
        )
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    if quantities is None:
        quantities = tuple(
            name
            for name in _extract.edges_by_quantity(profile)
            if name not in POSTERIOR_OMITTED_QUANTITIES
        )
    panels, masses = _extract.claim_posterior_data(
        samples,
        profile,
        weights=weights,
        truth=truth,
        quantities=quantities,
        display=display,
        bins=bins,
    )
    if risk is not None and np.shape(risk) not in ((), (len(profile.claims),)):
        raise ValueError(
            f"risk: expected shape () or {(len(profile.claims),)}, "
            f"received {np.shape(risk)}"
        )
    shape = (len(panels),)
    if axes is None:
        _, axes = plt.subplots(
            1,
            len(panels),
            squeeze=False,
            layout="constrained",
            figsize=(3.5 * len(panels), 3),
        )
        axes = axes[0]
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != shape:
            raise ValueError(f"axes: expected shape {shape}, received {axes.shape}")
    titles = {name: [] for name in panels}
    if show_mass:
        for claim, mass in zip(profile.claims, masses, strict=True):
            first = next(
                (
                    name
                    for name in panels
                    if any(e.quantity == name for e in claim.edges)
                ),
                None,
            )
            if first is not None:
                label = CLAIM_LABELS.get(claim.name, claim.name.replace("_", " "))
                titles[first].append(f"P({label}) = {mass:.2f}")
    artists = {key: [] for key in ("hist", "fill", "lines", "truth", "title", "text")}
    text_options = normalize_kwargs(text_kw, Text)
    for ax, (name, panel) in zip(axes, panels.items(), strict=True):
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        auto_x, auto_y = ax.get_autoscalex_on(), ax.get_autoscaley_on()
        ax.set_xscale(panel["display"].scale)
        result = ep.hist_fill(
            panel["values"],
            panel["bins"],
            ax=ax,
            color=ep.role(POSTERIOR_ROLE),
            bar_kw=hist_kw,
        )
        bars = result.artists["hist"]
        # hist_fill owns bar styling; importance weights supply the heights.
        for bar, count in zip(bars, panel["counts"], strict=True):
            bar.set_height(count)
        bars.datavalues = panel["counts"]
        ax.set_ylim(0, 1.08 * max(panel["counts"]) or 1)
        if not auto_x:
            ax.set_xlim(xlim)
        if not auto_y:
            ax.set_ylim(ylim)
        artists["hist"].append(bars)
        artists["fill"].append(
            ep.spans(
                intervals=panel["intervals"],
                ax=ax,
                color=ep.neutral(COUNTED_REGION_LEVEL),
                fill_kw=region_kw,
            ).artists["fill"]
        )
        artists["lines"].append(
            [
                ep.ref_line(
                    ax,
                    value,
                    orient="v",
                    color=ep.neutral(SCENERY_LEVEL),
                    line_kw=edge_kw,
                ).artists["line"]
                for value in panel["thresholds"]
            ]
        )
        actual = None
        if panel["truth"] is not None:
            actual = ep.ref_line(
                ax,
                panel["truth"],
                orient="v",
                color=ep.neutral(SCENERY_LEVEL),
                line_kw={"linestyle": "-", **normalize_kwargs(truth_kw, Line2D)},
            ).artists["line"]
        artists["truth"].append(actual)
        artists["title"].append(ax.set_title("\n".join(titles[name]), **text_options))
        ax.set_xlabel(panel["display"].label)
        ax.set_ylabel("weight" if weights is not None else "samples")
        decimal_log_ticks(ax.xaxis)
    block = []
    if risk is not None:
        block.append(f"risk = {risk}")
    if verdict is not None:
        block.append(str(verdict))
    if block:
        artists["text"].append(
            axes[-1].text(
                1.04,
                0.5,
                "\n".join(block),
                **{
                    "transform": axes[-1].transAxes,
                    "ha": "left",
                    "va": "center",
                    **text_options,
                },
            )
        )
    return ep.MosaicResult(axes, artists)


def plot_planet_plane(
    a_eeid,
    rp_re,
    *,
    values=None,
    active=None,
    in_range=None,
    marked=None,
    profile=None,
    truth=None,
    display=None,
    ax=None,
    vlim=(0.0, 1.0),
    cbar_label=None,
    scatter_kw=None,
    region_kw=None,
    cbar_kw=None,
):
    """Draw planets and a profile's counted region in the distance-radius plane.

    Args:
        a_eeid: Finite positive distance ratios shaped (P,).
        rp_re: Finite positive radii in Earth radii shaped (P,).
        values: CertificationOutcome (reads only p_pos and in_range), or bare
            (P,) probabilities in [0, 1]. None draws a faint model-colored cloud.
            Replicate arrays never contribute points.
        active: Boolean (P,) mask, default all true. Inactive planets are small
            neutral markers drawn first, without probability colors.
        in_range: Boolean (P,) mask, overriding the outcome's mask when given.
            Defaults to all true for arrays. Every false entry is a triangle,
            including inactive planets; no planet is dropped.
        marked: Boolean (P,) mask, default all false. Adds text-colored rings.
        profile: Optional Profile. Contours the intersection of plane-expressible
            claims at 0.5 on a 400 by 400 logarithmic physical grid spanning the
            current limits. Claims needing other quantities are skipped whole.
        truth: Optional finite positive (distance ratio, Earth radius) pair,
            drawn as a neutral star.
        display: Optional QuantityDisplay mapping for ln_a_eeid and ln_rp_re.
            Transforms apply to the natural logs of the physical inputs; the
            defaults give physical units on logarithmic axes.
        ax: Single Axes, shape (), or None to create one. Caller geometry,
            layout engine, sharing, and explicit limits are preserved.
        vlim: Finite increasing (minimum, maximum) probability color limits.
        cbar_label: Colorbar label; defaults to "P(certified positive)" for an
            outcome. Bare arrays have no default label.
        scatter_kw: ax.scatter overrides for active points. Array-valued options
            must match the active subset. Out-of-range markers remain triangles.
        region_kw: ax.contour overrides, with the contour level fixed at 0.5.
        cbar_kw: Overrides passed to eyepiece's inset colorbar.

    Returns:
        PlotResult with update=None. Artists contain ``scatter`` (inactive then
        active collections, one path per planet), ``collection`` (region contour
        or None), ``cbar`` (colorbar or None), ``region_skipped`` (claim names),
        ``marked`` (ring collection or None), and ``truth`` (star or None).
        Rings and truth are separate from the P planet markers. Value mode has
        one colorbar inset; cloud mode has none. A uniform region has no contour.

    Raises:
        ValueError: If axes, input shapes, coordinates, probabilities, or color
            limits are invalid.

    Example::

        from spaceodyssey.profiles import EXOEARTH_CANDIDATE
        from spaceodyssey.viz import plot_planet_plane
        result = plot_planet_plane(
            [0.5, 1, 2], [0.7, 1, 1.5], values=[0.1, 0.9, 0.2],
            profile=EXOEARTH_CANDIDATE,
        )
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.markers import MarkerStyle

    data = _extract.planet_plane_data(
        a_eeid, rp_re, values=values, active=active, in_range=in_range, marked=marked
    )
    limits = _extract._array(vlim, "vlim", (2,))
    if not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("vlim must be finite and increasing")
    coordinates = [
        _extract.transform_quantity(np.log(data[key]), quantity, display)
        for key, quantity in zip(("a_eeid", "rp_re"), PLANE_QUANTITIES, strict=True)
    ]
    truth_xy = None
    if truth is not None:
        truth = _extract._array(truth, "truth", (2,))
        truth_data = _extract.planet_plane_data(truth[:1], truth[1:])
        truth_xy = [
            _extract.transform_quantity(np.log(truth_data[key]), quantity, display)
            for key, quantity in zip(("a_eeid", "rp_re"), PLANE_QUANTITIES, strict=True)
        ]
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    elif np.shape(ax) != ():
        raise ValueError(f"ax: expected shape (), received {np.shape(ax)}")
    for axis, quantity in zip((ax.xaxis, ax.yaxis), PLANE_QUANTITIES, strict=True):
        spec = _extract.quantity_display(quantity, display)
        getattr(ax, f"set_{axis.axis_name}scale")(spec.scale)
        axis.set_label_text(spec.label)

    x, y = coordinates
    active = data["active"]
    inactive = ax.scatter(
        x[~active], y[~active], color=ep.neutral(INACTIVE_LEVEL), s=12, zorder=1
    )
    artists = {
        "scatter": [inactive],
        "collection": None,
        "cbar": None,
        "region_skipped": (),
        "marked": None,
        "truth": None,
    }
    options = {"s": 24, "zorder": 2, **(scatter_kw or {})}
    insets = ()
    if data["values"] is None:
        if "color" not in options and "c" not in options:
            options["color"] = ep.role(POSTERIOR_ROLE)
        options.setdefault("alpha", CLOUD_ALPHA)
        points = ax.scatter(x[active], y[active], **options)
    else:
        if cbar_label is None and hasattr(values, "p_pos"):
            cbar_label = OUTCOME_CBAR_LABEL
        result = ep.scatter_values(
            x[active],
            y[active],
            data["values"][active],
            ax=ax,
            cmap=PLANE_CMAP_ROLE,
            vmin=limits[0],
            vmax=limits[1],
            cbar_label=cbar_label,
            scatter_kw=options,
            cbar_kw=cbar_kw,
        )
        points = result.artists["scatter"]
        artists["cbar"] = result.artists["cbar"]
        insets = result.insets
    artists["scatter"].append(points)
    triangle = MarkerStyle("^")
    triangle_path = triangle.get_path().transformed(triangle.get_transform())
    for collection, mask in zip(artists["scatter"], (~active, active), strict=True):
        ordinary_path = collection.get_paths()[0]
        collection.set_paths(
            [
                ordinary_path if flag else triangle_path
                for flag in data["in_range"][mask]
            ]
        )
    if data["marked"].any():
        marked = data["marked"]
        artists["marked"] = ax.scatter(
            x[marked],
            y[marked],
            s=90,
            marker="o",
            facecolors="none",
            edgecolors=ep.neutral(1.0),
            zorder=3,
        )
    if truth_xy is not None:
        artists["truth"] = ax.scatter(
            *truth_xy, marker="*", s=90, color=ep.neutral(SCENERY_LEVEL), zorder=4
        )
    if profile is not None:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        gx, gy, mask, skipped = _extract.plane_region_grid(profile, xlim, ylim, display)
        artists["region_skipped"] = skipped
        if mask.any() and not mask.all():
            artists["collection"] = ax.contour(
                gx,
                gy,
                mask.astype(float),
                **{
                    "colors": [ep.neutral(SCENERY_LEVEL)],
                    "zorder": 0,
                    **(region_kw or {}),
                    "levels": [0.5],
                },
            )
            ax.set_xlim(xlim, auto=ax.get_autoscalex_on())
            ax.set_ylim(ylim, auto=ax.get_autoscaley_on())
    handles, labels = [], []
    # Prefer active examples so the range key uses the probability/cloud style.
    for flag, label in ((True, "in range"), (False, "out of range")):
        for collection, mask in ((points, active), (inactive, ~active)):
            indices = np.flatnonzero(data["in_range"][mask] == flag)
            if len(indices):
                handles.append(scatter_key(collection, indices[0]))
                labels.append(label)
                break
    if (~active).any():
        handles.append(scatter_key(inactive))
        labels.append("inactive")
    for key, label in (("marked", "marked (ring)"), ("truth", "truth")):
        if artists[key] is not None:
            handles.append(scatter_key(artists[key]))
            labels.append(label)
    if artists["collection"] is not None:
        from matplotlib.lines import Line2D

        contour = artists["collection"]
        width = contour.get_linewidths()[0]
        offset, dashes = contour.get_linestyles()[0]
        # Collection dash lengths are already scaled; Line2D scales on input.
        if plt.rcParams["lines.scale_dashes"] and width and dashes is not None:
            offset, dashes = offset / width, np.asarray(dashes) / width
        handles.append(
            Line2D(
                [],
                [],
                color=contour.get_edgecolors()[0],
                linewidth=width,
                linestyle=(offset, dashes),
            )
        )
        labels.append("claim region")
    if handles:
        legend_below(ax, handles, labels, ncols=3)
    decimal_log_ticks(ax.xaxis)
    decimal_log_ticks(ax.yaxis)
    return ep.PlotResult(ax, artists, insets=insets)
