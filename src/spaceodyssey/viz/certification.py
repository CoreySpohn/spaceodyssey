"""Certification probabilities across physical sweeps and fidelity tiers."""

import numpy as np

from spaceodyssey.viz import _extract
from spaceodyssey.viz._require import eyepiece

VERDICT_PALETTE_NAMES = ("positive", "negative")
WRONG_VERDICT_ROLE = "alert"
SCENERY_LEVEL = 0.6
OUT_OF_RANGE_LEVEL = 0.12
VERDICT_LABELS = {"pos": "positive", "neg": "negative", "wrong": "wrong"}
TIER_DATA_ROLE = "data"
# U1 ships in eyepiece 0.9.0; keep the upstream capability choice explicit.
TIER_AGREEMENT_PRIMITIVE = "calibration"


def _verdict_styles(ep):
    """Resolve the interim verdict palette and wrong role at call time."""
    styles = ep.SourceStyles(list(VERDICT_PALETTE_NAMES))
    return {
        "pos": dict(styles["positive"]),
        "neg": dict(styles["negative"]),
        "wrong": {"color": ep.role(WRONG_VERDICT_ROLE), "marker": "o"},
    }


def plot_certification_curve(
    x,
    outcome,
    *,
    reference=None,
    x_reference=None,
    reference_intervals=None,
    indicator=None,
    channels=("pos", "neg"),
    colors=None,
    label=None,
    confidence=0.95,
    log_x=False,
    x_label=None,
    threshold=None,
    ax=None,
    line_kw=None,
    point_kw=None,
    interval_kw=None,
    span_kw=None,
    ref_kw=None,
):
    """Plot certification probabilities and reference frequencies across a sweep.

    Args:
        x: Nonempty finite (S,) sweep coordinates, strictly increasing. Callers
            should supply x_label with the physical quantity and its units.
        outcome: CertificationOutcome with P == S, or a mapping of pos, neg,
            and wrong to (S,) probability arrays. Only requested channels are
            required. Optional in_range defaults to all true in a mapping.
            Replicate arrays are ignored; out-of-range points remain plotted.
        reference: Optional outcome or bare channel mapping, drawn as points.
            Zero probabilities are hollow. Bare mappings may include pos_se,
            neg_se, and wrong_se; missing errors give points without intervals.
        x_reference: Reference coordinates shaped (S_ref,), default x.
        reference_intervals: Mapping of channels to (lo, hi) endpoint arrays,
            each shaped (S_ref,). Supplied endpoints are drawn unchanged.
            Other channels use reference probabilities +/- z * standard error,
            clipped to [0, 1], where z is the central normal confidence quantile.
        indicator: Optional (S,) probability, dashed in the positive color.
        channels: Nonempty subset of pos, neg, and wrong, in drawing order.
        colors: Channel-to-color overrides. Positive and negative default to
            palette slots 0 and 1; wrong uses the alert role, resolved at call time.
        label: Optional prefix for series labels, useful for repeated overlays.
        confidence: Central normal interval confidence, strictly between 0 and 1.
        log_x: Use a logarithmic x axis and geometric midpoint span edges.
        x_label: Physical sweep quantity and units; None retains the axis label.
        threshold: Optional scalar x reference, drawn as a neutral dashed line.
        ax: Single Axes, shape (), or None to create one. Caller layout and
            sharing are preserved. Probability limits are fixed at [0, 1].
        line_kw: Line2D overrides for approximate curves and the indicator.
            The indicator remains dashed and shares the positive curve color.
        point_kw: Scatter overrides for reference points.
        interval_kw: Line2D overrides for reference interval bars.
        span_kw: Patch overrides for out-of-range shading.
        ref_kw: Line2D overrides for the scalar threshold line.

    Returns:
        PlotResult with update=None and no insets. Artists contain ``line``
        (approximate curves), ``scatter`` (reference collections), ``lines``
        (reference interval bars), and ``fill`` (out-of-range span patches).
        Curves and collections follow channels order; bars follow channel then
        reference-point order. Domain keys ``indicator`` and ``threshold`` are
        lists of optional lines; ``text`` holds the threshold label.

    Raises:
        ValueError: If axes or array shapes disagree, or probabilities,
            coordinates, confidence, errors, or intervals are invalid.

    Example::

        from spaceodyssey.viz import plot_certification_curve
        result = plot_certification_curve(
            [1, 2, 3], {"pos": [0.1, 0.5, 0.9]}, channels=("pos",),
            x_label="signal-to-noise ratio (dimensionless)",
        )
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D

    channels = tuple(channels)
    x, series, in_range, xr, points, indicator, threshold = (
        _extract.certification_curve_data(
            x,
            outcome,
            channels=channels,
            reference=reference,
            x_reference=x_reference,
            reference_intervals=reference_intervals,
            indicator=indicator,
            confidence=confidence,
            log_x=log_x,
            threshold=threshold,
        )
    )
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    elif np.shape(ax) != ():
        raise ValueError(f"ax: expected shape (), received {np.shape(ax)}")
    styles = _verdict_styles(ep)
    for channel, color in (colors or {}).items():
        styles[channel]["color"] = color
    line_options = normalize_kwargs(line_kw, Line2D)
    artists = {
        key: []
        for key in (
            "line",
            "scatter",
            "lines",
            "fill",
            "indicator",
            "threshold",
            "text",
        )
    }
    prefix = f"{label}: " if label is not None else ""
    for channel, values in series.items():
        color = line_options.get("color", styles[channel]["color"])
        artists["line"].extend(
            ax.plot(
                x,
                values,
                **{
                    "color": color,
                    "label": f"{prefix}{VERDICT_LABELS[channel]} (approximate)",
                    **line_options,
                },
            )
        )
        if channel in points:
            point, lo, hi = points[channel]
            result = ep.interval_points(
                xr,
                point,
                lo,
                hi,
                ax=ax,
                color=color,
                marker=styles[channel]["marker"],
                label=f"{prefix}{VERDICT_LABELS[channel]} (reference)",
                point_kw=point_kw,
                interval_kw=interval_kw,
            )
            artists["scatter"].append(result.artists["scatter"])
            artists["lines"].extend(result.artists["lines"])
    if indicator is not None:
        artists["indicator"].extend(
            ax.plot(
                x,
                indicator,
                **{
                    "color": styles["pos"]["color"],
                    "label": f"{prefix}indicator",
                    **line_options,
                    "linestyle": "--",
                },
            )
        )
    artists["fill"] = ep.spans(
        x,
        ~in_range,
        ax=ax,
        edges="midpoint",
        log=log_x,
        color=ep.neutral(OUT_OF_RANGE_LEVEL),
        fill_kw=span_kw,
    ).artists["fill"]
    if threshold is not None:
        result = ep.ref_line(
            ax,
            threshold,
            "threshold",
            orient="v",
            color=ep.neutral(SCENERY_LEVEL),
            line_kw=ref_kw,
        )
        artists["threshold"].append(result.artists["line"])
        artists["text"].append(result.artists["text"])
    if log_x:
        ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Certification probability")
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.legend()
    return ep.PlotResult(ax, artists, update=None, insets=())


def plot_tier_agreement(
    outcome,
    reference,
    *,
    claims=None,
    verdicts=("pos", "neg"),
    tolerance=None,
    axes=None,
    point_kw=None,
    ref_kw=None,
):
    """Compare approximate probabilities with reference frequencies per claim.

    Args:
        outcome: Approximate CertificationOutcome with claim_names, claim_p_pos,
            claim_p_neg, and claim_in_range shaped (P, K). Alternatively, a
            mapping with claim_names and pos, neg, in_range arrays shaped (K, P).
            Only requested verdict probability fields are required.
        reference: Reference outcome or mapping in the same form. Claim names
            and their order must agree; planet rows must correspond. Its range
            mask is required but does not control marker fill.
        claims: Nonempty unique subset of claim names in panel order; None
            selects all claims. Titles replace underscores with spaces.
        verdicts: Nonempty unique subset of pos and neg, pooled in each panel.
        tolerance: Optional finite nonnegative absolute half-width about the
            identity line. The band is clipped to [0, 1].
        axes: Axes array shaped (K_selected,), or None to create a row. Caller
            layout, slot geometry, and sharing are preserved. Both probability
            axes are linear with limits [0, 1].
        point_kw: Scatter overrides for pooled points, in verdict then planet
            order per panel. Filled points default to the data role. Points
            outside the approximate tier's range remain hollow with neutral
            edges, including when alpha or color is overridden.
        ref_kw: Line2D overrides for the identity line.

    Returns:
        MosaicResult with update=None. Artists contain lists in panel order:
        ``scatter`` (one pooled collection per claim), ``line`` (identity),
        ``fill`` (tolerance bands, empty when omitted), and ``title``. The
        ``lines`` list is empty because no intervals are estimated or drawn.

    Raises:
        ValueError: If claim fields are absent, claim names or shapes disagree,
            selections are invalid, or probabilities or tolerance are invalid.

    Example::

        from spaceodyssey.viz import plot_tier_agreement
        approximate = {
            "claim_names": ("water",), "pos": [[0.2, 0.8]],
            "in_range": [[True, False]],
        }
        reference = {**approximate, "pos": [[0.25, 0.75]]}
        result = plot_tier_agreement(approximate, reference, verdicts=("pos",))
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba

    panels = _extract.tier_agreement_data(
        outcome, reference, claims=claims, verdicts=verdicts, tolerance=tolerance
    )
    shape = (len(panels),)
    if axes is None:
        _, axes = plt.subplots(1, len(panels), squeeze=False, layout="constrained")
        axes = axes[0]
    elif np.shape(axes) != shape:
        raise ValueError(f"axes: expected shape {shape}, received {np.shape(axes)}")
    axes = np.asarray(axes, dtype=object)
    artists = {key: [] for key in ("scatter", "line", "lines", "fill", "title")}
    calibration = getattr(ep, TIER_AGREEMENT_PRIMITIVE)
    for ax, (name, x, y, hollow) in zip(axes, panels, strict=True):
        options = {"color": ep.role(TIER_DATA_ROLE), **(point_kw or {})}
        result = calibration(
            x,
            y,
            ax=ax,
            hollow=hollow,
            scale="linear",
            lim=(0, 1),
            tolerance=tolerance,
            point_kw=options,
            ref_kw=ref_kw,
        )
        scatter = result.artists["scatter"]
        if hollow.any():
            colors = scatter.get_edgecolors()
            edges = (
                np.broadcast_to(colors, (len(x), 4)).copy()
                if len(colors)
                else np.zeros((len(x), 4))
            )
            edges[hollow] = to_rgba(
                ep.neutral(SCENERY_LEVEL), alpha=options.get("alpha")
            )
            scatter.set_edgecolors(edges)
        for key, artist in result.artists.items():
            if key == "lines":
                artists[key].extend(artist)
            else:
                artists[key].append(artist)
        artists["title"].append(ax.set_title(name.replace("_", " ")))
        ax.set_xlabel("Approximate-tier probability")
        ax.set_ylabel("Reference-tier frequency")
    return ep.MosaicResult(axes, artists, update=None)
