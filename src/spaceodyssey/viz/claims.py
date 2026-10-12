"""Claim regions and planetary populations in physical coordinates."""

import numpy as np

from spaceodyssey.viz import _extract
from spaceodyssey.viz._require import eyepiece

# Resolve style roles at call time; eyepiece owns their fallback policy.
PLANE_CMAP_ROLE = "probability"
POSTERIOR_ROLE = "model"
SCENERY_LEVEL = 0.6
INACTIVE_LEVEL = 0.3
CLOUD_ALPHA = 0.2
OUTCOME_CBAR_LABEL = "P(certified positive)"
PLANE_QUANTITIES = ("ln_a_eeid", "ln_rp_re")


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
    return ep.PlotResult(ax, artists, insets=insets)
