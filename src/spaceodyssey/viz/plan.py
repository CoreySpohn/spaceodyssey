"""Visit timelines and measured sky positions and spectra."""

import numpy as np

from spaceodyssey.viz import _extract
from spaceodyssey.viz._legend import legend_below, scatter_key
from spaceodyssey.viz._require import eyepiece

# Interim defaults: eyepiece owns the statistic fallback; rows use plan indices.
SNR_CMAP_ROLE = "statistic"
PLAN_ROW_LABEL = "planet {i}"
SCENERY_LEVEL = 0.6
DATA_ROLE = "data"
STAR_ROLE = "star"


def plot_plan_visits(
    plan,
    *,
    planets=None,
    labels=None,
    ax=None,
    snr_vlim=None,
    scatter_kw=None,
    cbar_kw=None,
):
    """Draw visits by planet, with SNR colors and square spectrum markers.

    Args:
        plan: PlanSummary or mapping with visit_epochs_d, visit_success, and
            visit_photometric_snr shaped (P, V), and spectrum_visit shaped (P,).
            Optional k_precision_ms (P,) gives an M annotation when finite.
            Nonfinite epochs are padding and omitted, including spectrum visits.
        planets: Integer indices shaped (M,), default all, in top-to-bottom order.
        labels: Row labels shaped (M,), default "planet {i}" using plan indices.
        ax: Single Axes (shape ()), or None to create one. Caller geometry,
            layout engine, and axis sharing are preserved.
        snr_vlim: Optional (minimum, maximum) SNR color limits. None lets
            eyepiece scale to the successful visits.
        scatter_kw: Overrides for ax.scatter on successful and failed visits.
            Spectrum markers remain squares; failed visits remain hollow scenery.
        cbar_kw: Overrides passed to eyepiece's inset colorbar.

    Returns:
        PlotResult with update=None and one colorbar inset. Artists contain
        ``scatter`` (successful then failed collections, with one marker path
        per visit), ``cbar``, and ``text`` (mass annotations in row order).
        Every finite selected epoch contributes exactly one marker.

    Raises:
        ValueError: If axes, labels, limits, or input shapes are invalid.

    Example::

        from spaceodyssey.viz import plot_plan_visits
        result = plot_plan_visits({
            "visit_epochs_d": [[0, 90]], "visit_success": [[True, False]],
            "visit_photometric_snr": [[5, 0]], "spectrum_visit": [0],
        })
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.markers import MarkerStyle

    data = _extract.plan_visit_data(plan, planets)
    rows = data["planets"]
    labels = (
        [PLAN_ROW_LABEL.format(i=i) for i in rows]
        if labels is None
        else _extract._array(labels, "labels", rows.shape, str)
    )
    limits = (None, None)
    if snr_vlim is not None:
        limits = _extract._array(snr_vlim, "snr_vlim", (2,))
        if not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("snr_vlim must be finite and increasing")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    elif np.shape(ax) != ():
        raise ValueError(f"ax: expected shape (), received {np.shape(ax)}")
    success = data["success"]
    result = ep.scatter_values(
        data["epoch_d"][success],
        data["row"][success],
        data["snr"][success],
        ax=ax,
        cmap=SNR_CMAP_ROLE,
        vmin=limits[0],
        vmax=limits[1],
        cbar_label="photometric SNR",
        scatter_kw=scatter_kw,
        cbar_kw=cbar_kw,
    )
    failed_kw = dict(scatter_kw or {})
    for key in ("c", "color", "cmap", "norm", "vmin", "vmax", "fc", "ec"):
        failed_kw.pop(key, None)
    failed_kw.update(facecolors="none", edgecolors=ep.neutral(SCENERY_LEVEL))
    failed = ax.scatter(data["epoch_d"][~success], data["row"][~success], **failed_kw)
    collections = [result.artists["scatter"], failed]
    ordinary = MarkerStyle((scatter_kw or {}).get("marker", "o"))
    square = MarkerStyle("s")
    paths = [m.get_path().transformed(m.get_transform()) for m in (ordinary, square)]
    for collection, mask in zip(collections, (success, ~success), strict=True):
        collection.set_paths([paths[int(flag)] for flag in data["spectrum"][mask]])
    texts = [
        ax.text(
            0.98,
            row,
            "M",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize="small",
        )
        for row in np.flatnonzero(data["mass_measured"])
    ]
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.set_ylim(max(len(rows) - 0.5, 0.5), -0.5)
    ax.set_xlabel("visit epoch (d)")
    handles, key_labels = [], []
    for collection, mask, status in zip(
        collections, (success, ~success), ("successful", "failed"), strict=True
    ):
        for spectrum, kind in ((False, "visit"), (True, "spectrum")):
            indices = np.flatnonzero(data["spectrum"][mask] == spectrum)
            if len(indices):
                handles.append(scatter_key(collection, indices[0]))
                key_labels.append(f"{status} {kind}")
    if texts:
        from matplotlib.lines import Line2D

        handles.append(
            Line2D([], [], marker="$M$", linestyle="none", color=texts[0].get_color())
        )
        key_labels.append("mass measured")
    if handles:
        legend_below(ax, handles, key_labels)
    return ep.PlotResult(
        ax,
        {"scatter": collections, "cbar": result.artists["cbar"], "text": texts},
        insets=result.insets,
    )


def plot_observed_data(
    data,
    plan,
    *,
    planet=0,
    expected=None,
    astrometric_corr=0.0,
    n_sigma=1,
    axes=None,
    point_kw=None,
    ellipse_kw=None,
    interval_kw=None,
    line_kw=None,
):
    """Draw one planet's usable astrometry and measured spectrum.

    Args:
        data: ObservedData or mapping with astrometry_mas (P, V, 2),
            visit_usable (P, V), and spectrum (P, B). Optional has_spectrum (P,)
            suppresses missing spectra; absent in mappings means all present.
        plan: PlanSummary or mapping with astrometric_precision_mas (P, V),
            noise_sigma (P, B), and bin_wavelength_nm (B,) or (P, B).
        planet: Integer planet index, default zero.
        expected: Optional noiseless spectrum shaped (B,), in the same flux
            reference as data. Drawn in scenery, even if no spectrum is present.
        astrometric_corr: Scalar correlation in [-1, 1]. Each ellipse uses
            precision**2 * [[1, rho], [rho, 1]].
        n_sigma: Positive ellipse radius in standard deviations. Spectral bars
            always use the plan's one-sigma noise_sigma, without recomputation.
        axes: Axes array shaped (2,): sky then spectrum. None creates panels.
            Caller layout and sharing are preserved; sky aspect is equal.
        point_kw: Scatter overrides shared by sky and spectral data points.
        ellipse_kw: Overrides passed to eyepiece.cov_ellipse for each ellipse.
        interval_kw: Overrides for spectral interval lines.
        line_kw: Line2D overrides for the expected spectrum.

    Returns:
        MosaicResult with update=None. Artists contain ``scatter`` (sky then
        spectrum), ``ellipse`` (one per usable visit in visit order), ``star``
        (origin collection), ``lines`` (spectral intervals), ``line`` (optional
        expected spectrum), and ``text`` (missing-spectrum annotation).
        Both data panels use the data role; the origin uses the star role.

    Raises:
        ValueError: If axes or input shapes, planet, correlation, or n_sigma
            are invalid, or usable measurements have invalid uncertainties.

    Example::

        from spaceodyssey.viz import plot_observed_data
        result = plot_observed_data(data, plan, planet=0)
        # Compose an orbit computed by the caller, e.g. with orbix, in mas:
        result.axes[0].plot(orbit_xy_mas[:, 0], orbit_xy_mas[:, 1])
    """
    ep = eyepiece()
    from matplotlib import pyplot as plt
    from matplotlib.cbook import normalize_kwargs
    from matplotlib.lines import Line2D

    arrays = _extract.observed_data_arrays(
        data, plan, planet, expected, astrometric_corr
    )
    n_sigma = _extract._array(n_sigma, "n_sigma", ())
    if not np.isfinite(n_sigma) or n_sigma <= 0:
        raise ValueError("n_sigma must be finite and positive")
    if axes is None:
        _, axes = plt.subplots(1, 2, layout="constrained")
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != (2,):
            raise ValueError(f"axes: expected shape (2,), received {axes.shape}")
    sky_ax, spectrum_ax = axes
    color = ep.role(DATA_ROLE)
    options = {"color": color, "zorder": 3, **(point_kw or {})}
    if "c" in options:
        options.pop("color", None)
    sky = sky_ax.scatter(*arrays["sky"].T, **options)
    ellipses = [
        ep.cov_ellipse(
            mean,
            covariance,
            ax=sky_ax,
            n_sigma=n_sigma,
            color=color,
            ellipse_kw=ellipse_kw,
        ).artists["ellipse"]
        for mean, covariance in zip(arrays["sky"], arrays["covariance"], strict=True)
    ]
    star = sky_ax.scatter([0], [0], marker="*", color=ep.role(STAR_ROLE), s=80)
    sky_ax.set_aspect("equal", adjustable="box")
    sky_ax.set_xlabel("sky x (mas)")
    sky_ax.set_ylabel("sky y (mas)")
    wave = arrays["wavelength_um"]
    spectrum = ep.interval_points(
        wave if arrays["has_spectrum"] else [],
        arrays["spectrum"],
        arrays["lo"],
        arrays["hi"],
        ax=spectrum_ax,
        color=color,
        hollow_zero=False,
        point_kw=point_kw,
        interval_kw=interval_kw,
    )
    lines, texts = [], []
    if arrays["expected"] is not None:
        lines = spectrum_ax.plot(
            wave,
            arrays["expected"],
            **{"color": ep.neutral(SCENERY_LEVEL), **normalize_kwargs(line_kw, Line2D)},
        )
    if not arrays["has_spectrum"]:
        texts.append(
            spectrum_ax.text(
                0.5, 0.5, "no spectrum", transform=spectrum_ax.transAxes, ha="center"
            )
        )
    spectrum_ax.set_xlabel(r"wavelength ($\mu$m)")
    spectrum_ax.set_ylabel("flux / reference flux")
    return ep.MosaicResult(
        axes,
        {
            "scatter": [sky, spectrum.artists["scatter"]],
            "ellipse": ellipses,
            "star": star,
            "lines": spectrum.artists["lines"],
            "line": lines,
            "text": texts,
        },
    )
