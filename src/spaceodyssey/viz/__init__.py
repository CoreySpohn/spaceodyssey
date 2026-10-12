"""Optional plotting for spaceodyssey types, built on eyepiece primitives.

Plot names resolve lazily and require ``pip install 'spaceodyssey[viz]'``.
Labels and numerical extraction work without the plotting stack.
"""

import importlib

_LAZY = {
    "plot_claim_posterior": "spaceodyssey.viz.claims",
    "plot_planet_plane": "spaceodyssey.viz.claims",
    "plot_certification_curve": "spaceodyssey.viz.certification",
    "plot_tier_agreement": "spaceodyssey.viz.certification",
    "plot_yield_comparison": "spaceodyssey.viz.yields",
    "plot_yield_sweep": "spaceodyssey.viz.yields",
    "plot_plan_visits": "spaceodyssey.viz.plan",
    "plot_observed_data": "spaceodyssey.viz.plan",
}

_DATA = {
    "QuantityDisplay": "spaceodyssey.viz.labels",
    "QUANTITY_DISPLAY": "spaceodyssey.viz.labels",
    "CHANNEL_LABELS": "spaceodyssey.viz.labels",
}

__all__ = sorted([*_LAZY, *_DATA])


def __getattr__(name):
    """Resolve a lazy export, checking the plot dependency before dispatch.

    Args:
        name: Attribute requested from this package.

    Returns:
        The requested function or label data.

    Raises:
        AttributeError: If the name is not a declared export.
        ImportError: If a plot is requested without the viz extra.
    """
    if name in _DATA:
        return getattr(importlib.import_module(_DATA[name]), name)
    if name in _LAZY:
        from spaceodyssey.viz import _require

        _require.eyepiece()
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'spaceodyssey.viz' has no attribute {name!r}")


def __dir__():
    """List real attributes and lazy exports.

    Returns:
        Sorted attribute names advertised by this package.
    """
    return sorted(set(globals()) | set(__all__))
