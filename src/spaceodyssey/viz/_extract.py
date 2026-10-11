"""Numerical inputs for visualization, using NumPy without a renderer.

Device arrays are converted to host arrays here. Replicate arrays are never
treated as planets, and stored report intervals are never estimated again.
"""

from collections.abc import Mapping

import numpy as np

from spaceodyssey.viz.labels import QUANTITY_DISPLAY, QuantityDisplay


def _field(source, name, default=None):
    """Read an attribute or a field in a bare-array mapping."""
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _array(value, name, shape=None, dtype=float):
    """Convert a required array and validate its shape."""
    if value is None:
        raise ValueError(f"missing {name}")
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, received {array.shape}")
    return array


def edges_by_quantity(profile):
    """Group edges in order of first appearance in the profile.

    Args:
        profile: Profile whose claims and edges define panel order.

    Returns:
        Ordered dict of quantity names to tuples of original Edge objects.
        Repeated quantities share a panel, retaining every claim's edges.
    """
    grouped = {}
    for claim in profile.claims:
        for edge in claim.edges:
            grouped.setdefault(edge.quantity, []).append(edge)
    return {name: tuple(edges) for name, edges in grouped.items()}


def _claim_truth(claim, quantities):
    """Evaluate a claim's inclusive edge conjunction using NumPy."""
    truth = np.asarray(True)
    for edge in claim.edges:
        value = np.asarray(quantities[edge.quantity])
        holds = (
            value >= edge.threshold if edge.side == "above" else value <= edge.threshold
        )
        truth = truth & holds
    return truth


def plane_region_mask(profile, a_eeid, rp_re):
    """Evaluate claims expressible in the planet plane, skipping other claims.

    Args:
        profile: Profile defining the counted region.
        a_eeid: Positive distance ratios, broadcastable with rp_re. For a grid,
            pass meshgrid arrays or arrays shaped (1, A) and (R, 1).
        rp_re: Positive radii in Earth radii, broadcastable with a_eeid.

    Returns:
        Tuple of the boolean region mask and skipped claim names in claim order.
        The region ANDs all plane-expressible claims, including either-counted
        claims. If none are expressible, the mask is true everywhere.

    Raises:
        ValueError: If the plane coordinates are nonfinite or nonpositive.
    """
    a, rp = np.broadcast_arrays(
        np.asarray(a_eeid, dtype=float), np.asarray(rp_re, dtype=float)
    )
    if not (np.isfinite(a).all() and np.isfinite(rp).all()) or np.any(
        (a <= 0) | (rp <= 0)
    ):
        raise ValueError("plane coordinates must be finite and positive")
    ln_a, ln_rp = np.log(a), np.log(rp)
    quantities = {
        "ln_a_eeid": ln_a,
        "ln_rp_re": ln_rp,
        "ln_rp_sqrt_a_eeid": ln_rp + ln_a / 2,
    }
    mask = np.ones(a.shape, dtype=bool)
    skipped = []
    for claim in profile.claims:
        if any(edge.quantity not in quantities for edge in claim.edges):
            skipped.append(claim.name)
        else:
            mask &= _claim_truth(claim, quantities)
    return mask, tuple(skipped)


def claim_mass(samples, profile, weights=None):
    """Average inclusive claim truth over posterior draws.

    Args:
        samples: Mapping of profile quantities to arrays shaped (N,).
        profile: Profile defining claims in the returned order.
        weights: Nonnegative finite importance weights shaped (N,), or None.

    Returns:
        Array shaped (K,) of posterior masses in profile.claim_names order.

    Raises:
        ValueError: If keys, shapes, or weights are invalid, or no draws exist.
    """
    quantities = {}
    shape = None
    for name in edges_by_quantity(profile):
        if name not in samples:
            raise ValueError(
                f"missing quantity {name!r}; received keys {tuple(samples)}"
            )
        value = _array(samples[name], name, shape, None)
        if value.ndim != 1 or value.size == 0:
            raise ValueError(
                f"{name}: expected nonempty shape (N,), received {value.shape}"
            )
        shape = value.shape
        quantities[name] = value
    truth = np.stack(
        [_claim_truth(claim, quantities) for claim in profile.claims], axis=-1
    )
    if weights is None:
        return truth.mean(axis=0)
    w = _array(weights, "weights", shape)
    if not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and have positive sum")
    return np.average(truth, weights=w, axis=0)


def quantity_display(quantity, display=None):
    """Resolve an override or fall back to a raw linear identity axis.

    Args:
        quantity: Stored quantity name.
        display: Optional mapping overriding individual QuantityDisplay entries.

    Returns:
        QuantityDisplay for the requested quantity.
    """
    if display is not None and quantity in display:
        return display[quantity]
    return QUANTITY_DISPLAY.get(
        quantity, QuantityDisplay(quantity, "identity", "linear")
    )


def transform_quantity(values, quantity, display=None):
    """Convert stored samples or thresholds to their display coordinates.

    Args:
        values: Stored quantity values of any shape.
        quantity: Quantity name used to resolve the display transform.
        display: Optional display mapping overriding the defaults.

    Returns:
        NumPy array in displayed coordinates.

    Raises:
        ValueError: If the display specifies an unknown transform.
    """
    values = np.asarray(values, dtype=float)
    transform = quantity_display(quantity, display).transform
    if transform == "exp":
        return np.exp(values)
    if transform == "identity":
        return values
    raise ValueError(f"unknown display transform {transform!r}")


def report_interval(report, channel="y_cert_pos", interval="random"):
    """Read an estimate and its stored endpoints without recomputing uncertainty.

    Args:
        report: CertifiedYield, PairedDifference, float, (point, lo, hi), or None.
        channel: Report channel to read.
        interval: ``random``, ``total``, or ``none``. Bare triples supply their
            own endpoints for either interval kind.

    Returns:
        Tuple (point, lo, hi). Missing rows use a NaN point; absent or disabled
        intervals use None for both endpoints.

    Raises:
        ValueError: If the interval mode is invalid or a report has no stored
            interval for the requested channel. The latter names interval_kind.
    """
    if interval not in ("random", "total", "none"):
        raise ValueError("interval must be 'random', 'total', or 'none'")
    if report is None:
        return np.nan, None, None
    if np.isscalar(report):
        return float(report), None, None
    if isinstance(report, tuple | list | np.ndarray):
        point, lo, hi = _array(report, "report", (3,))
        return (point, None, None) if interval == "none" else (point, lo, hi)
    yields = _field(report, "yields")
    point = float(yields[channel] if yields is not None else _field(report, channel))
    if interval == "none":
        return point, None, None
    intervals = _field(
        report, "intervals" if interval == "random" else "total_intervals"
    )
    if not intervals or channel not in intervals:
        kind = _field(report, "interval_kind", "unknown")
        raise ValueError(
            f"{interval} interval unavailable for {channel}: interval_kind={kind!r}; "
            'use interval="none"'
        )
    lo, hi = intervals[channel]
    return point, lo, hi


def verdict_series(outcome, channels=("pos", "neg")):
    """Read per-planet verdict probabilities, ignoring replicate arrays.

    Args:
        outcome: CertificationOutcome or mapping of channel names to (P,) arrays.
            A mapping may include in_range; otherwise all planets are in range.
        channels: Subset of pos, neg, and wrong to extract.

    Returns:
        Tuple of a channel-to-array dict and a boolean in_range array, all (P,).

    Raises:
        ValueError: If channels are unknown or arrays disagree in shape.
    """
    series = {}
    shape = None
    for channel in channels:
        if channel not in ("pos", "neg", "wrong"):
            raise ValueError(f"unknown verdict channel {channel!r}")
        name = channel if isinstance(outcome, Mapping) else f"p_{channel}"
        value = _array(_field(outcome, name), name, shape)
        if value.ndim != 1:
            raise ValueError(f"{name}: expected shape (P,), received {value.shape}")
        shape = value.shape
        series[channel] = value
    if shape is None:
        raise ValueError("channels must not be empty")
    in_range = _array(
        _field(outcome, "in_range", np.ones(shape, dtype=bool)), "in_range", shape, bool
    )
    return series, in_range


def planet_plane_data(
    a_eeid, rp_re, *, values=None, active=None, in_range=None, marked=None
):
    """Collect plane coordinates and masks without dropping any planets.

    Args:
        a_eeid: Distance ratios shaped (P,).
        rp_re: Radii in Earth radii shaped (P,).
        values: Probabilities shaped (P,) or a CertificationOutcome.
        active: Boolean (P,) mask, default all true.
        in_range: Boolean (P,) mask overriding the outcome, default all true.
        marked: Boolean (P,) mask, default all false.

    Returns:
        Dict of a_eeid, rp_re, values, active, in_range, and marked. Arrays retain
        all P entries, including inactive and out-of-range planets.

    Raises:
        ValueError: If arrays disagree in shape or coordinates are not 1D.
    """
    a = _array(a_eeid, "a_eeid")
    if a.ndim != 1:
        raise ValueError(f"a_eeid: expected shape (P,), received {a.shape}")
    rp = _array(rp_re, "rp_re", a.shape)
    if values is not None and hasattr(values, "p_pos"):
        series, outcome_range = verdict_series(values, ("pos",))
        values = series["pos"]
        if in_range is None:
            in_range = outcome_range
    result = {"a_eeid": a, "rp_re": rp}
    result["values"] = None if values is None else _array(values, "values", a.shape)
    for name, value, default in (
        ("active", active, True),
        ("in_range", in_range, True),
        ("marked", marked, False),
    ):
        result[name] = _array(
            np.full(a.shape, default) if value is None else value, name, a.shape, bool
        )
    return result


def out_of_range_spans(x, in_range, *, log=False):
    """Compute contiguous out-of-range intervals using midpoint cell edges.

    Args:
        x: Strictly increasing finite sweep coordinates shaped (S,), S >= 2.
            Empty sweeps are allowed; a singleton has no defined cell width.
        in_range: Boolean mask shaped (S,).
        log: Use geometric midpoints and log-space end extrapolation.

    Returns:
        Array shaped (N, 2) of interval endpoints. End cells extend by half
        the neighboring spacing, in linear or log coordinates.

    Raises:
        ValueError: If coordinates, mask shape, or logarithmic values are invalid.
    """
    x = _array(x, "x")
    valid = _array(in_range, "in_range", x.shape, bool)
    if x.ndim != 1 or x.size == 1:
        raise ValueError("x must have shape (S,) with S >= 2, or be empty")
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("x must be finite and strictly increasing")
    if log and np.any(x <= 0):
        raise ValueError("logarithmic x must be positive")
    if x.size == 0:
        return np.empty((0, 2))
    coordinates = np.log(x) if log else x
    mids = (coordinates[:-1] + coordinates[1:]) / 2
    edges = np.concatenate(
        ([2 * coordinates[0] - mids[0]], mids, [2 * coordinates[-1] - mids[-1]])
    )
    if log:
        edges = np.exp(edges)
    changes = np.diff(np.r_[False, ~valid, False].astype(int))
    return np.column_stack(
        (edges[np.flatnonzero(changes == 1)], edges[np.flatnonzero(changes == -1)])
    )


def profile_bounds(profiles, reports):
    """Multiply each profile's stated issued-error bound by issued yield.

    Args:
        profiles: Sequence of Profile objects.
        reports: Matching sequence of reports with y_issued fields.

    Returns:
        Bound array shaped (S,), including the claim-count factor for separate
        support rules. No empirical wrong-to-issued ratio is substituted.

    Raises:
        ValueError: If sequence lengths disagree.
    """
    if len(profiles) != len(reports):
        raise ValueError("profiles and reports must have the same length")
    return np.asarray(
        [
            profile.issued_error_bound() * _field(report, "y_issued")
            for profile, report in zip(profiles, reports, strict=True)
        ],
        dtype=float,
    )


def zero_floor_values(values, zero_floor):
    """Move exact zeros to a positive display floor, retaining their mask.

    Args:
        values: Nonnegative finite counts of any shape.
        zero_floor: Positive finite display value for exact zeros.

    Returns:
        Tuple of displayed values and the boolean mask of original zeros.
        Positive values are unchanged, including values below the floor.

    Raises:
        ValueError: If the floor or counts are invalid.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(zero_floor) or zero_floor <= 0:
        raise ValueError("zero_floor must be finite and positive")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("values must be finite and nonnegative")
    zeros = values == 0
    return np.where(zeros, zero_floor, values), zeros


def plan_visit_data(plan, planets=None):
    """Flatten finite visit epochs in the requested planet row order.

    Args:
        plan: PlanSummary or mapping with visit_epochs_d, visit_success,
            visit_photometric_snr shaped (P, V), and spectrum_visit shaped (P,).
            Optional k_precision_ms shaped (P,) supplies the mass-measured mask.
        planets: Integer planet indices shaped (M,), default all planets.

    Returns:
        Dict with flattened epoch_d, row, planet, visit, success, snr, and
        spectrum arrays. planets and mass_measured are shaped (M,) in row order.
        Padding is omitted; spectrum flags identify only finite selected visits.

    Raises:
        ValueError: If shapes or selected planet indices are invalid.
    """
    epochs = _array(_field(plan, "visit_epochs_d"), "visit_epochs_d")
    if epochs.ndim != 2:
        raise ValueError(
            f"visit_epochs_d: expected shape (P, V), received {epochs.shape}"
        )
    success = _array(_field(plan, "visit_success"), "visit_success", epochs.shape, bool)
    snr = _array(
        _field(plan, "visit_photometric_snr"), "visit_photometric_snr", epochs.shape
    )
    spectra = _array(
        _field(plan, "spectrum_visit"), "spectrum_visit", (len(epochs),), int
    )
    selected = np.arange(len(epochs)) if planets is None else np.asarray(planets)
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("planets must be a one-dimensional array of integer indices")
    if np.any((selected < 0) | (selected >= len(epochs))):
        raise ValueError("planets contains an out-of-range index")
    row, visit = np.nonzero(np.isfinite(epochs[selected]))
    planet = selected[row]
    precision = _array(
        _field(plan, "k_precision_ms", np.full(len(epochs), np.inf)),
        "k_precision_ms",
        (len(epochs),),
    )
    return {
        "epoch_d": epochs[planet, visit],
        "row": row,
        "planet": planet,
        "visit": visit,
        "success": success[planet, visit],
        "snr": snr[planet, visit],
        "spectrum": visit == spectra[planet],
        "planets": selected,
        "mass_measured": np.isfinite(precision[selected]),
    }


def _claim_arrays(outcome, verdicts):
    """Normalize typed (P, K) fields or bare (K, P) channels to claim-first arrays."""
    names = _field(outcome, "claim_names")
    if names is None:
        raise ValueError("claim fields and claim_names are required")
    names = tuple(names)
    series = {}
    shape = None
    for verdict in verdicts:
        if verdict not in ("pos", "neg"):
            raise ValueError(f"unknown claim verdict {verdict!r}")
        key = verdict if isinstance(outcome, Mapping) else f"claim_p_{verdict}"
        values = _array(_field(outcome, key), key)
        if not isinstance(outcome, Mapping):
            values = values.T
        if values.ndim != 2 or values.shape[0] != len(names):
            raise ValueError(
                f"{key}: expected shape ({len(names)}, P), received {values.shape}"
            )
        values = _array(values, key, shape)
        shape = values.shape
        series[verdict] = values
    if shape is None:
        raise ValueError("verdicts must not be empty")
    key = "in_range" if isinstance(outcome, Mapping) else "claim_in_range"
    in_range = _array(_field(outcome, key), key, dtype=bool)
    if not isinstance(outcome, Mapping):
        in_range = in_range.T
    return names, series, _array(in_range, key, shape, bool)


def tier_agreement_series(outcome, reference, *, claims=None, verdicts=("pos", "neg")):
    """Read paired per-claim probabilities and their approximate-tier range mask.

    Args:
        outcome: CertificationOutcome with claim fields, or mapping with
            claim_names, pos/neg arrays shaped (K, P), and in_range shaped (K, P).
        reference: Same form, with matching planet and claim axes.
        claims: Optional subset of claim names, in requested panel order.
        verdicts: Nonempty subset of pos and neg.

    Returns:
        Tuple of selected names, verdict-to-(x, y) array pairs shaped (K, P),
        and an in_range mask shaped (K, P). Each verdict uses the same mask;
        the hollow count is its false count times the number of verdicts.

    Raises:
        ValueError: If claim fields are missing, names disagree, shapes disagree,
            or a requested claim is unknown.
    """
    names, left, in_range = _claim_arrays(outcome, verdicts)
    other_names, right, _ = _claim_arrays(reference, verdicts)
    if names != other_names:
        raise ValueError(f"claim_names disagree: {names!r} and {other_names!r}")
    for verdict in verdicts:
        _array(right[verdict], f"reference {verdict}", left[verdict].shape)
    selected = names if claims is None else tuple(claims)
    unknown = [name for name in selected if name not in names]
    if unknown:
        raise ValueError(f"unknown claims {unknown!r}; received claim_names {names!r}")
    indices = np.asarray([names.index(name) for name in selected], dtype=int)
    return (
        selected,
        {
            verdict: (left[verdict][indices], right[verdict][indices])
            for verdict in verdicts
        },
        in_range[indices],
    )
