"""Numerical inputs for visualization, using NumPy without a renderer.

Device arrays are converted to host arrays here. Replicate arrays are never
treated as planets, and stored report intervals are never estimated again.
"""

from collections.abc import Mapping

import numpy as np

from spaceodyssey.viz.labels import CHANNEL_LABELS, QUANTITY_DISPLAY, QuantityDisplay


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
    if isinstance(report, tuple | list) or hasattr(report, "shape"):
        report = np.asarray(report, dtype=float)
        if report.ndim == 0:
            return float(report), None, None
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


def yield_comparison_data(
    rows,
    channel="y_cert_pos",
    *,
    references=None,
    margin=None,
    predicted=(),
    difference=None,
    difference_channels=("y_cert_pos", "y_wrong", "y_neg"),
    interval="random",
    gap=None,
):
    """Read comparison rows and paired channels without estimating uncertainty.

    Args:
        rows: Nonempty mapping of labels to reports, points, triples, or None.
        channel: Yield channel for the first panel.
        references: Optional mapping of row labels to reference points or reports.
        margin: Nonnegative scalar, or mapping keyed by row and channel names.
        predicted: Row names whose estimates are predictions.
        difference: PairedDifference or channel-to-point/triple mapping.
        difference_channels: Nonempty channel sequence for the optional panel.
        interval: Stored interval selection: random, total, or none.
        gap: Optional pair of row names with finite point estimates.

    Returns:
        Tuple of panels and optional (left point, right point, signed gap).
        Each panel is an ordered mapping of names to (point, lo, hi, reference,
        margin) tuples. Missing points retain their rows as NaN.

    Raises:
        ValueError: If channels, keys, margins, or stored intervals are invalid.

    Example::

        panels, gap = yield_comparison_data({"trial": (2, 1, 3)}, margin=0.5)
    """
    if channel not in CHANNEL_LABELS:
        raise ValueError(f"unknown yield channel {channel!r}")
    if not isinstance(rows, Mapping) or not rows:
        raise ValueError("rows must be a nonempty mapping")
    if interval not in ("random", "total", "none"):
        raise ValueError("interval must be 'random', 'total', or 'none'")
    references = {} if references is None else references
    if not isinstance(references, Mapping) or set(references) - rows.keys():
        raise ValueError("references must be a mapping keyed by row names")
    if set(predicted) - rows.keys():
        raise ValueError("predicted must contain only row names")
    channels = tuple(difference_channels) if difference is not None else ()
    if difference is not None and (
        not channels or any(name not in CHANNEL_LABELS for name in channels)
    ):
        raise ValueError("difference_channels must be a nonempty subset of channels")
    if isinstance(margin, Mapping):
        if set(margin) - (rows.keys() | set(channels)):
            raise ValueError("margin keys must be row names or difference channels")
        margins = dict(margin)
    else:
        margins = dict.fromkeys((*rows, *channels), margin)
    for name, value in margins.items():
        if value is not None:
            value = np.asarray(value, dtype=float)
            if value.ndim or not np.isfinite(value) or value < 0:
                raise ValueError("margin must be finite and nonnegative")
            margins[name] = float(value)
    panels = [{}]
    for name, report in rows.items():
        point, lo, hi = report_interval(report, channel, interval)
        ref = None
        if name in references:
            ref, _, _ = report_interval(references[name], channel, "none")
            if not np.isfinite(ref):
                raise ValueError("references must be finite")
        panels[0][name] = (point, lo, hi, ref, margins.get(name))
    if difference is not None:
        panels.append({})
        for name in channels:
            if isinstance(difference, Mapping):
                if name not in difference:
                    raise ValueError(f"difference missing channel {name!r}")
                value = difference[name]
            else:
                if name not in difference.yields:
                    raise ValueError(f"difference missing channel {name!r}")
                value = difference
            point, lo, hi = report_interval(value, name, interval)
            panels[1][name] = (point, lo, hi, 0.0, margins.get(name))
    gap_values = None
    if gap is not None:
        if len(gap) != 2 or any(name not in rows for name in gap):
            raise ValueError("gap must contain two row names")
        left, right = (panels[0][name][0] for name in gap)
        if not np.isfinite([left, right]).all():
            raise ValueError("gap rows must have finite point estimates")
        gap_values = (left, right, right - left)
    return panels, gap_values


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


def _sweep_channel(reports, channel, interval, size):
    """Read one sweep channel from report rows or bare columns."""
    if isinstance(reports, Mapping):
        values = _array(reports.get(channel), channel)
        if values.shape not in ((size,), (size, 3)):
            raise ValueError(
                f"{channel}: expected shape {(size,)} or {(size, 3)}, "
                f"received {values.shape}"
            )
        rows = [report_interval(value, interval=interval) for value in values]
    else:
        if len(reports) != size:
            raise ValueError(
                f"reports: expected length {size}, received {len(reports)}"
            )
        rows = [report_interval(report, channel, interval) for report in reports]
    point, lo, hi = zip(*rows, strict=True)
    point = np.asarray(point, dtype=float)
    if not np.isfinite(point).all() or np.any(point < 0):
        raise ValueError(f"{channel} yields must be finite and nonnegative")
    # Missing intervals remain absent; NaNs suppress only those interval artists.
    lo = None if all(v is None for v in lo) else np.asarray(lo, dtype=float)
    hi = None if all(v is None for v in hi) else np.asarray(hi, dtype=float)
    return point, lo, hi


def yield_sweep_data(x, reports, channels, bound=None, interval="random"):
    """Extract sweep coordinates, stored intervals, and stated bounds.

    Args:
        x: Nonempty sequence of numbers, strings, or Profiles.
        reports: Report sequence, channel-to-array mapping, or mapping of
            series names to either form. Bare columns have shape (S,) or
            (S, 3), with triples holding point, lower, and upper endpoints.
        channels: Nonempty sequence of yield channel names.
        bound: None, ``profile``, or nonnegative values shaped (S,).
        interval: Stored interval selection: ``random``, ``total``, or ``none``.

    Returns:
        Tuple of numeric coordinates, categorical labels or None, whether x
        contains profiles, series-to-channel data, and series-to-bound arrays.

    Raises:
        ValueError: If channels, coordinates, counts, bounds, or shapes disagree.

    Example::

        data = yield_sweep_data([1, 2], {"y_wrong": [0, 0.1]}, ("y_wrong",))
    """
    if not channels or any(channel not in CHANNEL_LABELS for channel in channels):
        raise ValueError(
            f"channels must be a nonempty subset of {tuple(CHANNEL_LABELS)}"
        )
    if interval not in ("random", "total", "none"):
        raise ValueError("interval must be 'random', 'total', or 'none'")
    raw = np.asarray(x, dtype=object)
    if raw.ndim != 1 or not raw.size:
        raise ValueError(f"x: expected nonempty shape (S,), received {raw.shape}")
    profiles = all(hasattr(value, "issued_error_bound") for value in raw)
    categorical = all(isinstance(value, str) for value in raw)
    labels = tuple(raw) if categorical else None
    coordinates = (
        np.arange(len(raw), dtype=float)
        if categorical
        else _array([value.alpha for value in raw] if profiles else x, "x")
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("x must be finite")
    if isinstance(bound, str) and (bound != "profile" or not profiles):
        raise ValueError('bound="profile" requires x to be a sequence of Profiles')
    if not isinstance(reports, Mapping) or any(k in CHANNEL_LABELS for k in reports):
        reports = {"yield": reports}
    if not reports:
        raise ValueError("reports must contain at least one series")
    series, bounds = {}, {}
    for name, rows in reports.items():
        series[name] = {
            channel: _sweep_channel(rows, channel, interval, len(raw))
            for channel in channels
        }
        if bound is not None:
            if isinstance(bound, str):
                issued, _, _ = _sweep_channel(rows, "y_issued", "none", len(raw))
                values = profile_bounds(raw, [{"y_issued": v} for v in issued])
            else:
                values = _array(bound, "bound", coordinates.shape)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("bound must be finite and nonnegative")
            bounds[name] = values
    return coordinates, labels, profiles, series, bounds


def sweep_floor_data(point, lo, hi, zero_floor):
    """Map zero yields to their display floor and clip interval endpoints.

    Args:
        point: Nonnegative yield array.
        lo: Lower interval endpoints or None.
        hi: Upper interval endpoints or None.
        zero_floor: Positive finite floor for the logarithmic display.

    Returns:
        Displayed points, lower and upper endpoints, and original zero mask.

    Example::

        point, lo, hi, zeros = sweep_floor_data([0, 1], None, None, 0.001)
    """
    displayed, zeros = zero_floor_values(point, zero_floor)
    return (
        displayed,
        None if lo is None else np.maximum(lo, zero_floor),
        None if hi is None else np.maximum(hi, zero_floor),
        zeros,
    )


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
