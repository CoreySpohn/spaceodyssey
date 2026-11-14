"""Named measurements on detection and band axes, with scorer-only truth."""

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.core.enums import EVERYONE, F64, I64, Audience, RecordStatus
from spaceodyssey.core.records import RecordKind

MEASUREMENT_KIND_NAME = "measurement"


def usable_measurement(status, valid_science, variance) -> jax.Array:
    """Completed rows, or partial rows with science and finite positive variance.

    Variance has trailing detection and band axes; the result has the leading
    row axes. Unexposed bands may be zero. Eligibility uses delivered columns.
    """
    valid_variance = jnp.any(jnp.isfinite(variance) & (variance > 0), axis=(-2, -1))
    return (status == RecordStatus.COMPLETED) | (
        (status == RecordStatus.PARTIAL) & (valid_science > 0) & valid_variance
    )


class MeasurementRow(eqx.Module):
    """One exposure: scalar header, detection [D] and signal [D, B] leaves.

    Counts are electrons and variance is electrons squared. Astrometry is arcsec;
    time_d is the science midpoint in JD. Mode and duration_index address the
    channel's ModeTable. Identity and true signal leaves are scorer-only.
    """

    time_d: jax.Array
    science_s: jax.Array
    occupied_s: jax.Array
    cadence: jax.Array
    mode: jax.Array
    duration_index: jax.Array
    calibration_id: jax.Array
    valid_science: jax.Array
    detected: jax.Array
    assignment: jax.Array
    snr: jax.Array
    ra: jax.Array
    dec: jax.Array
    ra_err: jax.Array
    dec_err: jax.Array
    corr: jax.Array
    limit_counts: jax.Array
    counts: jax.Array
    variance: jax.Array
    identity: jax.Array
    true_counts: jax.Array
    true_variance: jax.Array


def measurement_template(n_detections, n_bands) -> MeasurementRow:
    """Shape and dtype declaration of one measurement payload."""
    scalar = jax.ShapeDtypeStruct((), F64)
    index = jax.ShapeDtypeStruct((), I64)
    detections = jax.ShapeDtypeStruct((n_detections,), F64)
    assignments = jax.ShapeDtypeStruct((n_detections,), I64)
    signal = jax.ShapeDtypeStruct((n_detections, n_bands), F64)
    return MeasurementRow(
        time_d=scalar,
        science_s=scalar,
        occupied_s=scalar,
        cadence=index,
        mode=index,
        duration_index=index,
        calibration_id=index,
        valid_science=scalar,
        detected=detections,
        assignment=assignments,
        snr=detections,
        ra=detections,
        dec=detections,
        ra_err=detections,
        dec_err=detections,
        corr=detections,
        limit_counts=detections,
        counts=signal,
        variance=signal,
        identity=assignments,
        true_counts=signal,
        true_variance=signal,
    )


def measurement_kind(
    n_detections,
    n_bands,
    measured_audience=EVERYONE,
    true_audience=Audience.SCORER,
) -> RecordKind:
    """A measurement layout whose generating identities and signals stay private."""
    return RecordKind.from_template(
        MEASUREMENT_KIND_NAME,
        measurement_template(n_detections, n_bands),
        lambda path: (
            true_audience
            if path == "identity" or path.startswith("true_")
            else measured_audience
        ),
    )
