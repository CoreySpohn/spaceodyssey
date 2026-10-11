"""Observation records and the likelihood-factory waist.

The simulator never interprets data. Every dataset kind exposes
``to_likelihood``, a factory producing a plain log-density callable for that
datum, and the belief layer consumes only those callables. This is how
heterogeneous data (summary measurements, spectra, image frames) flow into
one belief without the mission loop knowing any physics.
"""

import math
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from planit_py.vocabulary import Action


@final
class SummaryDataset(eqx.Module):
    """Summary-level measurements from one observation.

    Attributes:
        values: Measured summary quantities, shape (d,).
        sigma: One-sigma uncertainty per quantity, shape (d,).
        snr: Scalar detection statistic for the observation.
        epoch_d: Epoch of the measurement in days of mission time.
        labels: Names of the summary quantities, for bookkeeping and plots.
        band: Name of the band the observation was taken in.
    """

    values: jax.Array = eqx.field(converter=jnp.asarray)
    sigma: jax.Array = eqx.field(converter=jnp.asarray)
    snr: jax.Array = eqx.field(converter=jnp.asarray)
    epoch_d: float
    labels: tuple[str, ...] = eqx.field(static=True, default=())
    band: str = eqx.field(static=True, default="")

    def __check_init__(self):
        """Validate shapes and label consistency."""
        if self.values.shape != self.sigma.shape:
            raise ValueError(
                f"values and sigma must share a shape, got {self.values.shape} "
                f"and {self.sigma.shape}"
            )
        if self.labels and len(self.labels) != self.values.shape[-1]:
            raise ValueError(
                f"{len(self.labels)} labels for {self.values.shape[-1]} values"
            )

    def to_likelihood(self, predict):
        """Return a log-density callable for this datum.

        Args:
            predict: Maps a parameter pytree to predicted summary values of
                the same shape as ``values``; supplied by the measurement
                model, so this dataset stays physics-free.

        Returns:
            A callable ``loglike(params) -> scalar`` computing the Gaussian
            log-likelihood of the measured values under the prediction. The
            callable is differentiable through ``params``.
        """
        values = self.values
        sigma = self.sigma
        norm = -jnp.sum(jnp.log(sigma)) - 0.5 * values.shape[-1] * math.log(
            2.0 * math.pi
        )

        def loglike(params):
            residual = (values - predict(params)) / sigma
            return norm - 0.5 * jnp.sum(residual**2)

        return loglike


@final
class Observation(eqx.Module):
    """The record of one executed observation.

    Attributes:
        action: The action that was executed.
        epoch_d: Epoch at which the observation started, in days.
        duration_d: The realized integration duration in days; this is what
            the ledger charges, and with adaptive integration policies it
            differs from any requested value.
        dataset: The dataset produced (a summary dataset, spectrum, frame
            set, ...); consumers interpret it only through its contract type
            or its likelihood factory.
        meta: Free-form provenance entries.
    """

    action: Action
    epoch_d: float
    duration_d: float
    dataset: Any
    meta: dict = eqx.field(default_factory=dict)
