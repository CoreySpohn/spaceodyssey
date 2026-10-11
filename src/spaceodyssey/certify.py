"""The certification seam: when does an accumulating result count.

A certificate is a persistent per-target state that hardens as evidence
accumulates and gates what the mission announces; it is distinct from any
score a policy uses to rank actions, and the two are never converted into
each other. One certificate definition supports up to three evaluations,
matching the outcome operators: update against realized data, propagate
against predictive moments, and integrate to a certification probability.
"""

import abc
from typing import ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp

from spaceodyssey.errors import UnsupportedOperator
from spaceodyssey.registry import SeamInfo, register


@final
class CertificateState(eqx.Module):
    """The running state of one certificate on one target.

    Attributes:
        statistic: The certificate's accumulated statistic.
        crossed: Whether the certificate has hardened past its threshold.
        epoch_d: Epoch of the last update, in days of mission time.
    """

    statistic: jax.Array = eqx.field(converter=jnp.asarray)
    crossed: jax.Array = eqx.field(converter=jnp.asarray)
    epoch_d: float = 0.0


class AbstractCertificate(eqx.Module):
    """One certificate definition, evaluable under multiple operators."""

    PROTOCOL_VERSION: ClassVar[int] = 1

    @abc.abstractmethod
    def initial(self) -> CertificateState:
        """Return the state before any evidence."""

    @abc.abstractmethod
    def update(self, state: CertificateState, dataset) -> CertificateState:
        """Advance the state with one realized dataset."""

    @abc.abstractmethod
    def propagate(self, state: CertificateState, moments) -> CertificateState:
        """Advance the state with predictive moments instead of a draw."""

    @abc.abstractmethod
    def integrate(self, predictive) -> jax.Array:
        """Return the certification probability under a predictive."""


@final
@register(
    "certification",
    "snr_threshold",
    SeamInfo(
        operators={"sample"},
        consumes=("spaceodyssey.SummaryDataset",),
        produces=("spaceodyssey.CertificateState",),
        version="1",
        fidelity="threshold",
        cost_hint="low",
    ),
)
class SnrThreshold(AbstractCertificate):
    """Certify a detection when the best observed statistic crosses a bar.

    The running statistic is the maximum detection statistic seen so far;
    weaker later data never un-certifies a target.

    Attributes:
        threshold: The statistic value at which the certificate hardens.
    """

    threshold: float

    def initial(self) -> CertificateState:
        """Return an uncrossed state with a zero statistic."""
        return CertificateState(statistic=0.0, crossed=False, epoch_d=0.0)

    def update(self, state: CertificateState, dataset) -> CertificateState:
        """Fold one summary dataset's detection statistic into the state."""
        statistic = jnp.maximum(state.statistic, dataset.snr)
        return CertificateState(
            statistic=statistic,
            crossed=statistic >= self.threshold,
            epoch_d=dataset.epoch_d,
        )

    def propagate(self, state: CertificateState, moments) -> CertificateState:
        """Raise: this certificate evaluates realized data only."""
        raise UnsupportedOperator("snr_threshold supports only the sample operator")

    def integrate(self, predictive) -> jax.Array:
        """Raise: this certificate evaluates realized data only."""
        raise UnsupportedOperator("snr_threshold supports only the sample operator")
