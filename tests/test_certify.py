"""Tests for the certification seam and its reference implementation."""

import pytest

from spaceodyssey.certify import AbstractCertificate, CertificateState, SnrThreshold
from spaceodyssey.errors import UnsupportedOperator
from spaceodyssey.registry import REGISTRY, SeamInfo
from spaceodyssey.testing import check_certificate
from tests.test_data import make_dataset


class TestSnrThreshold:
    """Threshold-crossing behavior of the reference certificate."""

    def test_crosses_at_threshold(self):
        """A statistic at or above the bar hardens the certificate."""
        cert = SnrThreshold(threshold=7.0)
        state = cert.update(cert.initial(), make_dataset(snr=7.0))
        assert bool(state.crossed)

    def test_below_threshold_stays_open(self):
        """A weaker statistic leaves the certificate open."""
        cert = SnrThreshold(threshold=7.0)
        state = cert.update(cert.initial(), make_dataset(snr=5.0))
        assert not bool(state.crossed)
        assert float(state.statistic) == pytest.approx(5.0)

    def test_statistic_is_monotone(self):
        """Weaker later data never lowers the statistic or un-certifies."""
        cert = SnrThreshold(threshold=7.0)
        state = cert.update(cert.initial(), make_dataset(snr=9.0))
        state = cert.update(state, make_dataset(snr=2.0, epoch_d=40.0))
        assert float(state.statistic) == pytest.approx(9.0)
        assert bool(state.crossed)
        assert state.epoch_d == 40.0

    def test_other_operators_unsupported(self):
        """Propagation and integration are declared out of scope."""
        cert = SnrThreshold(threshold=7.0)
        with pytest.raises(UnsupportedOperator):
            cert.propagate(cert.initial(), moments=None)
        with pytest.raises(UnsupportedOperator):
            cert.integrate(predictive=None)

    def test_registered_by_default(self):
        """The reference implementation is discoverable in the registry."""
        entry = REGISTRY.get("certification", "snr_threshold")
        assert entry.cls is SnrThreshold
        assert entry.info.operators == frozenset({"sample"})


class TestConformance:
    """The certificate conformance suite."""

    def test_snr_threshold_passes(self):
        """The reference implementation satisfies the suite."""
        cert = SnrThreshold(threshold=7.0)
        info = REGISTRY.get("certification", "snr_threshold").info
        state = check_certificate(
            cert, info, [make_dataset(snr=4.0), make_dataset(snr=9.0)]
        )
        assert bool(state.crossed)

    def test_dishonest_declaration_fails(self):
        """Declaring an operator that raises is caught by the suite."""

        class Overclaims(AbstractCertificate):
            """Declares propagate support it does not have."""

            def initial(self):
                """Return an empty state."""
                return CertificateState(statistic=0.0, crossed=False)

            def update(self, state, dataset):
                """Accept any dataset without changing state."""
                return state

            def propagate(self, state, moments):
                """Raise despite the declared support."""
                raise UnsupportedOperator("not actually implemented")

            def integrate(self, predictive):
                """Raise, undeclared."""
                raise UnsupportedOperator("not implemented")

        info = SeamInfo(operators={"sample", "propagate"})
        with pytest.raises(AssertionError, match="propagate"):
            check_certificate(Overclaims(), info, [make_dataset()])
