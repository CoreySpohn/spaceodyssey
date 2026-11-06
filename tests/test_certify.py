"""Tests for the certification seam and its reference implementation."""

from spaceodyssey.certify import CertificateState, SnrThreshold
from tests.test_data import make_dataset


class TestSupportVersusAnnouncement:
    """An announcement is permanent; current support can be revoked."""

    def test_support_defaults_to_announcement(self):
        """A state built without explicit support mirrors crossed."""
        state = CertificateState(statistic=1.0, crossed=True)
        assert bool(state.supported)
        assert not bool(CertificateState(statistic=0.0, crossed=False).supported)

    def test_revise_revokes_support_but_keeps_announcement(self):
        """Weakened evidence clears support and leaves crossed latched."""
        state = CertificateState(statistic=0.0, crossed=False)
        state = state.revise(statistic=0.99, supported=True, epoch_d=10.0)
        assert bool(state.crossed) and bool(state.supported)
        state = state.revise(statistic=0.80, supported=False, epoch_d=20.0)
        assert bool(state.crossed)
        assert not bool(state.supported)
        assert state.epoch_d == 20.0

    def test_revise_never_announces_without_support(self):
        """Unsupported evidence on an open certificate leaves it open."""
        state = CertificateState(statistic=0.0, crossed=False)
        state = state.revise(statistic=0.5, supported=False, epoch_d=1.0)
        assert not bool(state.crossed)

    def test_snr_threshold_support_tracks_announcement(self):
        """SnrThreshold keeps its irreversible meaning: support never drops."""
        cert = SnrThreshold(threshold=7.0)
        state = cert.update(cert.initial(), make_dataset(snr=9.0))
        state = cert.update(state, make_dataset(snr=2.0, epoch_d=40.0))
        assert bool(state.crossed)
        assert bool(state.supported)
