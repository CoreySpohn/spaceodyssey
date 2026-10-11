"""Tests for the mission belief containers."""

from spaceodyssey.belief import MissionBelief
from spaceodyssey.certify import SnrThreshold
from spaceodyssey.data import Observation
from tests.test_data import make_action, make_dataset


def make_observation(target_id: int = 3, snr: float = 8.0) -> Observation:
    """An observation record for the given target."""
    import equinox as eqx

    action = eqx.tree_at(lambda a: a.target_id, make_action(), target_id)
    return Observation(
        action=action,
        epoch_d=12.0,
        duration_d=1.0,
        dataset=make_dataset(snr=snr),
    )


class TestMissionBelief:
    """Pure functional updates of the belief containers."""

    def test_record_appends_per_target(self):
        """Recording stores the observation under its target."""
        belief = MissionBelief().record(make_observation(target_id=5))
        assert len(belief.target(5).observations) == 1
        assert belief.target(6).observations == ()

    def test_record_is_pure(self):
        """The original belief is untouched by record()."""
        original = MissionBelief()
        original.record(make_observation())
        assert original.per_target == {}

    def test_certificates_round_trip(self):
        """Certificate states store and read back per target and name."""
        cert = SnrThreshold(threshold=7.0)
        state = cert.update(cert.initial(), make_dataset(snr=9.0))
        belief = MissionBelief().with_certificate(4, "detection", state)
        stored = belief.certificate(4, "detection")
        assert stored is state
        assert belief.certificate(4, "other") is None

    def test_certified_filters_and_sorts(self):
        """certified() returns sorted ids of crossed certificates only."""
        cert = SnrThreshold(threshold=7.0)
        crossed = cert.update(cert.initial(), make_dataset(snr=9.0))
        open_state = cert.update(cert.initial(), make_dataset(snr=2.0))
        belief = (
            MissionBelief()
            .with_certificate(9, "detection", crossed)
            .with_certificate(2, "detection", crossed)
            .with_certificate(5, "detection", open_state)
        )
        assert belief.certified("detection") == (2, 9)

    def test_reserved_slots_default_none(self):
        """The population and instrument slots exist and default to None."""
        belief = MissionBelief()
        assert belief.population is None
        assert belief.instrument is None
