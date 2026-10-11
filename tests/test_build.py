"""Tests for compiling declarative configurations into stacks."""

import pytest

from spaceodyssey.build import Mission, ModelStack, TruthStack, build
from spaceodyssey.errors import WiringError
from spaceodyssey.spec import FidelityConfig, SeamChoice, SeamRef
from tests.doubles import build_test_registry, make_config, make_spec


class TestBuild:
    """The declarative-to-compiled step."""

    def test_compiles_distinct_stack_types(self):
        """Truth and model compile into distinct types (the firewall)."""
        mission = build(make_spec(), make_config(), build_test_registry())
        assert isinstance(mission, Mission)
        assert isinstance(mission.truth, TruthStack)
        assert isinstance(mission.model, ModelStack)
        assert not isinstance(mission.truth, ModelStack)
        assert not isinstance(mission.model, TruthStack)

    def test_shared_choice_shares_instances(self):
        """Without a model override, both stacks hold one instance."""
        mission = build(make_spec(), make_config(), build_test_registry())
        assert mission.truth.observation is mission.model.observation

    def test_mismatched_pair_compiles_two_impls(self):
        """A (truth, model) override materializes different objects."""
        config = make_config()
        override = SeamChoice(
            truth=SeamRef.make("certification", "snr_threshold", threshold=7.0),
            model=SeamRef.make("certification", "snr_threshold", threshold=5.0),
        )
        choices = tuple(
            (seam, override if seam == "certification" else choice)
            for seam, choice in config.choices
        )
        config = FidelityConfig(operator="sample", choices=choices)
        mission = build(make_spec(), config, build_test_registry())
        assert mission.truth.certification.threshold == 7.0
        assert mission.model.certification.threshold == 5.0

    def test_unchosen_seams_stay_none(self):
        """Seams the config leaves out compile to None slots."""
        mission = build(make_spec(), make_config(), build_test_registry())
        assert mission.truth.post_processing is None
        assert mission.model.population is None

    def test_check_rejects_incoherent_config(self):
        """Validation runs first and rejects an unknown implementation."""
        config = FidelityConfig.make(
            "sample", observation=SeamRef.make("observation", "nonexistent")
        )
        with pytest.raises(WiringError, match="nonexistent"):
            build(make_spec(), config, build_test_registry())

    def test_bad_params_surface_seam_context(self):
        """A constructor rejection names the seam and implementation."""
        config = FidelityConfig.make(
            "sample",
            observation=SeamRef.make("observation", "presence", bogus=1.0),
        )
        with pytest.raises(WiringError, match="observation=presence"):
            build(make_spec(), config, build_test_registry(), check=False)
