"""Tests for the engine registry and the end-to-end sampling path."""

import pytest

import spaceodyssey
from spaceodyssey import engines
from spaceodyssey.engines import analytic
from spaceodyssey.errors import RegistryError
from spaceodyssey.metrics import YieldSummary
from tests.doubles import PLANET_TARGETS, build_test_registry, make_config, make_spec


class TestEngineRegistry:
    """Engines are a registry keyed by operator name."""

    def test_sample_engine_registered(self):
        """The sampling engine loads on first lookup."""
        assert callable(engines.get_engine("sample"))

    def test_integrate_engine_registered_once(self):
        """Default loading shares the integrate callable and default registry."""
        assert engines.get_engine("integrate") is analytic.run_integrate
        assert engines.get_engine("integrate") is analytic.run_integrate
        assert list(engines.ENGINES.values()).count(analytic.run_integrate) == 1
        assert engines.REGISTRY is spaceodyssey.REGISTRY

    def test_unknown_operator_lists_known(self):
        """An unknown operator error names the registered ones."""
        with pytest.raises(RegistryError, match="sample"):
            engines.get_engine("teleport")

    def test_duplicate_operator_rejected(self):
        """Registering the same operator twice is an error."""
        with pytest.raises(RegistryError, match="sample"):

            @engines.register_engine("sample")
            def duplicate(mission, *, seed=0):
                """Never registered; the decorator raises first."""
                return None


class TestEndToEnd:
    """spec + config in, YieldSummary out."""

    def test_run_returns_scored_summary(self):
        """The full path certifies exactly the planet-bearing targets."""
        summary = engines.run(
            make_spec(duration_d=6.0),
            make_config(),
            seed=0,
            registry=build_test_registry(),
        )
        assert isinstance(summary, YieldSummary)
        assert summary.operator == "sample"
        assert summary.certified_targets == PLANET_TARGETS
        assert summary.false_certified_targets == ()
        assert summary.certified == 2
        assert summary.epochs == 6
        assert dict(summary.resources_spent)["time_d"] == pytest.approx(6.6)

    def test_false_certifications_are_counted(self):
        """A measurement that certifies empty targets is scored, not hidden."""
        summary = engines.run(
            make_spec(duration_d=6.0),
            make_config(snr_absent=9.0),
            seed=0,
            registry=build_test_registry(),
        )
        assert summary.certified_targets == (1, 2, 3)
        assert summary.false_certified_targets == (2,)
        assert summary.false_certifications == 1

    def test_wrong_claims_scored_apart_from_absent_sources(self):
        """Claim-level scoring counts real targets only, never absent ones."""
        summary = engines.run(
            make_spec(duration_d=6.0),
            make_config(snr_absent=9.0),
            seed=0,
            registry=build_test_registry(),
            is_wrong=lambda scene: scene == "planet",
        )
        assert summary.false_certified_targets == (2,)
        assert summary.wrong_certified_targets == (1, 3)
        assert summary.wrong_certifications == 2

    def test_claim_scoring_off_by_default(self):
        """Without a claim test no certification is scored wrong."""
        summary = engines.run(
            make_spec(duration_d=6.0),
            make_config(),
            seed=0,
            registry=build_test_registry(),
        )
        assert summary.wrong_certified_targets == ()

    def test_validation_runs_before_the_engine(self):
        """An incoherent config is rejected before anything executes."""
        from spaceodyssey.errors import WiringError
        from spaceodyssey.spec import FidelityConfig, SeamRef

        config = FidelityConfig.make(
            "sample", scene=SeamRef.make("scene", "nonexistent")
        )
        with pytest.raises(WiringError):
            engines.run(make_spec(), config, seed=0, registry=build_test_registry())


class TestYieldSummary:
    """The report schema."""

    def test_json_round_trip(self):
        """A summary survives serialization exactly."""
        summary = YieldSummary(
            operator="sample",
            certificate="detection",
            certified_targets=(1, 3),
            false_certified_targets=(3,),
            epochs=6,
            resources_spent=(("time_d", 6.6),),
        )
        assert YieldSummary.from_json(summary.to_json()) == summary
        assert summary.false_certifications == 1

    def test_reads_reports_without_claim_scoring(self):
        """Reports written before claim-level scoring still load."""
        text = (
            '{"certificate":"detection","certified_targets":[1],"epochs":1,'
            '"false_certified_targets":[],"operator":"sample",'
            '"resources_spent":[["time_d",1.0]]}'
        )
        assert YieldSummary.from_json(text).wrong_certified_targets == ()
