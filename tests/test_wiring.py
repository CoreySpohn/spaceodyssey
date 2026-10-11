"""Tests for configuration validation against the registry."""

import pytest

from spaceodyssey.errors import WiringError
from spaceodyssey.registry import Registry, SeamInfo
from spaceodyssey.spec import FidelityConfig, SeamChoice, SeamRef


class SummaryData:
    """Toy contract type: summary-level measurements."""


class FrameData:
    """Toy contract type: image frames."""


class DetStats:
    """Toy contract type: detection statistics."""


def qual(cls) -> str:
    """Module-qualified contract name for a class defined in this module."""
    return f"{__name__}.{cls.__name__}"


class Impl:
    """Placeholder implementation class."""


@pytest.fixture()
def registry():
    """A registry with two observation impls and two post-processing impls."""
    reg = Registry()
    reg.register(
        "observation",
        "summary_rates",
        SeamInfo(
            operators={"sample", "propagate", "integrate"},
            produces=(qual(SummaryData),),
        ),
    )(Impl)
    reg.register(
        "observation",
        "image_frames",
        SeamInfo(operators={"sample"}, produces=(qual(FrameData),)),
    )(Impl)
    reg.register(
        "post_processing",
        "throughput_factor",
        SeamInfo(
            operators={"sample", "propagate", "integrate"},
            consumes=(qual(SummaryData),),
            produces=(qual(SummaryData),),
        ),
    )(Impl)
    reg.register(
        "post_processing",
        "matched_filter",
        SeamInfo(
            operators={"sample"},
            consumes=(qual(FrameData),),
            produces=(qual(DetStats),),
            needs_calibration=("null_threshold",),
        ),
    )(Impl)
    reg.register(
        "certification",
        "snr_threshold",
        SeamInfo(
            operators={"sample"},
            consumes=(qual(SummaryData), qual(DetStats)),
        ),
    )(Impl)
    return reg


def config_for(
    operator="sample", observation="summary_rates", post_processing="throughput_factor"
):
    """A config over observation, post-processing, and certification."""
    return FidelityConfig.make(
        operator,
        observation=SeamRef.make("observation", observation),
        post_processing=SeamRef.make("post_processing", post_processing),
        certification=SeamRef.make("certification", "snr_threshold"),
    )


class TestValidation:
    """The three check families and the report contents."""

    def test_coherent_config_passes(self, registry):
        """A type- and operator-consistent selection validates cleanly."""
        report = config_for().validate(registry)
        assert report.ok
        report.raise_if_failed()

    def test_unchosen_edges_are_skipped(self, registry):
        """Edges touching unchosen seams are reported, not checked."""
        report = config_for().validate(registry)
        assert ("instrument", "observation") in report.skipped_edges
        assert ("scene", "observation") in report.skipped_edges

    def test_port_mismatch_names_both_sides_and_alternatives(self, registry):
        """Frame-based post-processing on summary observations is rejected."""
        report = config_for(post_processing="matched_filter").validate(registry)
        assert not report.ok
        ports = [f for f in report.failures if f.kind == "ports"]
        assert len(ports) == 1
        message = ports[0].message
        assert "matched_filter consumes" in message
        assert "summary_rates produces" in message
        assert "throughput_factor" in message
        assert "image_frames" in message

    def test_operator_mismatch_is_independent_of_ports(self, registry):
        """A port-coherent config still fails on undeclared operators."""
        report = config_for(
            operator="integrate",
            observation="image_frames",
            post_processing="matched_filter",
        ).validate(registry)
        kinds = {f.kind for f in report.failures}
        assert "ports" not in kinds
        assert "operator" in kinds
        offenders = {f.impl for f in report.failures if f.kind == "operator"}
        assert {"image_frames", "matched_filter", "snr_threshold"} <= offenders

    def test_unknown_impl(self, registry):
        """Referencing an unregistered name fails with the known names."""
        config = FidelityConfig.make(
            "sample",
            observation=SeamRef.make("observation", "nonexistent"),
        )
        report = config.validate(registry)
        assert not report.ok
        assert report.failures[0].kind == "unknown_impl"
        assert "summary_rates" in report.failures[0].message

    def test_unavailable_impl(self, registry):
        """A missing optional dependency reports the entry as unavailable."""
        registry.register(
            "characterization",
            "needs_extra",
            SeamInfo(
                operators={"sample"},
                import_requirement="not_a_real_package_xyz",
            ),
        )(Impl)
        config = FidelityConfig.make(
            "sample",
            characterization=SeamRef.make("characterization", "needs_extra"),
        )
        report = config.validate(registry)
        assert [f.kind for f in report.failures] == ["unavailable"]

    def test_unresolvable_contract(self, registry):
        """A contract string naming a missing package is a contract failure."""
        registry.register(
            "characterization",
            "bad_contract",
            SeamInfo(operators={"sample"}, consumes=("no_such_pkg_qq.T",)),
        )(Impl)
        config = FidelityConfig.make(
            "sample",
            observation=SeamRef.make("observation", "summary_rates"),
            characterization=SeamRef.make("characterization", "bad_contract"),
        )
        report = config.validate(registry)
        assert [f.kind for f in report.failures] == ["contract"]

    def test_calibration_needs_flagged_not_failed(self, registry):
        """Unmet calibration needs flag the run without failing it."""
        report = config_for(
            observation="image_frames", post_processing="matched_filter"
        ).validate(registry)
        assert report.ok
        assert report.uncalibrated == ("post_processing:matched_filter",)

    def test_model_stack_validated_when_overridden(self, registry):
        """A broken model-side ref fails on the model stack specifically."""
        config = FidelityConfig.make(
            "sample",
            observation=SeamChoice(
                truth=SeamRef.make("observation", "summary_rates"),
                model=SeamRef.make("observation", "nonexistent"),
            ),
        )
        report = config.validate(registry)
        assert not report.ok
        stacks = {(f.kind, f.stack) for f in report.failures}
        assert ("unknown_impl", "model") in stacks
        assert ("unknown_impl", "truth") not in stacks

    def test_raise_if_failed_collects_messages(self, registry):
        """raise_if_failed raises one WiringError listing every failure."""
        report = config_for(post_processing="matched_filter").validate(registry)
        with pytest.raises(WiringError, match="matched_filter"):
            report.raise_if_failed()
