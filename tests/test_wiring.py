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

    @pytest.mark.parametrize("stack", ["truth", "model"])
    def test_ref_seam_must_match_slot(self, registry, stack):
        """A shared implementation name cannot hide a mismatched ref seam."""
        registry.register(
            "post_processing", "summary_rates", SeamInfo(operators={"sample"})
        )(Impl)
        ref = SeamRef.make("post_processing", "summary_rates")
        choice = SeamChoice(
            truth=ref if stack == "truth" else SeamRef.make("observation", ref.name),
            model=ref if stack == "model" else None,
        )
        config = FidelityConfig.make("sample", observation=choice)
        report = config.validate(registry)
        assert not report.ok
        assert len(report.failures) == 1
        failure = report.failures[0]
        assert failure.kind == "seam_mismatch"
        assert failure.stack == stack
        assert failure.seam == "observation"
        assert failure.impl == "summary_rates"
        assert failure.edge is None
        assert "observation" in failure.message
        assert "post_processing" in failure.message

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


class TestReferenceValidation:
    """Version pins and constructor signatures are checked without constructing."""

    @pytest.mark.parametrize("entrypoint", ["validate", "build"])
    @pytest.mark.parametrize("failure_side", [None, "truth", "model"])
    def test_operator_error_keeps_legacy_message(self, entrypoint, failure_side):
        """Missing parameters never add diagnostics to a legacy operator failure."""
        from spaceodyssey.build import build
        from tests.doubles import make_spec

        calls = []

        class Required:
            def __init__(self, required):
                calls.append(self)

        registry = Registry()
        registry.register("scene", "required", SeamInfo(operators={"sample"}))(Required)
        registry.register("scene", "integrating", SeamInfo(operators={"integrate"}))(
            Required
        )
        ref = SeamRef.make("scene", "required")
        integrating = SeamRef.make("scene", "integrating")
        if failure_side is None:
            choice = ref
        elif failure_side == "truth":
            choice = SeamChoice(ref, integrating)
        else:
            choice = SeamChoice(integrating, ref)
        config = FidelityConfig.make("integrate", scene=choice)
        with pytest.raises(WiringError) as caught:
            if entrypoint == "validate":
                config.validate(registry).raise_if_failed()
            else:
                build(make_spec(), config, registry)
        assert str(caught.value) == (
            "configuration failed validation:\n"
            "- scene=required supports operators sample; the run requires 'integrate'"
        )
        assert calls == []

    def test_version_failure_kind(self, registry):
        """A nonmatching pin has a distinct report failure family."""
        from dataclasses import replace

        ref = replace(SeamRef.make("observation", "summary_rates"), version="2")
        config = FidelityConfig.make("sample", observation=ref)
        report = config.validate(registry)
        assert [failure.kind for failure in report.failures] == ["version"]
        assert "requires version '2'" in report.failures[0].message

    @pytest.mark.parametrize(
        "params",
        [dict(bogus=1), {}, dict(required=1, positional=2)],
    )
    def test_parameter_binding_failure_kind(self, params):
        """Unknown, missing, and positional-only parameters fail before a call."""
        calls = []

        class Required:
            def __init__(self, positional=0, /, *, required):
                calls.append(self)

        registry = Registry()
        registry.register("scene", "required", SeamInfo(operators={"sample"}))(Required)
        config = FidelityConfig.make(
            "sample", scene=SeamRef.make("scene", "required", **params)
        )
        report = config.validate(registry)
        assert [failure.kind for failure in report.failures] == ["params"]
        assert "scene=required" in report.failures[0].message
        assert calls == []

    @pytest.mark.parametrize("error_type", [TypeError, ValueError])
    def test_unavailable_signature_is_skipped(self, monkeypatch, registry, error_type):
        """An uninspectable class does not cause a new validation rejection."""
        import inspect

        def unavailable(cls, *, follow_wrapped):
            assert follow_wrapped is False
            raise error_type("signature unavailable")

        monkeypatch.setattr(inspect, "signature", unavailable)
        assert config_for().validate(registry).ok

    def test_ports_error_matches_prepared_preflight(self, registry):
        """Prepared validation retains the complete alternatives diagnostic."""
        from spaceodyssey.build import prepare
        from tests.doubles import make_spec

        config = config_for(post_processing="matched_filter")
        with pytest.raises(WiringError) as expected:
            config.validate(registry).raise_if_failed()
        with pytest.raises(WiringError) as actual:
            prepare(make_spec(), config, registry)
        assert str(actual.value) == str(expected.value)
