"""Tests for compiling declarative configurations into stacks."""

import pytest

from spaceodyssey.build import Mission, ModelStack, TruthStack, build
from spaceodyssey.errors import WiringError
from spaceodyssey.registry import SeamInfo
from spaceodyssey.spec import FidelityConfig, SeamChoice, SeamRef
from tests.doubles import ThreeTargetSource, build_test_registry, make_config, make_spec


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

    @pytest.mark.parametrize("check", [True, False])
    @pytest.mark.parametrize("stack", ["truth", "model"])
    def test_ref_seam_mismatch_refused(self, check, stack):
        """Construction rejects mismatched ref seams even without validation."""
        registry = build_test_registry()
        registry.register("certification", "presence", SeamInfo(operators={"sample"}))(
            ThreeTargetSource
        )
        ref = SeamRef.make("certification", "presence")
        choice = SeamChoice(
            truth=ref if stack == "truth" else SeamRef.make("observation", ref.name),
            model=ref if stack == "model" else None,
        )
        config = FidelityConfig.make("sample", observation=choice)
        with pytest.raises(WiringError, match=r"observation.*certification"):
            build(make_spec(), config, registry, check=check)


class TestPreparedBuild:
    """Prepared plans retain the exact entries used by their preflight."""

    def test_decorated_constructor_accepts_wrapper_parameters(self):
        """Preflight binds the callable wrapper rather than its wrapped method."""
        from functools import wraps

        from spaceodyssey.registry import Registry

        def accepts_extra_parameters(init):
            @wraps(init)
            def wrapper(self, **kwargs):
                self.extra = kwargs.pop("extra")
                init(self, **kwargs)

            return wrapper

        class Decorated:
            @accepts_extra_parameters
            def __init__(self, value):
                self.value = value

        registry = Registry()
        registry.register("scene", "decorated", SeamInfo(operators={"sample"}))(
            Decorated
        )
        config = FidelityConfig.make(
            "sample", scene=SeamRef.make("scene", "decorated", value=7, extra=9)
        )
        mission = build(make_spec(), config, registry, check=True)
        assert mission.truth.scene.value == 7
        assert mission.truth.scene.extra == 9
        assert mission.model.scene is mission.truth.scene

    @pytest.mark.parametrize("check", [True, False])
    def test_duplicate_slots_rejected_before_resolution(self, check):
        """A later valid choice cannot hide an earlier bad version pin."""
        from dataclasses import replace

        from spaceodyssey.registry import Registry

        constructed = []
        resolved = []

        class Tracked:
            def __init__(self):
                constructed.append(self)

        class TrackedRegistry(Registry):
            def get(self, seam, name):
                resolved.append((seam, name))
                return super().get(seam, name)

        registry = TrackedRegistry()
        registry.register("scene", "tracked", SeamInfo(operators={"sample"}))(Tracked)
        ref = SeamRef.make("scene", "tracked")
        config = FidelityConfig(
            "sample",
            choices=(
                ("scene", SeamChoice(replace(ref, version="BAD"))),
                ("scene", SeamChoice(ref)),
            ),
        )
        with pytest.raises(WiringError, match="duplicate slot 'scene'"):
            build(make_spec(), config, registry, check=check)
        assert constructed == []
        assert resolved == []

    def test_build_resolves_once(self):
        """A changing registry cannot replace the preflighted class."""
        from tests.doubles import ChangingRegistry, FirstImplementation

        registry = ChangingRegistry()
        config = FidelityConfig.make("sample", scene=SeamRef.make("scene", "changing"))
        mission = build(make_spec(), config, registry)
        assert type(mission.truth.scene) is FirstImplementation
        assert mission.model.scene is mission.truth.scene
        assert registry.calls == [("scene", "changing")]

    def test_plan_records_each_side_and_preserves_parameters(self):
        """Construction uses a frozen plan even if registration later changes."""
        from dataclasses import FrozenInstanceError

        from spaceodyssey.build import BuildPlan, ResolvedImpl, construct, prepare
        from tests.doubles import (
            ChangingRegistry,
            FirstImplementation,
            SecondImplementation,
        )

        registry = ChangingRegistry()
        ref = SeamRef.make("scene", "changing", value=7)
        config = FidelityConfig.make("sample", scene=SeamChoice(ref, ref))
        files = object()
        plan = prepare(make_spec(), config, registry, files=files)
        assert isinstance(plan, BuildPlan)
        assert plan.files is files
        assert plan.resolution == tuple(
            ResolvedImpl(
                slot="scene",
                side=side,
                name="changing",
                registered_name="changing",
                qualname=f"{cls.__module__}.{cls.__qualname__}",
                version="0",
            )
            for side, cls in (
                ("truth", FirstImplementation),
                ("model", SecondImplementation),
            )
        )
        with pytest.raises(FrozenInstanceError):
            plan.files = None
        registry.get("scene", "changing")
        calls_before = list(registry.calls)
        mission = construct(plan)
        assert type(mission.truth.scene) is FirstImplementation
        assert type(mission.model.scene) is SecondImplementation
        assert mission.truth.scene.value == mission.model.scene.value == 7
        assert registry.calls == calls_before

    def test_resolution_retains_written_qualified_name(self):
        """The record distinguishes a qualified reference from its registry key."""
        from spaceodyssey.build import prepare
        from tests.doubles import PresenceMeasurement

        name = f"{PresenceMeasurement.__module__}.{PresenceMeasurement.__qualname__}"
        config = FidelityConfig.make(
            "sample", observation=SeamRef.make("observation", name)
        )
        plan = prepare(make_spec(), config, build_test_registry())
        assert tuple(item.side for item in plan.resolution) == ("truth", "model")
        for item in plan.resolution:
            assert item.name == item.qualname == name
            assert item.registered_name == "presence"

    @pytest.mark.parametrize("check", [True, False])
    @pytest.mark.parametrize("side", ["truth", "model"])
    def test_mismatched_pin_refused(self, check, side):
        """Every preparation enforces pins on both sides."""
        from dataclasses import replace

        ref = SeamRef.make("observation", "presence")
        pinned = replace(ref, version="2")
        choice = SeamChoice(pinned if side == "truth" else ref, pinned)
        config = FidelityConfig.make("sample", observation=choice)
        with pytest.raises(WiringError, match="requires version '2'"):
            build(make_spec(), config, build_test_registry(), check=check)

    def test_matching_pin_builds(self):
        """A matching registered version allows construction."""
        from dataclasses import replace

        ref = replace(SeamRef.make("observation", "presence"), version="0")
        config = FidelityConfig.make("sample", observation=ref)
        assert build(make_spec(), config, build_test_registry()).truth.observation

    def test_registration_change_after_validation_is_caught(self):
        """An earlier successful validation cannot bypass current version pins."""
        from dataclasses import replace

        from spaceodyssey.registry import Registry
        from tests.doubles import FirstImplementation

        class VersionRegistry(Registry):
            """Expose metadata updates through the public lookup boundary."""

            version = "1"

            def get(self, seam, name):
                entry = super().get(seam, name)
                return replace(entry, info=replace(entry.info, version=self.version))

        registry = VersionRegistry()
        registry.register("scene", "test", SeamInfo(operators={"sample"}))(
            FirstImplementation
        )
        ref = replace(SeamRef.make("scene", "test"), version="1")
        config = FidelityConfig.make("sample", scene=ref)
        assert config.validate(registry).ok
        registry.version = "2"
        with pytest.raises(WiringError, match="registered version is '2'"):
            build(make_spec(), config, registry)

    @pytest.mark.parametrize("check", [True, False])
    def test_last_reference_failure_precedes_all_constructors(self, check):
        """All references are resolved before construction begins."""
        from spaceodyssey.errors import RegistryError
        from spaceodyssey.registry import Registry

        constructed = []

        class Tracked:
            def __init__(self):
                constructed.append(self)

        registry = Registry()
        registry.register("scene", "tracked", SeamInfo(operators={"sample"}))(Tracked)
        config = FidelityConfig(
            "sample",
            choices=(
                ("scene", SeamChoice(SeamRef.make("scene", "tracked"))),
                ("cost", SeamChoice(SeamRef.make("cost", "missing"))),
            ),
        )
        with pytest.raises(WiringError if check else RegistryError):
            build(make_spec(), config, registry, check=check)
        assert constructed == []

    @pytest.mark.parametrize("side", ["truth", "model"])
    def test_science_constructor_error_keeps_type_and_adds_context(self, side):
        """Science failures keep their identity and name the failing side."""
        from spaceodyssey.errors import UnsupportedProfile
        from spaceodyssey.registry import Registry
        from tests.doubles import FirstImplementation

        error = UnsupportedProfile("unsupported science profile")

        class Rejecting:
            def __init__(self):
                raise error

        registry = Registry()
        registry.register("scene", "ok", SeamInfo(operators={"sample"}))(
            FirstImplementation
        )
        registry.register("scene", "rejecting", SeamInfo(operators={"sample"}))(
            Rejecting
        )
        ref = SeamRef.make("scene", "rejecting")
        choice = (
            SeamChoice(ref)
            if side == "truth"
            else SeamChoice(SeamRef.make("scene", "ok"), ref)
        )
        config = FidelityConfig.make("sample", scene=choice)
        with pytest.raises(UnsupportedProfile) as caught:
            build(make_spec(), config, registry)
        assert caught.value is error
        note = error.__notes__[-1]
        assert "slot 'scene'" in note
        assert f"side '{side}'" in note
        assert "scene=rejecting" in note

    def test_constructor_typeerror_still_becomes_wiring_error(self):
        """A TypeError from a correctly bound constructor retains conversion."""
        from spaceodyssey.registry import Registry

        class Rejecting:
            def __init__(self):
                raise TypeError("invalid science parameter")

        registry = Registry()
        registry.register("scene", "rejecting", SeamInfo(operators={"sample"}))(
            Rejecting
        )
        config = FidelityConfig.make("sample", scene=SeamRef.make("scene", "rejecting"))
        with pytest.raises(WiringError, match="could not construct scene=rejecting"):
            build(make_spec(), config, registry)
