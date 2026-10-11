"""Compile the declarative layer into runnable stacks.

``prepare`` resolves and checks a specification and configuration once,
recording classes and parameters in a frozen plan before constructors run.
``construct`` turns that plan into live objects whose continuous parameters
are pytree leaves, without further resolution. ``build`` performs both steps.
Nothing downstream of it reads configuration.

Truth and model sides compile into two distinct types, and that distinction
is the firewall: only the observation function is typed to accept a
:class:`TruthStack`, so the drawn truth can never leak into policies, belief
updates, or certification by construction rather than by discipline.
"""

from dataclasses import dataclass
from typing import Any, final

import equinox as eqx

from spaceodyssey.errors import RegistryError, WiringError
from spaceodyssey.registry import REGISTRY, RegisteredImpl, Registry
from spaceodyssey.spec import FidelityConfig, MissionSpec, SeamRef
from spaceodyssey.wiring import _seam_mismatch, _version_mismatch, validate_config


class _SeamSlots(eqx.Module):
    """One compiled implementation per seam; unchosen seams stay None."""

    scene: Any | None = None
    instrument: Any | None = None
    observation: Any | None = None
    post_processing: Any | None = None
    characterization: Any | None = None
    certification: Any | None = None
    policy: Any | None = None
    population: Any | None = None
    context: Any | None = None
    cost: Any | None = None


@final
class TruthStack(_SeamSlots):
    """The data-generating side; only the observation path ever holds it."""


@final
class ModelStack(_SeamSlots):
    """The agent-believed side; what policies, updates, and certificates see."""


@final
class Mission(eqx.Module):
    """A compiled, runnable mission: spec + config + both stacks.

    Attributes:
        spec: The declarative problem statement.
        config: The declarative implementation selection.
        truth: The compiled truth-side stack.
        model: The compiled model-side stack.
    """

    spec: MissionSpec = eqx.field(static=True)
    config: FidelityConfig = eqx.field(static=True)
    truth: TruthStack
    model: ModelStack


@dataclass(frozen=True)
class ResolvedImpl:
    """The resolved identity of one configuration slot and side.

    Attributes:
        slot: Configuration slot holding the implementation.
        side: The truth or model side.
        name: Implementation name as written in the reference.
        registered_name: Name of the entry returned by the registry.
        qualname: Module-qualified name of the recorded class.
        version: Version declared by the recorded entry's metadata.
    """

    slot: str
    side: str
    name: str
    registered_name: str
    qualname: str
    version: str


@dataclass(frozen=True)
class _PreparedImpl:
    """One constructor call retained by a prepared plan."""

    slot: str
    side: str
    ref: SeamRef
    entry: RegisteredImpl


@dataclass(frozen=True)
class BuildPlan:
    """A frozen construction plan with resolved implementation identities.

    Attributes:
        spec: The declarative problem statement.
        config: The declarative implementation selection.
        resolution: Resolved identities in configuration slot and side order.
        files: Reserved file context, carried without being read.
    """

    spec: MissionSpec
    config: FidelityConfig
    resolution: tuple[ResolvedImpl, ...]
    _implementations: tuple[_PreparedImpl, ...]
    files: Any = None


def _instantiate(impl: _PreparedImpl):
    ref = impl.ref
    try:
        return impl.entry.cls(**ref.params_dict)
    except TypeError as err:
        raise WiringError(
            f"could not construct {ref.seam}={ref.name} with parameters "
            f"{ref.params_dict}: {err}"
        ) from err
    except Exception as err:
        err.add_note(
            f"constructing slot '{impl.slot}', side '{impl.side}', "
            f"implementation {ref.seam}={ref.name}"
        )
        raise


def prepare(
    spec: MissionSpec,
    config: FidelityConfig,
    registry: Registry = REGISTRY,
    *,
    check: bool = True,
    files=None,
) -> BuildPlan:
    """Resolve a configuration once before any implementation is constructed.

    Args:
        spec: The declarative problem statement.
        config: The implementation selection to prepare.
        registry: The registry to resolve references through.
        check: Check parameters, operators, ports, and calibration metadata.
            Duplicate slots, availability, reference seams, and version pins
            are always checked.
        files: Reserved file context, carried without being read.

    Returns:
        A frozen plan retaining the resolved classes and parameters.

    Raises:
        WiringError: On validation failure or an unavailable reference.
        RegistryError: On an unknown reference when check is False.
    """
    slots = set()
    for seam, _ in config.choices:
        if seam in slots:
            raise WiringError(f"duplicate slot '{seam}' in configuration")
        slots.add(seam)
    entries = {}
    implementations = []
    for seam, choice in config.choices:
        for side, ref in (("truth", choice.truth), ("model", choice.model)):
            if ref is None:
                entries[seam, side] = entries[seam, "truth"]
                continue
            mismatch = _seam_mismatch(seam, ref)
            if mismatch:
                if not check:
                    raise WiringError(mismatch)
                entries[seam, side] = None
                continue
            try:
                entry = registry.get(seam, ref.name)
            except RegistryError as err:
                if not check:
                    raise
                entries[seam, side] = err
                continue
            entries[seam, side] = entry
            if not check:
                if not entry.available:
                    raise WiringError(
                        f"cannot build {ref.seam}={ref.name}: {entry.reason}"
                    )
                mismatch = _version_mismatch(ref, entry)
                if mismatch:
                    raise WiringError(mismatch)
            implementations.append(_PreparedImpl(seam, side, ref, entry))
    if check:
        validate_config(config, registry, entries=entries).raise_if_failed()
    resolution = tuple(
        ResolvedImpl(
            slot=seam,
            side=side,
            name=ref.name,
            registered_name=entries[seam, side].name,
            qualname=(
                f"{entries[seam, side].cls.__module__}."
                f"{entries[seam, side].cls.__qualname__}"
            ),
            version=entries[seam, side].info.version,
        )
        for seam, choice in config.choices
        for side, ref in (("truth", choice.truth), ("model", choice.model_ref))
    )
    return BuildPlan(spec, config, resolution, tuple(implementations), files)


def construct(plan: BuildPlan) -> Mission:
    """Construct a mission from recorded classes without further resolution.

    Args:
        plan: A prepared construction plan.

    Returns:
        The compiled mission. Slots without a model override share one instance.

    Raises:
        WiringError: If a constructor rejects its parameters with TypeError.
    """
    truth_impls: dict[str, Any] = {}
    model_impls: dict[str, Any] = {}
    for impl in plan._implementations:
        instance = _instantiate(impl)
        if impl.side == "truth":
            truth_impls[impl.slot] = instance
            model_impls[impl.slot] = instance
        else:
            model_impls[impl.slot] = instance
    return Mission(
        spec=plan.spec,
        config=plan.config,
        truth=TruthStack(**truth_impls),
        model=ModelStack(**model_impls),
    )


def build(
    spec: MissionSpec,
    config: FidelityConfig,
    registry: Registry = REGISTRY,
    *,
    check: bool = True,
    files=None,
) -> Mission:
    """Compile a spec and config into a runnable Mission.

    Args:
        spec: The declarative problem statement.
        config: The implementation selection to compile.
        registry: The registry to resolve references through.
        check: Validate parameters, operators, ports, and calibration metadata.
            Duplicate slots, availability, reference seams, and version pins
            are always checked.
        files: Reserved file context, carried without being read.

    Returns:
        The compiled mission with separate truth and model stacks. Where a
        seam has no model override, both stacks share one instance.

    Raises:
        WiringError: On validation failure or an unconstructable reference.
    """
    return construct(prepare(spec, config, registry, check=check, files=files))
