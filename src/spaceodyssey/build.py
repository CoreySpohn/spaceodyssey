"""Compile the declarative layer into runnable stacks.

``build`` resolves a specification and configuration through the registry
into live objects whose continuous parameters are pytree leaves. Nothing
downstream of it ever reads configuration; nothing upstream of it touches
JAX.

Truth and model sides compile into two distinct types, and that distinction
is the firewall: only the observation function is typed to accept a
:class:`TruthStack`, so the drawn truth can never leak into policies, belief
updates, or certification by construction rather than by discipline.
"""

from typing import Any, final

import equinox as eqx

from spaceodyssey.errors import WiringError
from spaceodyssey.registry import REGISTRY, Registry
from spaceodyssey.spec import FidelityConfig, MissionSpec, SeamRef


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


def _instantiate(registry: Registry, ref: SeamRef):
    entry = registry.get(ref.seam, ref.name)
    if not entry.available:
        raise WiringError(f"cannot build {ref.seam}={ref.name}: {entry.reason}")
    try:
        return entry.cls(**ref.params_dict)
    except TypeError as err:
        raise WiringError(
            f"could not construct {ref.seam}={ref.name} with parameters "
            f"{ref.params_dict}: {err}"
        ) from err


def build(
    spec: MissionSpec,
    config: FidelityConfig,
    registry: Registry = REGISTRY,
    *,
    check: bool = True,
) -> Mission:
    """Compile a spec and config into a runnable Mission.

    Args:
        spec: The declarative problem statement.
        config: The implementation selection to compile.
        registry: The registry to resolve references through.
        check: Validate the configuration first and raise on failure; pass
            False only when the caller has already validated.

    Returns:
        The compiled mission with separate truth and model stacks. Where a
        seam has no model override, both stacks share one instance.

    Raises:
        WiringError: On validation failure or an unconstructable reference.
    """
    if check:
        config.validate(registry).raise_if_failed()
    truth_impls: dict[str, Any] = {}
    model_impls: dict[str, Any] = {}
    for seam, choice in config.choices:
        for ref in (choice.truth, choice.model_ref):
            if ref.seam != seam:
                raise WiringError(
                    f"slot '{seam}' references {ref.seam}={ref.name}; "
                    "reference seam must match the slot"
                )
        truth_impls[seam] = _instantiate(registry, choice.truth)
        if choice.model is None:
            model_impls[seam] = truth_impls[seam]
        else:
            model_impls[seam] = _instantiate(registry, choice.model)
    return Mission(
        spec=spec,
        config=config,
        truth=TruthStack(**truth_impls),
        model=ModelStack(**model_impls),
    )
