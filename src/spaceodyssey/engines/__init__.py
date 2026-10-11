"""Engines: one run callable per outcome operator.

The operator vocabulary is open, and this registry is where it lives: an
engine is a callable ``(mission, *, seed, **options) -> report``, keyed by
the operator name a configuration requests, where the report is one of the
schemas in :mod:`spaceodyssey.metrics`. There is no engine class
hierarchy because operators genuinely differ in control flow (a stochastic
event loop, a deterministic belief recursion, a vectorized closed-form
integral cannot stand in for one another behind a shared signature); the
shared surface is the report schema they all emit.

A genuinely new evaluation semantics enters as one new module registering
one new operator name; nothing else in the package enumerates operators.
"""

import importlib
from collections.abc import Callable

from spaceodyssey.aliases import resolve_qualname_alias
from spaceodyssey.build import construct, prepare
from spaceodyssey.errors import RegistryError
from spaceodyssey.metrics import CertifiedYield, YieldSummary
from spaceodyssey.registry import REGISTRY, Registry
from spaceodyssey.spec import FidelityConfig, MissionSpec

ENGINES: dict[str, Callable] = {}

_DEFAULT_ENGINE_MODULES: tuple[str, ...] = (
    "spaceodyssey.engines.mc",
    "spaceodyssey.engines.analytic",
)
_defaults_loaded = False


def register_engine(operator: str):
    """Return a decorator registering an engine for one operator name."""

    def decorate(fn):
        if operator in ENGINES:
            raise RegistryError(
                f"an engine for operator '{operator}' is already registered"
            )
        ENGINES[operator] = fn
        return fn

    return decorate


def _load_defaults() -> None:
    global _defaults_loaded
    if _defaults_loaded:
        return
    _defaults_loaded = True
    for module_path in _DEFAULT_ENGINE_MODULES:
        importlib.import_module(resolve_qualname_alias(module_path))


def get_engine(operator: str) -> Callable:
    """Return the engine for an operator.

    Raises:
        RegistryError: For an unknown operator; the message lists the
            registered ones.
    """
    _load_defaults()
    try:
        return ENGINES[operator]
    except KeyError:
        known = ", ".join(sorted(ENGINES)) or "(none)"
        raise RegistryError(
            f"no engine registered for operator '{operator}'; registered: {known}"
        ) from None


def run(
    spec: MissionSpec,
    config: FidelityConfig,
    *,
    seed: int = 0,
    registry: Registry = REGISTRY,
    files=None,
    **options,
) -> YieldSummary | CertifiedYield:
    """Validate, compile, and run one configuration end to end.

    Args:
        spec: The declarative problem statement.
        config: The implementation selection; its operator picks the engine.
        seed: The run's root seed.
        registry: The implementation registry to resolve against.
        files: Reserved file context, consumed without being read.
        **options: Forwarded to the engine.

    Returns:
        The run's report.
    """
    plan = prepare(spec, config, registry, files=files)
    engine = get_engine(config.operator)
    return engine(construct(plan), seed=seed, **options)
