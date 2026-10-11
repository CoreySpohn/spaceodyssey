"""Configuration validation: typed ports, operator support, calibration flags.

A configuration is checked in three independent families before anything
runs: every chosen implementation must exist and be available, must support
the run's outcome operator, and every data-plane edge between chosen seams
must be type-compatible (some produced contract type a subclass of some
consumed one). Failures name both sides and the registered alternatives that
would be compatible, so the report doubles as compatibility documentation.

Truth and model sides are validated independently whenever a configuration
deliberately mismatches them; a clean configuration with no model overrides
is checked once.
"""

from dataclasses import dataclass

from spaceodyssey.contracts import compatible
from spaceodyssey.errors import RegistryError, WiringError
from spaceodyssey.seams import DATA_PLANE_EDGES


@dataclass(frozen=True)
class WiringFailure:
    """One reason a configuration cannot run.

    Attributes:
        kind: Failure family: "unknown_impl", "unavailable", "operator",
            "ports", or "contract".
        stack: Which side failed, "truth" or "model".
        edge: The (producer, consumer) seam pair for edge failures, else None.
        seam: The seam the failure is reported against.
        impl: The implementation name the failure is reported against.
        message: Self-contained explanation, naming compatible alternatives
            where any exist.
        alternatives: Registered implementation names that would be
            compatible on the failing edge.
    """

    kind: str
    stack: str
    edge: tuple[str, str] | None
    seam: str
    impl: str
    message: str
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class WiringReport:
    """The outcome of validating one configuration against a registry.

    Attributes:
        ok: True when there are no failures.
        operator: The outcome operator the configuration requested.
        failures: Every failure found, across both stacks.
        skipped_edges: Data-plane edges not checked because an endpoint seam
            was not chosen in the configuration.
        uncalibrated: "seam:name" entries whose declared calibration needs are
            not met; the run may proceed but is flagged uncalibrated.
    """

    ok: bool
    operator: str
    failures: tuple[WiringFailure, ...] = ()
    skipped_edges: tuple[tuple[str, str], ...] = ()
    uncalibrated: tuple[str, ...] = ()

    def raise_if_failed(self) -> None:
        """Raise WiringError listing every failure message, if any."""
        if self.ok:
            return
        lines = "\n".join(f"- {failure.message}" for failure in self.failures)
        raise WiringError(f"configuration failed validation:\n{lines}")


def _stack_refs(config, stack: str) -> dict:
    return {
        seam: (choice.truth if stack == "truth" else choice.model_ref)
        for seam, choice in config.choices
    }


def _compatible_alternatives(registry, seam: str, against, direction: str):
    """Names registered for a seam whose ports fit the given counterpart."""
    try:
        candidates = registry.names(seam)
    except RegistryError:
        return ()
    names = []
    for candidate in candidates:
        try:
            info = registry.get(seam, candidate).info
            if direction == "producer":
                ok = compatible(info.produces, against.consumes)
            else:
                ok = compatible(against.produces, info.consumes)
        except (WiringError, RegistryError):
            ok = False
        if ok:
            names.append(candidate)
    return tuple(names)


def _check_stack(config, registry, stack: str, failures, uncalibrated) -> None:
    refs = _stack_refs(config, stack)
    infos = {}
    for seam, ref in sorted(refs.items()):
        try:
            entry = registry.get(seam, ref.name)
        except RegistryError as err:
            failures.append(
                WiringFailure(
                    kind="unknown_impl",
                    stack=stack,
                    edge=None,
                    seam=seam,
                    impl=ref.name,
                    message=str(err),
                )
            )
            continue
        if not entry.available:
            failures.append(
                WiringFailure(
                    kind="unavailable",
                    stack=stack,
                    edge=None,
                    seam=seam,
                    impl=ref.name,
                    message=(
                        f"{seam}={ref.name} is registered but unavailable: "
                        f"{entry.reason}"
                    ),
                )
            )
            continue
        if config.operator not in entry.info.operators:
            supported = ", ".join(sorted(entry.info.operators)) or "(none)"
            failures.append(
                WiringFailure(
                    kind="operator",
                    stack=stack,
                    edge=None,
                    seam=seam,
                    impl=ref.name,
                    message=(
                        f"{seam}={ref.name} supports operators {supported}; "
                        f"the run requires '{config.operator}'"
                    ),
                )
            )
        infos[seam] = entry.info
        if entry.info.needs_calibration:
            uncalibrated.add(f"{seam}:{ref.name}")

    for edge in DATA_PLANE_EDGES:
        producer_seam, consumer_seam = edge
        if producer_seam not in infos or consumer_seam not in infos:
            continue
        producer_info = infos[producer_seam]
        consumer_info = infos[consumer_seam]
        producer_name = refs[producer_seam].name
        consumer_name = refs[consumer_seam].name
        try:
            ok = compatible(producer_info.produces, consumer_info.consumes)
        except WiringError as err:
            failures.append(
                WiringFailure(
                    kind="contract",
                    stack=stack,
                    edge=edge,
                    seam=consumer_seam,
                    impl=consumer_name,
                    message=f"edge {producer_seam}->{consumer_seam}: {err}",
                )
            )
            continue
        if ok:
            continue
        alt_consumers = _compatible_alternatives(
            registry, consumer_seam, producer_info, "consumer"
        )
        alt_producers = _compatible_alternatives(
            registry, producer_seam, consumer_info, "producer"
        )
        consumed = ", ".join(consumer_info.consumes) or "(nothing)"
        produced = ", ".join(producer_info.produces) or "(nothing)"
        hints = []
        if alt_consumers:
            hints.append(f"{consumer_seam}={{{', '.join(alt_consumers)}}}")
        if alt_producers:
            hints.append(f"{producer_seam}={{{', '.join(alt_producers)}}}")
        if hints:
            hint = "; compatible: " + " or ".join(hints)
        else:
            hint = "; no compatible registered alternatives"
        failures.append(
            WiringFailure(
                kind="ports",
                stack=stack,
                edge=edge,
                seam=consumer_seam,
                impl=consumer_name,
                message=(
                    f"{consumer_seam}={consumer_name} consumes {consumed}; "
                    f"{producer_seam}={producer_name} produces {produced}"
                    f"{hint}"
                ),
                alternatives=alt_consumers + alt_producers,
            )
        )


def validate_config(config, registry) -> WiringReport:
    """Validate a FidelityConfig against a registry.

    Args:
        config: The configuration to check (operator + seam choices).
        registry: The implementation registry to resolve against.

    Returns:
        A WiringReport; call raise_if_failed() to enforce it.
    """
    failures: list[WiringFailure] = []
    uncalibrated: set[str] = set()
    chosen = {seam for seam, _ in config.choices}
    skipped = tuple(
        edge
        for edge in DATA_PLANE_EDGES
        if edge[0] not in chosen or edge[1] not in chosen
    )
    has_override = any(choice.model is not None for _, choice in config.choices)
    stacks = ("truth", "model") if has_override else ("truth",)
    for stack in stacks:
        _check_stack(config, registry, stack, failures, uncalibrated)
    return WiringReport(
        ok=not failures,
        operator=config.operator,
        failures=tuple(failures),
        skipped_edges=skipped,
        uncalibrated=tuple(sorted(uncalibrated)),
    )
