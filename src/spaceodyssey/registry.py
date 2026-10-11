"""The per-seam implementation registry.

Every pipeline stage (seam) has an unordered set of named implementations,
each registered with metadata: which outcome operators it supports, which
contract types it consumes and produces, what it costs, and what it can
calibrate. Fidelity is metadata here, not structure; competing
implementations of the same seam are peers.

The registry stores classes, never instances. Instances are built by
:func:`spaceodyssey.build.build` from declarative references, which is what makes
runs reproducible from manifests alone.
"""

import importlib
import importlib.util
from dataclasses import dataclass

from spaceodyssey.aliases import resolve_qualname_alias
from spaceodyssey.contracts import compatible, resolve_contract
from spaceodyssey.errors import RegistryError, WiringError
from spaceodyssey.seams import SEAMS


@dataclass(frozen=True)
class SeamInfo:
    """Metadata registered alongside a seam implementation.

    Attributes:
        operators: Outcome operators the implementation supports ("sample",
            "propagate", "integrate", ...).
        consumes: Contract-type names accepted on the implementation's
            data-plane input edge; empty means it accepts anything.
        produces: Contract-type names of the implementation's data-plane
            output.
        version: Implementation version string, recorded in manifests.
        fidelity: Free-form fidelity label; metadata only, never ordered.
        cost_hint: Free-form relative cost label ("low", "medium", "high").
        needs_calibration: Names of calibration artifacts that must exist for
            the implementation to run calibrated; unmet needs flag the run.
        calibratable: Names of parameters a costlier implementation can refit.
        import_requirement: Distribution/module name the implementation needs
            at run time; when it is not installed the registry lists the
            implementation as present but unavailable.
    """

    operators: frozenset[str]
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    version: str = "0"
    fidelity: str = ""
    cost_hint: str = ""
    needs_calibration: tuple[str, ...] = ()
    calibratable: tuple[str, ...] = ()
    import_requirement: str | None = None

    def __post_init__(self):
        """Coerce collection fields so callers may pass any iterable."""
        object.__setattr__(self, "operators", frozenset(self.operators))
        for name in ("consumes", "produces", "needs_calibration", "calibratable"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class RegisteredImpl:
    """One registry entry: an implementation class plus its metadata.

    Attributes:
        seam: The seam the implementation belongs to.
        name: The implementation's registered name, unique within its seam.
        cls: The implementation class, or None when unavailable.
        info: The registered metadata.
        available: False when the implementation's import requirement is not
            installed.
        reason: Human-readable explanation when unavailable.
    """

    seam: str
    name: str
    cls: type | None
    info: SeamInfo
    available: bool = True
    reason: str = ""


def _check_seam(seam: str) -> None:
    if seam not in SEAMS:
        known = ", ".join(SEAMS)
        raise RegistryError(f"unknown seam '{seam}'; seams are: {known}")


class Registry:
    """Named seam implementations with lazy adapter-module loading."""

    def __init__(self):
        """Create an empty registry covering every seam."""
        self._impls: dict[str, dict[str, RegisteredImpl]] = {s: {} for s in SEAMS}
        self._pending: dict[str, list[str]] = {s: [] for s in SEAMS}

    def register(self, seam: str, name: str, info: SeamInfo):
        """Return a class decorator registering an implementation.

        Args:
            seam: Seam the implementation belongs to.
            name: Registry name, unique within the seam.
            info: Implementation metadata.

        Raises:
            RegistryError: On an unknown seam or a duplicate name.
        """
        _check_seam(seam)

        def decorate(cls):
            if name in self._impls[seam]:
                raise RegistryError(
                    f"seam '{seam}' already has an implementation named '{name}'"
                )
            self._impls[seam][name] = RegisteredImpl(seam, name, cls, info)
            return cls

        return decorate

    def add_module(self, seam: str, module_path: str) -> None:
        """Record a module whose import registers implementations of a seam.

        The module is imported on the first lookup touching the seam, so
        importing spaceodyssey never drags in heavyweight adapter targets.
        """
        _check_seam(seam)
        self._pending[seam].append(module_path)

    def _load(self, seam: str) -> None:
        while self._pending[seam]:
            module_path = self._pending[seam].pop(0)
            try:
                importlib.import_module(resolve_qualname_alias(module_path))
            except ImportError as err:
                raise RegistryError(
                    f"adapter module '{module_path}' for seam '{seam}' failed "
                    f"to import: {err}"
                ) from err

    def _availability(self, entry: RegisteredImpl) -> RegisteredImpl:
        requirement = entry.info.import_requirement
        if requirement is None:
            return entry
        try:
            present = (
                importlib.util.find_spec(resolve_qualname_alias(requirement))
                is not None
            )
        except (ImportError, ValueError):
            present = False
        if present:
            return entry
        return RegisteredImpl(
            seam=entry.seam,
            name=entry.name,
            cls=None,
            info=entry.info,
            available=False,
            reason=(f"requires the '{requirement}' package, which is not installed"),
        )

    def get(self, seam: str, name: str) -> RegisteredImpl:
        """Look up a registered name or class qualname after loading adapters.

        Qualified names may use a historical package alias. They resolve only
        to a unique class already registered for the requested seam; stored
        names and registry keys are never changed.

        Raises:
            RegistryError: On an unknown seam or name; the message lists the
                names registered for the seam.
        """
        _check_seam(seam)
        self._load(seam)
        try:
            entry = self._impls[seam][name]
        except KeyError:
            # Qualified references must still name a registered implementation
            # of this seam, so metadata and availability checks remain intact.
            if "." in name:
                try:
                    cls = resolve_contract(name)
                except WiringError as err:
                    raise RegistryError(str(err)) from err
                matches = [
                    candidate
                    for candidate in self._impls[seam].values()
                    if candidate.cls is cls
                ]
                if len(matches) == 1:
                    return self._availability(matches[0])
                if len(matches) > 1:
                    raise RegistryError(
                        f"qualified name '{name}' is ambiguous for seam '{seam}'; "
                        "use an implementation's registered name"
                    ) from None
            known = ", ".join(sorted(self._impls[seam])) or "(none)"
            raise RegistryError(
                f"seam '{seam}' has no implementation named '{name}'; "
                f"registered: {known}"
            ) from None
        return self._availability(entry)

    def names(self, seam: str) -> tuple[str, ...]:
        """Return the sorted names registered for a seam."""
        _check_seam(seam)
        self._load(seam)
        return tuple(sorted(self._impls[seam]))

    def matrix(self, seam_a: str, seam_b: str) -> dict[tuple[str, str], bool]:
        """Render the pairwise port-compatibility table for two seams.

        Entry (name_a, name_b) is True when the output of seam_a's
        implementation type-checks against the input of seam_b's. Pairs whose
        contract types cannot be resolved (for example, an optional dependency
        that is not installed) are reported as incompatible.
        """
        table = {}
        for name_a in self.names(seam_a):
            info_a = self.get(seam_a, name_a).info
            for name_b in self.names(seam_b):
                info_b = self.get(seam_b, name_b).info
                try:
                    ok = compatible(info_a.produces, info_b.consumes)
                except Exception:
                    ok = False
                table[(name_a, name_b)] = ok
        return table


REGISTRY = Registry()
"""The default registry, used by :func:`spaceodyssey.build.build` unless overridden."""


def register(seam: str, name: str, info: SeamInfo):
    """Register an implementation on the default registry (class decorator)."""
    return REGISTRY.register(seam, name, info)


_DEFAULT_ADAPTER_MODULES: dict[str, tuple[str, ...]] = {
    "certification": ("spaceodyssey.certify",),
    "context": ("spaceodyssey.context",),
    "cost": ("spaceodyssey.account",),
}

for _seam, _modules in _DEFAULT_ADAPTER_MODULES.items():
    for _module in _modules:
        REGISTRY.add_module(_seam, _module)
