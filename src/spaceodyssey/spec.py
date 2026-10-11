"""The declarative layer: problem statement and implementation selection.

Everything in this module is plain frozen dataclasses of primitives: fully
JSON-serializable, content-hashable, holding no arrays and no live objects.
This layer is the core of a run manifest; identity, diffing, and provenance
live here. The compiled layer (:mod:`spaceodyssey.build`) resolves these references
through the registry into live objects. Nothing downstream of build() reads
configuration, and nothing here touches JAX.
"""

import hashlib
import json
from dataclasses import dataclass

from spaceodyssey.wiring import WiringReport, validate_config

_PRIMITIVES = (str, int, float, bool, type(None))


def _canonical_value(value, what: str):
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_value(item, what) for item in value)
    raise TypeError(
        f"{what} must be a primitive (str/int/float/bool/None) or a sequence "
        f"of primitives, got {type(value).__name__}; reference large values "
        "by file path"
    )


def _canonical_pairs(pairs, what: str) -> tuple:
    canonical = []
    for key, value in pairs:
        if not isinstance(key, str):
            raise TypeError(f"{what} keys must be strings, got {key!r}")
        canonical.append((key, _canonical_value(value, f"{what}['{key}']")))
    return tuple(canonical)


@dataclass(frozen=True)
class FileRef:
    """A reference to a file by path, optionally pinned by content hash.

    Attributes:
        path: Filesystem path (or workspace-relative path) of the file.
        sha256: Optional content hash pinning the file's identity.
    """

    path: str
    sha256: str | None = None


@dataclass(frozen=True)
class SeamRef:
    """A declarative reference to one registered seam implementation.

    Attributes:
        seam: The seam name.
        name: The implementation's registered name or qualified class name.
            Historical qualified names are resolved only when building; the
            original spelling remains part of serialized provenance.
        version: Optional pin that must equal the registered version.
        params: Constructor parameters as sorted (key, value) pairs of
            primitives or nested tuples of primitives.
    """

    seam: str
    name: str
    version: str = ""
    params: tuple[tuple[str, object], ...] = ()

    @classmethod
    def make(cls, seam: str, name: str, /, **params) -> "SeamRef":
        """Build a SeamRef with keyword parameters, sorted and canonicalized."""
        pairs = _canonical_pairs(sorted(params.items()), "params")
        return cls(seam=seam, name=name, params=pairs)

    @property
    def params_dict(self) -> dict:
        """The constructor parameters as a dictionary."""
        return dict(self.params)


@dataclass(frozen=True)
class SeamChoice:
    """The (truth, model) implementation pair chosen for one seam.

    The truth side generates the data; the model side is what the agent
    believes and fits with. They default to the same implementation, and
    deliberately mismatching them is how model-misspecification studies are
    expressed.

    Attributes:
        truth: Implementation reference used to generate data.
        model: Implementation reference the agent assumes; None means "same
            as truth".
    """

    truth: SeamRef
    model: SeamRef | None = None

    @property
    def model_ref(self) -> SeamRef:
        """The effective model-side reference."""
        return self.truth if self.model is None else self.model


@dataclass(frozen=True)
class FidelityConfig:
    """One run's implementation selection and outcome operator.

    Attributes:
        operator: The outcome operator the run evaluates ("sample",
            "propagate", "integrate", ...).
        choices: (seam, SeamChoice) pairs, sorted by seam name.
    """

    operator: str
    choices: tuple[tuple[str, SeamChoice], ...] = ()

    @classmethod
    def make(cls, operator: str, /, **choices) -> "FidelityConfig":
        """Build a config from keyword seam choices.

        Values may be SeamRef (meaning truth == model) or SeamChoice.
        """
        normalized = []
        for seam, choice in sorted(choices.items()):
            if isinstance(choice, SeamRef):
                choice = SeamChoice(truth=choice)
            if not isinstance(choice, SeamChoice):
                raise TypeError(
                    f"choice for seam '{seam}' must be SeamRef or SeamChoice, "
                    f"got {type(choice).__name__}"
                )
            normalized.append((seam, choice))
        return cls(operator=operator, choices=tuple(normalized))

    def choice(self, seam: str) -> SeamChoice | None:
        """Return the choice for a seam, or None when the seam is unchosen."""
        for name, choice in self.choices:
            if name == seam:
                return choice
        return None

    def validate(self, registry) -> WiringReport:
        """Check this configuration against a registry.

        Verifies that every chosen implementation exists and is available,
        that its version pin and constructor parameters match, that it
        supports the run's operator, and that every data-plane edge
        between chosen seams is type-compatible; unmet calibration needs are
        flagged. See :mod:`spaceodyssey.wiring`.
        """
        return validate_config(self, registry)


@dataclass(frozen=True)
class MissionSpec:
    """The tier-agnostic problem statement.

    The same spec feeds every engine, so a gradient computed on one engine
    lives in the same parameter space another engine validates. Simplifying
    assumptions belong in the run manifest, never baked in here.

    Attributes:
        name: Human-readable spec name.
        duration_d: Mission duration in days.
        budgets: Named resource caps as (resource, amount) pairs; "time_d" is
            the universal resource.
        bands: Spectral band names available to actions.
        modes: Observing mode names available to actions.
        catalog: Reference to the target catalog file.
        target_priors: Per-target prior overrides as (target_id, FileRef)
            pairs, for precursor knowledge.
        extra: Free-form (key, primitive) pairs for forward-compatible growth.
    """

    name: str
    duration_d: float
    budgets: tuple[tuple[str, float], ...] = ()
    bands: tuple[str, ...] = ()
    modes: tuple[str, ...] = ("imaging",)
    catalog: FileRef | None = None
    target_priors: tuple[tuple[int, FileRef], ...] = ()
    extra: tuple[tuple[str, object], ...] = ()

    @classmethod
    def make(
        cls,
        name: str,
        duration_d: float,
        /,
        *,
        budgets: dict | None = None,
        bands: tuple[str, ...] = (),
        modes: tuple[str, ...] = ("imaging",),
        catalog: FileRef | None = None,
        target_priors: dict | None = None,
        **extra,
    ) -> "MissionSpec":
        """Build a MissionSpec from dictionary-friendly arguments."""
        budget_pairs = tuple(sorted((budgets or {}).items()))
        prior_pairs = tuple(sorted((target_priors or {}).items()))
        extra_pairs = _canonical_pairs(sorted(extra.items()), "extra")
        return cls(
            name=name,
            duration_d=duration_d,
            budgets=budget_pairs,
            bands=tuple(bands),
            modes=tuple(modes),
            catalog=catalog,
            target_priors=prior_pairs,
            extra=extra_pairs,
        )

    @property
    def budgets_dict(self) -> dict:
        """The resource budgets as a dictionary."""
        return dict(self.budgets)


def _encode(obj):
    if isinstance(obj, FileRef):
        return {"__type__": "FileRef", "path": obj.path, "sha256": obj.sha256}
    if isinstance(obj, SeamRef):
        return {
            "__type__": "SeamRef",
            "seam": obj.seam,
            "name": obj.name,
            "version": obj.version,
            "params": [[k, v] for k, v in obj.params],
        }
    if isinstance(obj, SeamChoice):
        return {
            "__type__": "SeamChoice",
            "truth": _encode(obj.truth),
            "model": None if obj.model is None else _encode(obj.model),
        }
    if isinstance(obj, FidelityConfig):
        return {
            "__type__": "FidelityConfig",
            "operator": obj.operator,
            "choices": [[seam, _encode(choice)] for seam, choice in obj.choices],
        }
    if isinstance(obj, MissionSpec):
        return {
            "__type__": "MissionSpec",
            "name": obj.name,
            "duration_d": obj.duration_d,
            "budgets": [[k, v] for k, v in obj.budgets],
            "bands": list(obj.bands),
            "modes": list(obj.modes),
            "catalog": None if obj.catalog is None else _encode(obj.catalog),
            "target_priors": [[tid, _encode(ref)] for tid, ref in obj.target_priors],
            "extra": [[k, v] for k, v in obj.extra],
        }
    if isinstance(obj, _PRIMITIVES):
        return obj
    raise TypeError(f"cannot encode {type(obj).__name__} declaratively")


def _decode(data):
    if not isinstance(data, dict) or "__type__" not in data:
        return data
    kind = data["__type__"]
    if kind == "FileRef":
        return FileRef(path=data["path"], sha256=data["sha256"])
    if kind == "SeamRef":
        return SeamRef(
            seam=data["seam"],
            name=data["name"],
            version=data["version"],
            params=_canonical_pairs(data["params"], "params"),
        )
    if kind == "SeamChoice":
        model = data["model"]
        return SeamChoice(
            truth=_decode(data["truth"]),
            model=None if model is None else _decode(model),
        )
    if kind == "FidelityConfig":
        return FidelityConfig(
            operator=data["operator"],
            choices=tuple((seam, _decode(choice)) for seam, choice in data["choices"]),
        )
    if kind == "MissionSpec":
        catalog = data["catalog"]
        return MissionSpec(
            name=data["name"],
            duration_d=data["duration_d"],
            budgets=tuple((k, v) for k, v in data["budgets"]),
            bands=tuple(data["bands"]),
            modes=tuple(data["modes"]),
            catalog=None if catalog is None else _decode(catalog),
            target_priors=tuple(
                (tid, _decode(ref)) for tid, ref in data["target_priors"]
            ),
            extra=_canonical_pairs(data["extra"], "extra"),
        )
    raise TypeError(f"unknown declarative type tag '{kind}'")


def to_json(obj) -> str:
    """Serialize a declarative object canonically (sorted keys, no spaces)."""
    return json.dumps(_encode(obj), sort_keys=True, separators=(",", ":"))


def from_json(text: str):
    """Deserialize any declarative object from its canonical JSON."""
    return _decode(json.loads(text))


def spec_from_json(text: str) -> MissionSpec:
    """Deserialize a MissionSpec, checking the decoded type."""
    obj = from_json(text)
    if not isinstance(obj, MissionSpec):
        raise TypeError(f"expected MissionSpec JSON, got {type(obj).__name__}")
    return obj


def config_from_json(text: str) -> FidelityConfig:
    """Deserialize a FidelityConfig, checking the decoded type."""
    obj = from_json(text)
    if not isinstance(obj, FidelityConfig):
        raise TypeError(f"expected FidelityConfig JSON, got {type(obj).__name__}")
    return obj


def content_hash(obj) -> str:
    """Return the first 12 hex characters of the object's canonical hash."""
    return hashlib.sha256(to_json(obj).encode("utf-8")).hexdigest()[:12]
