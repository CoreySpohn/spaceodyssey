"""Run manifests and run-directory persistence.

A run is a directory with a manifest; replay rebuilds from the manifest and
never unpickles live objects. The manifest holds the declarative spec and
configuration, the seed, the library versions, and the precision flag, so a
run's identity is its content: identical configurations collide on run id
loudly instead of duplicating silently.
"""

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import jax

from spaceodyssey.spec import (
    FidelityConfig,
    MissionSpec,
    config_from_json,
    content_hash,
    spec_from_json,
    to_json,
)

_SUITE_DISTRIBUTIONS: tuple[str, ...] = (
    "spaceodyssey",
    "planit-py",
    "equinox",
    "jax",
)


@dataclass(frozen=True)
class RunManifest:
    """Everything needed to rebuild one run.

    Attributes:
        spec: The declarative problem statement.
        config: The declarative implementation selection.
        seed: The run's root seed.
        library_versions: Installed (distribution, version) pairs for the
            packages the run depends on.
        precision_x64: Whether 64-bit floats were enabled.
        calibrations: Identifiers of calibration artifacts the run consumed.
        tag: Optional human-readable suffix distinguishing deliberate
            re-runs of one configuration.
    """

    spec: MissionSpec
    config: FidelityConfig
    seed: int
    library_versions: tuple[tuple[str, str], ...]
    precision_x64: bool
    calibrations: tuple[str, ...] = ()
    tag: str = ""

    @classmethod
    def create(
        cls,
        spec: MissionSpec,
        config: FidelityConfig,
        *,
        seed: int,
        calibrations: tuple[str, ...] = (),
        tag: str = "",
    ) -> "RunManifest":
        """Build a manifest, capturing installed versions and precision."""
        versions = []
        for distribution in _SUITE_DISTRIBUTIONS:
            try:
                versions.append(
                    (distribution, importlib.metadata.version(distribution))
                )
            except importlib.metadata.PackageNotFoundError:
                continue
        return cls(
            spec=spec,
            config=config,
            seed=seed,
            library_versions=tuple(versions),
            precision_x64=bool(jax.config.jax_enable_x64),
            calibrations=tuple(calibrations),
            tag=tag,
        )

    @property
    def run_id(self) -> str:
        """Content-derived run identifier.

        Depends on the spec, configuration, seed, and precision, not on
        library versions: re-running an identical configuration after an
        upgrade collides on purpose, and the tag is how deliberate re-runs
        distinguish themselves.
        """
        identity = (
            f"{content_hash(self.spec)}:{content_hash(self.config)}:"
            f"{self.seed}:{int(self.precision_x64)}"
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return f"{digest}-{self.tag}" if self.tag else digest

    def to_json(self) -> str:
        """Serialize canonically (sorted keys)."""
        return json.dumps(
            {
                "spec": json.loads(to_json(self.spec)),
                "config": json.loads(to_json(self.config)),
                "seed": self.seed,
                "library_versions": [[k, v] for k, v in self.library_versions],
                "precision_x64": self.precision_x64,
                "calibrations": list(self.calibrations),
                "tag": self.tag,
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> "RunManifest":
        """Deserialize from :meth:`to_json` output."""
        data = json.loads(text)
        return cls(
            spec=spec_from_json(json.dumps(data["spec"])),
            config=config_from_json(json.dumps(data["config"])),
            seed=data["seed"],
            library_versions=tuple((k, v) for k, v in data["library_versions"]),
            precision_x64=data["precision_x64"],
            calibrations=tuple(data["calibrations"]),
            tag=data["tag"],
        )


def write_run(run_dir, manifest: RunManifest, summary) -> Path:
    """Write one run's manifest and report into a directory.

    Args:
        run_dir: Directory to create (parents included).
        manifest: The run's manifest, written as ``manifest.json``.
        summary: The run's YieldSummary, written as ``report.json``.

    Returns:
        The run directory path.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(manifest.to_json() + "\n")
    (run_dir / "report.json").write_text(summary.to_json() + "\n")
    return run_dir


def read_manifest(run_dir) -> RunManifest:
    """Read the manifest back from a run directory."""
    text = (Path(run_dir) / "manifest.json").read_text()
    return RunManifest.from_json(text)
