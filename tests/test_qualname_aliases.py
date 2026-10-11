"""Compatibility of historical qualified names and persisted provenance."""

import importlib

import pytest

import spaceodyssey
from spaceodyssey.contracts import compatible, resolve_contract
from spaceodyssey.errors import RegistryError
from spaceodyssey.reporting import PairedDifference
from spaceodyssey.spec import from_json


@pytest.fixture(autouse=True)
def forbid_legacy_import(monkeypatch):
    original = importlib.import_module

    def checked(name, *args, **kwargs):
        assert name != "hwosim" and not name.startswith("hwosim.")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", checked)


@pytest.mark.parametrize(
    "name",
    [
        name
        for name in spaceodyssey.__all__
        if isinstance(getattr(spaceodyssey, name), type)
    ]
    + ["data.SummaryDataset", "certify.SnrThreshold"],
)
def test_contract_alias_resolves_identical_class(name):
    assert resolve_contract(f"hwosim.{name}") is resolve_contract(
        f"spaceodyssey.{name}"
    )


def test_mixed_contract_names_are_compatible():
    assert compatible(("hwosim.SummaryDataset",), ("spaceodyssey.SummaryDataset",))
    assert compatible(("spaceodyssey.SummaryDataset",), ("hwosim.SummaryDataset",))


@pytest.mark.parametrize(
    "qualname",
    [
        "spaceodyssey.engines.analytic.PairedDifference",
        "spaceodyssey.reporting.PairedDifference",
        "hwosim.engines.analytic.PairedDifference",
        "hwosim.reporting.PairedDifference",
    ],
)
def test_paired_difference_contract_resolves_identical_class(qualname):
    assert resolve_contract(qualname) is PairedDifference


@pytest.mark.parametrize("name", ["SnrThreshold", "certify.SnrThreshold"])
def test_manifest_alias_build_preserves_provenance(tmp_path, name):
    spec = spaceodyssey.MissionSpec.make(
        "campaign", 30.0, scientific_id="target-123", model_id="model-456"
    )
    old_ref = spaceodyssey.SeamRef.make(
        "certification", f"hwosim.{name}", threshold=7.0
    )
    config = spaceodyssey.FidelityConfig.make("sample", certification=old_ref)
    manifest = spaceodyssey.RunManifest(
        spec=spec,
        config=config,
        seed=23,
        library_versions=(("hwosim", "0.1.0"),),
        precision_x64=False,
        calibrations=("calibration-789",),
        tag="original",
    )
    payload = manifest.to_json()
    run_id = manifest.run_id
    hashes = (spaceodyssey.content_hash(spec), spaceodyssey.content_hash(config))
    (tmp_path / "manifest.json").write_text(payload)
    loaded = spaceodyssey.read_manifest(tmp_path)
    mission = spaceodyssey.build(loaded.spec, loaded.config)
    new_config = spaceodyssey.FidelityConfig.make(
        "sample",
        certification=spaceodyssey.SeamRef.make(
            "certification", f"spaceodyssey.{name}", threshold=7.0
        ),
    )
    current = spaceodyssey.build(spec, new_config)
    assert type(mission.truth.certification) is type(current.truth.certification)
    assert mission.truth.certification.threshold == 7.0
    assert mission.model.certification is mission.truth.certification
    assert loaded.to_json() == payload
    assert loaded.run_id == run_id
    assert (
        spaceodyssey.content_hash(mission.spec),
        spaceodyssey.content_hash(mission.config),
    ) == hashes
    assert from_json(spaceodyssey.to_json(old_ref)) == old_ref
    assert mission.config.choice("certification").truth.name == f"hwosim.{name}"


def test_qualified_lookup_keeps_registry_names_and_seam_validation():
    entry = spaceodyssey.REGISTRY.get("certification", "hwosim.SnrThreshold")
    assert entry is spaceodyssey.REGISTRY.get("certification", "snr_threshold")
    assert entry.name == "snr_threshold"
    with pytest.raises(RegistryError):
        spaceodyssey.REGISTRY.get("cost", "hwosim.SnrThreshold")


def test_lazy_module_alias_uses_current_module():
    registry = spaceodyssey.Registry()
    registry.add_module("certification", "hwosim.certify")
    assert registry.names("certification") == ()


def test_alias_does_not_change_unrelated_names():
    from spaceodyssey.aliases import resolve_qualname_alias

    assert resolve_qualname_alias("hwosim_extra.Type") == "hwosim_extra.Type"
    assert resolve_qualname_alias("other.hwosim.Type") == "other.hwosim.Type"
    assert resolve_contract("builtins.str") is str


def test_ambiguous_qualified_name_requires_registered_name():
    registry = spaceodyssey.Registry()
    info = spaceodyssey.SeamInfo(operators={"sample"})
    registry.register("certification", "first", info)(spaceodyssey.SnrThreshold)
    registry.register("certification", "second", info)(spaceodyssey.SnrThreshold)
    with pytest.raises(RegistryError, match="ambiguous"):
        registry.get("certification", "hwosim.SnrThreshold")
    assert registry.get("certification", "first").cls is spaceodyssey.SnrThreshold


def test_qualified_name_retains_availability_checks():
    registry = spaceodyssey.Registry()
    info = spaceodyssey.SeamInfo(
        operators={"sample"}, import_requirement="missing_optional_adapter_123"
    )
    registry.register("certification", "optional", info)(spaceodyssey.SnrThreshold)
    entry = registry.get("certification", "hwosim.SnrThreshold")
    assert not entry.available
    assert entry.cls is None
    config = spaceodyssey.FidelityConfig.make(
        "sample",
        certification=spaceodyssey.SeamRef.make("certification", "hwosim.SnrThreshold"),
    )
    with pytest.raises(spaceodyssey.WiringError, match="missing_optional_adapter"):
        spaceodyssey.build(
            spaceodyssey.MissionSpec.make("campaign", 30.0), config, registry
        )
