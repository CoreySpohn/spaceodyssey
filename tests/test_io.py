"""Tests for run manifests and run-directory persistence."""

from spaceodyssey.io import RunManifest, read_manifest, write_run
from spaceodyssey.metrics import YieldSummary
from tests.doubles import make_config, make_spec


def make_manifest(seed: int = 0, tag: str = "") -> RunManifest:
    """A manifest over the toy spec and config."""
    return RunManifest.create(make_spec(), make_config(), seed=seed, tag=tag)


class TestRunManifest:
    """Manifest capture, identity, and round-tripping."""

    def test_create_captures_versions_and_precision(self):
        """The manifest records installed versions and the precision flag."""
        manifest = make_manifest()
        versions = dict(manifest.library_versions)
        assert "spaceodyssey" in versions
        assert "planit-py" in versions
        assert manifest.precision_x64 in (True, False)

    def test_run_id_is_content_derived(self):
        """Equal content gives equal ids; the seed moves the id."""
        assert make_manifest().run_id == make_manifest().run_id
        assert make_manifest(seed=1).run_id != make_manifest(seed=0).run_id

    def test_tag_suffixes_the_id(self):
        """A tag distinguishes deliberate re-runs readably."""
        tagged = make_manifest(tag="rerun")
        assert tagged.run_id.endswith("-rerun")
        assert tagged.run_id.split("-")[0] == make_manifest().run_id

    def test_json_round_trip(self):
        """The manifest survives serialization exactly."""
        manifest = make_manifest(tag="a")
        assert RunManifest.from_json(manifest.to_json()) == manifest


class TestRunDirectory:
    """Writing and reading a run directory."""

    def test_write_and_read_back(self, tmp_path):
        """write_run produces exactly the manifest and report files."""
        manifest = make_manifest()
        summary = YieldSummary(
            operator="sample",
            certificate="detection",
            certified_targets=(1,),
            false_certified_targets=(),
            epochs=3,
            resources_spent=(("time_d", 3.3),),
        )
        run_dir = write_run(tmp_path / manifest.run_id, manifest, summary)
        assert sorted(p.name for p in run_dir.iterdir()) == [
            "manifest.json",
            "report.json",
        ]
        assert read_manifest(run_dir) == manifest
