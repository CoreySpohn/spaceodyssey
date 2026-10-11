"""Tests for the per-seam implementation registry."""

import ast
import sys
import textwrap
from pathlib import Path

import pytest

from spaceodyssey.errors import RegistryError
from spaceodyssey.registry import _DEFAULT_ADAPTER_MODULES, REGISTRY, Registry, SeamInfo


class SummaryOut:
    """Toy contract type: a summary-level product."""


class FrameOut:
    """Toy contract type: an image-frame product."""


class StatsOut:
    """Toy contract type: detection statistics."""


def qual(cls) -> str:
    """Module-qualified contract name for a class defined in this module."""
    return f"{__name__}.{cls.__name__}"


class ToyImpl:
    """Placeholder implementation class for registration tests."""


@pytest.fixture()
def registry():
    """A fresh, empty registry."""
    return Registry()


class TestRegistration:
    """Registering and looking up implementations."""

    def test_register_and_get(self, registry):
        """A registered class round-trips through get with its metadata."""
        info = SeamInfo(operators={"sample"})
        registry.register("observation", "toy", info)(ToyImpl)
        entry = registry.get("observation", "toy")
        assert entry.cls is ToyImpl
        assert entry.available
        assert entry.info.operators == frozenset({"sample"})

    def test_duplicate_name_raises(self, registry):
        """Registering the same (seam, name) twice is an error."""
        info = SeamInfo(operators={"sample"})
        registry.register("observation", "toy", info)(ToyImpl)
        with pytest.raises(RegistryError, match="already has"):
            registry.register("observation", "toy", info)(ToyImpl)

    def test_unknown_seam_raises(self, registry):
        """Seam names outside the fixed vocabulary are rejected."""
        with pytest.raises(RegistryError, match="unknown seam"):
            registry.register("optics", "toy", SeamInfo(operators={"sample"}))

    def test_unknown_name_lists_known(self, registry):
        """A missing implementation error names the registered alternatives."""
        registry.register("observation", "toy", SeamInfo(operators={"sample"}))(ToyImpl)
        with pytest.raises(RegistryError, match="toy"):
            registry.get("observation", "nope")

    def test_info_coerces_iterables(self):
        """SeamInfo accepts plain lists and sets for its collection fields."""
        info = SeamInfo(operators=["sample"], consumes=["a.B"], produces=["c.D"])
        assert isinstance(info.operators, frozenset)
        assert info.consumes == ("a.B",)


class TestAvailability:
    """import_requirement gates availability without failing lookups."""

    def test_missing_requirement_marks_unavailable(self, registry):
        """An uninstalled requirement yields available=False and a hint."""
        info = SeamInfo(
            operators={"sample"}, import_requirement="not_a_real_package_xyz"
        )
        registry.register("post_processing", "needs_pkg", info)(ToyImpl)
        entry = registry.get("post_processing", "needs_pkg")
        assert not entry.available
        assert entry.cls is None
        assert "not_a_real_package_xyz" in entry.reason

    def test_present_requirement_is_available(self, registry):
        """An installed requirement leaves the entry available."""
        info = SeamInfo(operators={"sample"}, import_requirement="json")
        registry.register("post_processing", "has_pkg", info)(ToyImpl)
        assert registry.get("post_processing", "has_pkg").available


class TestLazyModules:
    """Adapter modules load on first lookup, not at registration."""

    def test_every_builtin_registration_has_a_lazy_module(self):
        """Every module applying register is listed under its declared seam."""
        package_dir = Path(__file__).parents[1] / "src" / "spaceodyssey"
        registrations = []
        for source in sorted(package_dir.rglob("*.py")):
            parts = source.relative_to(package_dir).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module = ".".join(("spaceodyssey", *parts))
            tree = ast.parse(source.read_text(), filename=str(source))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "register"
                    ):
                        seam = ast.literal_eval(decorator.args[0])
                        registrations.append((seam, module))
        assert registrations, "expected built-in seam registrations"
        for seam, module in registrations:
            assert module in _DEFAULT_ADAPTER_MODULES.get(seam, ()), (
                f"{module} must be listed as a lazy adapter for {seam}"
            )

    def test_add_module_defers_import(self, tmp_path, monkeypatch):
        """The module is imported only when its seam is first queried."""
        mod = tmp_path / "toy_lazy_adapter.py"
        mod.write_text(
            textwrap.dedent(
                """
                '''Toy adapter module that self-registers on import.'''

                from spaceodyssey.registry import REGISTRY, SeamInfo


                @REGISTRY.register(
                    "characterization",
                    "toy_lazy",
                    SeamInfo(operators={"sample"}),
                )
                class ToyLazy:
                    '''Registered by importing this module.'''
                """
            )
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        REGISTRY.add_module("characterization", "toy_lazy_adapter")
        assert "toy_lazy_adapter" not in sys.modules
        entry = REGISTRY.get("characterization", "toy_lazy")
        assert "toy_lazy_adapter" in sys.modules
        assert entry.cls.__name__ == "ToyLazy"

    def test_broken_module_raises_registry_error(self, registry):
        """A pending module that fails to import surfaces seam context."""
        registry.add_module("scene", "definitely_missing_adapter_module")
        with pytest.raises(RegistryError, match="scene"):
            registry.names("scene")


class TestMatrix:
    """The pairwise compatibility table derives from declared ports."""

    def test_matrix_from_ports(self, registry):
        """Compatible pairs are True, incompatible pairs False."""
        registry.register(
            "observation",
            "summaries",
            SeamInfo(operators={"sample"}, produces=(qual(SummaryOut),)),
        )(ToyImpl)
        registry.register(
            "observation",
            "frames",
            SeamInfo(operators={"sample"}, produces=(qual(FrameOut),)),
        )(ToyImpl)
        registry.register(
            "post_processing",
            "frame_filter",
            SeamInfo(
                operators={"sample"},
                consumes=(qual(FrameOut),),
                produces=(qual(StatsOut),),
            ),
        )(ToyImpl)
        table = registry.matrix("observation", "post_processing")
        assert table[("frames", "frame_filter")] is True
        assert table[("summaries", "frame_filter")] is False
