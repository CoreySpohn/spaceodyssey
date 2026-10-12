"""Lazy visualization exports and the base-install import contract."""

import subprocess
import sys

import pytest

import spaceodyssey
import spaceodyssey.viz as viz

PLOT_NAMES = (
    "plot_claim_posterior",
    "plot_planet_plane",
    "plot_certification_curve",
    "plot_tier_agreement",
    "plot_yield_comparison",
    "plot_yield_sweep",
    "plot_plan_visits",
    "plot_observed_data",
)


def test_base_install_without_eyepiece():
    """All plot names fail with an install hint while extraction still works."""
    code = (
        "import sys; sys.modules['eyepiece'] = None\n"
        "import spaceodyssey, spaceodyssey.viz\n"
        "from spaceodyssey.viz import QUANTITY_DISPLAY, CHANNEL_LABELS\n"
        "from spaceodyssey.viz import QuantityDisplay\n"
        "from spaceodyssey.viz import _extract\n"
        f"for name in {PLOT_NAMES!r}:\n"
        "    try:\n"
        "        getattr(spaceodyssey.viz, name)\n"
        "    except ImportError as e:\n"
        "        assert \"pip install 'spaceodyssey[viz]'\" in str(e)\n"
        "    else:\n"
        "        raise SystemExit('no ImportError')\n"
        "assert QUANTITY_DISPLAY['ln_a_eeid'].transform == 'exp'\n"
        "assert CHANNEL_LABELS['y_issued'] == 'issued'\n"
        "assert _extract.quantity_display('unknown') == "
        "QuantityDisplay('unknown', 'identity', 'linear')\n"
        "assert 'matplotlib' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_extract_imports_no_plotting_dependencies():
    """A fresh import of the numerical module loads no plotting stack."""
    code = (
        "import sys\n"
        "import spaceodyssey.viz\n"
        "from spaceodyssey.viz import QUANTITY_DISPLAY, CHANNEL_LABELS, _extract\n"
        "assert not any(name == root or name.startswith(root + '.') "
        "for name in sys.modules for root in ('eyepiece', 'matplotlib', 'hwostyle'))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_dir_lists_lazy_exports():
    """Every planned plot and label export is discoverable."""
    assert set(PLOT_NAMES) <= set(dir(viz))
    assert {"QUANTITY_DISPLAY", "CHANNEL_LABELS", "QuantityDisplay"} <= set(dir(viz))
    assert viz.__all__ == sorted((*PLOT_NAMES, *viz._DATA))
    assert not set(PLOT_NAMES) & set(spaceodyssey.__all__)
    assert "viz" not in spaceodyssey.__all__


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'plot_nonexistent'"):
        _ = viz.plot_nonexistent


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_lazy_plot_dispatch(name, monkeypatch):
    """Dispatch can be checked before the rendering modules are implemented."""
    from spaceodyssey.viz import _require

    function = object()
    calls = []

    def require():
        calls.append("require")

    def import_module(module_name):
        calls.append(module_name)
        return type("Exports", (), {name: function})

    monkeypatch.setattr(_require, "eyepiece", require)
    monkeypatch.setattr(viz.importlib, "import_module", import_module)
    assert getattr(viz, name) is function
    assert calls == ["require", viz._LAZY[name]]
