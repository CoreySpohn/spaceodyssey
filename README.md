<p align="center">
  <a href="https://pypi.org/project/spaceodyssey/"><img src="https://img.shields.io/pypi/v/spaceodyssey.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://spaceodyssey.readthedocs.io"><img src="https://readthedocs.org/projects/spaceodyssey/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/CoreySpohn/spaceodyssey/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/spaceodyssey/tests.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/spaceodyssey/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/CoreySpohn/spaceodyssey"><img src="https://img.shields.io/badge/python-3.11%20|%203.12-blue.svg?style=flat-square&logo=python" alt="Python Versions"></a>
</p>

---

# spaceodyssey

**spaceodyssey** supports astronomical analysis and observing campaigns: multi-visit observing programs with inference in the loop. Its JAX-based campaign core runs closed-loop mission simulations and estimates exoplanet yields. It ties together the full suite of HWO direct imaging simulation tools:

- **[yippy](https://github.com/CoreySpohn/yippy)** — Coronagraph performance modeling
- **[orbix](https://github.com/CoreySpohn/orbix)** — Orbital dynamics and target scheduling
- **[coronagraphoto](https://github.com/CoreySpohn/coronagraphoto)** — Image simulation for coronagraphic observations
- **[coronalyze](https://github.com/CoreySpohn/coronalyze)** — Post-processing and SNR analysis
- **planit-py**: Action and observing-context vocabulary

## Key Features

*   **Mission Simulation**: End-to-end simulation pipelines from target selection through observation and analysis
*   **Yield Estimation**: Detection completeness and exoplanet yield calculations
*   **Integration Orchestration**: Unified interface across the HWO simulation suite
*   **JAX-Native**: Array-based scientific state and composable simulation components

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install spaceodyssey
```

Or with pip:

```bash
pip install spaceodyssey
```

The base installation includes the campaign core. Optional physics libraries
for planned adapters are grouped by role:

| Extra | Libraries | Adapter role |
| --- | --- | --- |
| `physics-etc` | jaxedith | Exposure-time measurements |
| `physics-scene` | skyscapes, orbix | Universe sources and observing context |
| `physics-image` | coronagraphoto, coronachrome, coronalyze, yippy | Image-level instruments, post-processing and characterization |

Install all physics libraries with `uv pip install "spaceodyssey[all-physics]"`,
or select an extra such as `uv pip install "spaceodyssey[physics-scene]"`.
Plotting requires `spaceodyssey[viz]`, which includes eyepiece and matplotlib.
Physics libraries remain optional until their adapters import them.

*(Note: You may need to install JAX separately to match your specific hardware acceleration requirements.)*

## Quick Start

```python
import spaceodyssey
```

## Documentation

Full documentation is available at [spaceodyssey.readthedocs.io](https://spaceodyssey.readthedocs.io).

The campaign core was migrated from hwosim at commit edff0c6; hwosim remains as a deprecated compatibility alias.
