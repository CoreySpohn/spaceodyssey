<p align="center">
  <a href="https://pypi.org/project/spaceodyssey/"><img src="https://img.shields.io/pypi/v/spaceodyssey.svg?style=flat-square&logo=pypi" alt="PyPI"/></a>
  <a href="https://spaceodyssey.readthedocs.io"><img src="https://readthedocs.org/projects/spaceodyssey/badge/?version=latest&style=flat-square" alt="Documentation Status"/></a>
  <a href="https://github.com/CoreySpohn/spaceodyssey/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/CoreySpohn/spaceodyssey/ci.yml?branch=main&style=flat-square&logo=github&label=tests" alt="Tests"/></a>
  <a href="https://github.com/CoreySpohn/spaceodyssey/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License"/></a>
  <a href="https://github.com/CoreySpohn/spaceodyssey/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/python-3.12%2B-blue.svg?style=flat-square&logo=python" alt="Python 3.12+"/></a>
</p>

---

# spaceodyssey

**spaceodyssey** supports astronomical analysis and observing campaigns: multi-visit observing programs with inference in the loop. Its JAX-based transaction core simulates a campaign as discrete events against a private simulated world, and a document codec writes and reads studies as JSON. Physical adapters connect it to the HWO direct imaging simulation tools:

- **[yippy](https://github.com/CoreySpohn/yippy)**: coronagraph performance data
- **[orbix](https://github.com/CoreySpohn/orbix)**: orbits and keepout geometry
- **[skyscapes](https://github.com/CoreySpohn/skyscapes)**: astrophysical scenes
- **[jaxedith](https://github.com/CoreySpohn/jaxedith)**: count rates and summary radiometry
- **[photomancy](https://github.com/CoreySpohn/photomancy)**: priors and posteriors
- **planit-py**: action and observing-context vocabulary

The physical adapters need the `physical` extra.

## Key Features

*   **Transaction core**: `spaceodyssey.core` orders events, admits effects, accounts for resources and keeps simulated truth apart from what a policy can see
*   **Two executors**: a host executor and a compiled executor that scans over steps and batches worlds with `vmap`, on the same transaction rules
*   **Study documents**: `spaceodyssey.codec` dumps and loads studies through registered, versioned kinds, with checkpoint and resume for long runs
*   **JAX-native**: array-based scientific state and composable simulation components

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install spaceodyssey
```

Or with pip:

```bash
pip install spaceodyssey
```

*(Note: You may need to install JAX separately to match your specific hardware acceleration requirements.)*

## Quick Start

```python
from spaceodyssey import codec, core

print(core.HostExecutor.__name__, codec.load_study.__name__)
```

## Documentation

Full documentation is available at [spaceodyssey.readthedocs.io](https://spaceodyssey.readthedocs.io).

The campaign core was migrated from hwosim at commit edff0c6; hwosim remains as a deprecated compatibility alias.
