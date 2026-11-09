# Golden pins under one jaxlib, in their own CI job

The golden history pins are bit-exact only under one jaxlib release. Float64 normal draws
go through an inverse error function whose rounding differs between jaxlib releases: under
jaxlib 0.10.2, 21 of the 120 pinned history entries differ, all in float leaves (the first
by 2 ULP in `payload`), while the same machine under 0.12.0.1 matches all 120. CI resolved
JAX from PyPI without an upper bound, so a new JAX release could move the pins on CI with
no change to this library.

## What changed

- **CI installs one JAX release.** `.github/jax-pin.txt` pins `jax` and `jaxlib` to
  0.12.0.1, the release the pins were taken under, and is applied as a uv override in
  `ci.yml` and `pilot.yml`. Users are unaffected: `pyproject.toml` keeps its floors and no
  upper bound. The upstream canary does not read the file, so pre-release and nightly JAX
  are still tested there.
- **Failures name the environment.** The goldens read the pinned jaxlib from the same
  file. The history check compares the integer digests before the float digests, and a
  float, fingerprint or fault mismatch under another jaxlib says which release the pins
  were taken under, so a float-only move there is read as an environment difference first.
  The pin file itself is unchanged; its platform note predates this finding.
- **The pins have their own nightly job.** `Golden pins` runs
  `tests/core/test_golden_histories.py` on the nightly schedule and on a manual slow or
  full run. The slow tier skips that file, so a slow tier that overruns its job limit no
  longer takes the pins with it. The slice gates still run the golden tests in their own
  child runs.

## Moving to a new JAX release

Bump both lines of `.github/jax-pin.txt` in a commit that changes nothing else, and run
the goldens and the slice gates under the new release. If everything matches, the bump
lands. If only float digests moved, the pin no longer holds under that release: the pin is
never regenerated in place, and the slice gates compare its bytes with its capture commit,
so either the bump waits or a new pin capture is decided and recorded in its own change
record. Integer digests that moved are a regression under any release.
