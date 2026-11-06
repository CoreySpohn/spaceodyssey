---
orphan: true
---

# Slice 2 T4d: the OFTI host-loop refit

T4d binds photomancy's OFTI rejection sampler to the refresh protocol as a
`HostLoopRefit`, and declares the model it samples. Everything is in
`src/spaceodyssey/adapters/photomancy.py`. No file under `core/` or `codec/` changed, no
`pyproject.toml` entry changed, and no golden history moved.

## What is in

- `InferenceModel(name, prior, likelihood, support, parameterization, mass_model)`, a
  frozen dataclass of plain JSON data. `declaration()` is the JSON object and `digest()`
  is the positive 48-bit integer from the sha256 of its canonical bytes. The type did not
  exist; D13 says a declaration is "hashed into each binding", and the codec already
  carries `model_hash` as an integer, so the minimal type lives beside its first
  consumer. T7 can import it or promote it.
- `ofti_model(stellar_mass_kg, dist_pc, ecc_prior, log_P_range, e_max)` declares OFTI's
  actual target: isotropic orientation, `kipping13` eccentricity by default, the
  semimajor axis and node prior induced by scale-and-rotate, uniform mean anomaly at the
  reference epoch, period and eccentricity support, a fixed mass and distance, and the
  seven-name parameterization `PARAM_NAMES`. It needed no undeclared model change, so the
  D13 stop rule did not trigger.
- `OftiRefit`, a `HostLoopRefit` subclass. Its `model_hash` is `ofti_model(...).digest()`,
  so it matches how `AbstractRefitBoundary.binding`, `dump_run` and `load_run` read and
  check the hash.
- `ofti_binding()` and `photomancy_bindings()`: the `BindingKind` (name
  `photomancy.ofti`, version 1) and a `Bindings` table holding it. The configuration is
  `OftiRefit.configuration()`; `build` rebuilds the boundary and the codec then compares
  the rebuilt hash with the stored one.

## How a refit reads its entry

- Counting rows are `entry.rows.valid`. Payload columns, by index and in the order epoch
  (days), east and north offset (arcsec), their uncertainties and the correlation, come
  from `columns` (default 0 to 5).
- The key is `entry.key`, the core's `refit_key`, passed unchanged to `ofti`. The adapter
  holds no counter, clock or cache, so the answer is a function of the rows and the key
  (D4).
- The particle count comes from the observer's belief layout: the `particles` leaf has
  tail shape `[n_particles, 7]`, `n_accept` is `n_particles`, and the samples are cut to
  `[:n_particles]` (OFTI already truncates to `n_accept`). An optional `count` leaf gets
  the number of epochs fit. Any other subject leaf is a `ValueError` naming the leaves,
  since a configuration mismatch is a programming error and not a refit outcome.
- The reference epoch is the highest signal-to-noise valid epoch, OFTI's own rule, passed
  explicitly. With `loglike_ref="strict"` (the default) the rejection bound is the sum of
  the Gaussian log normalizations of the nonreference epochs, a true bound, so the
  particles sample the posterior. `"first_batch"` is OFTI's empirical bound. Neither is
  part of the model hash, which declares the target and not the sampler.

## Statuses

| Case | Status | Installed |
|---|---|---|
| `n_particles` orbits accepted | `OK` | yes |
| No counting rows | `NO_SUPPORT` | no |
| `RuntimeError` from OFTI whose message starts "OFTI accepted" (batches ran out) | `NO_SUPPORT` | no |
| A counting row with a non-finite value, a non-positive uncertainty or `abs(corr) >= 1`, or non-finite particles | `FAILED` | no |
| `DEGRADED` | never returned | |

A non-installing status leaves the belief as it was (the core's `commit_refresh`) and is
recorded on the product row and in `Carry.refresh_status`. Any other exception, including
a JAX runtime error, propagates. The exhaustion test pins the message prefix, so a change
in photomancy's wording fails a test instead of turning into a crash.

A masked audience column (an observer-only uncertainty, say) reads as zero for the policy,
so the policy's entry is `FAILED` while the observer's is `OK`; the core installs each
audience independently.

## Registration

`Bindings` is an explicit table passed to `dump_run` and `load_run`; T5c's entry points
register study kinds only (`FAMILIES` has no binding family), so there is nothing to add
to `pyproject.toml`. photomancy is already a hard dependency (`photomancy>=0.3`), as
planit-py is, so there is no extra to add either. Core imports no adapter and no
photomancy; `tests/core/test_lints.py` and `tools/check_core_imports.py` pass unchanged,
and its negative control for `spaceodyssey.adapters.photomancy` already existed.

## Tests

`tests/adapters/test_photomancy_ofti.py`, six tests; a toy study of one subject with
three scripted astrometry rows (ticks 10, 40, 70) and a checkpoint at tick 100, a
particle observer that changes only by refresh, 16 particles and OFTI batches of 4,096.
The module skips when photomancy is not installed.

Quick (unit budget):

- `test_a_failed_or_unsupported_entry_is_a_status_never_an_exception`: exhausted batches,
  no rows and a zero uncertainty map to `NO_SUPPORT`, `NO_SUPPORT`, `FAILED` with no
  slices; a wrong observer is a `ValueError`.
- `test_the_same_rows_and_key_give_the_same_particles_and_a_new_set_a_new_key`: equal
  entry and key give equal particle bytes, another key gives other particles, and a
  changed revision changes the refit key.
- `test_a_model_hash_is_stable_and_a_mismatch_between_executors_is_refused`: the digest is
  stable and moves with prior, mass, support and eccentricity cap; `dump_run` of a host
  and a compiled boundary of different models is `D003`.

Medium (the study compiles once, shared through a module fixture of the frozen world):

- `test_a_checkpoint_fit_installs_for_both_audiences_and_replays`: one checkpoint fit
  installs the particles and the epoch count for both audiences (the policy copy is the
  deduplicated entry), clears the dirty flags, and answering the same frozen request again
  gives the installed particles bit for bit.
- `test_exhausted_batches_leave_the_belief_unchanged`: the same frozen world with a
  starved sampler records `NO_SUPPORT` for both audiences and leaves the belief at zero.
- `test_a_reloaded_run_document_reproduces_the_host_and_compiled_histories`: a run
  document naming the OFTI boundary reloads to an equal history on the host executor, and
  the compiled chunked driver (`chunk=7`) with the host boundary equals the host run.

Replay gap: keyed replay of a refresh (D12) is T4c and is not on main, so "replay" here is
the re-answered frozen request plus the document round trip, both through public pieces
(`gather_refit_rows`, `refit_batch`, `load_run`). When T4c lands, a test that replays the
run's trace through it and compares with the installed particles belongs beside these.

## Timing

Cold (compile cache off, four-thread compute launcher, module fixture counted in the first
medium test):

| Test | Seconds |
|---|---|
| status mapping | 1.6 |
| key determinism | 3.2 |
| model hash and D003 | 0.1 |
| fixture (study to its checkpoint) | 4.0 |
| install and replay | 3.3 |
| exhausted batches | 0.8 |
| document reload, host and compiled | 12.0 |

All six passed in 26 s cold and 9 s with the persistent cache warm.

## Decisions

- `InferenceModel` is added here, not in `core/`: it is data without behavior in the
  core, and only adapters declare models.
- The strict rejection bound is the default: the empirical bound is an approximation
  (OFTI's own docstring), and the model pair in D13 compares against OFTI as a posterior.
- A mismatch between the observer's leaves and the adapter's is an exception, not a
  status.

## Full suite

Worktree, entry-point install test deselected, `-n 2 --dist loadfile`, four threads: 3,073 tests, 3,072 passed and 1 failed in 1,961 s, peak footprint 8.7 GiB. The failure,
`test_slice1_gate.py::test_corpus_invariants[baseline-compiled-K3]`, was a `UserWarning` for a truncated entry in the shared persistent compile cache (`Error -5 while
decompressing`), written while another pytest process used the same cache directory; it passes alone (9 s). Three existing unit tests exceed the 5 s report budget
(`test_statuses_agree_between_executors_and_replay`, `test_gain_enters_measurement`, `test_every_hand_study_reproduces_its_history`); none is touched by T4d. The golden-history
test passed, so no history moved.
