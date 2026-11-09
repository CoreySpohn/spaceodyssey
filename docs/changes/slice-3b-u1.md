---
orphan: true
---

# Slice 3b U1: realize context

A channel model that declares `version = 2` now receives a context describing the record it
realizes and returns the whole payload row. A version-1 model keeps its call, its payload
layout and its compiled program.

## What changed

- `core/kinds.py`: `RealizeContext` (`start_tick`, `end_tick`, `acquisition_index`,
  `fraction`, each read from the record being realized) and `realize_row`, which branches on
  the static `channel_model.version`.
- Version 1: `realize(truth, subject, channel_index, fraction, key, design, channels)`
  returns `(value, variate)`; the core writes value, variate and fraction into columns 0 to 2,
  exactly as before.
- Version 2: the same arguments plus a keyword-only `context` return a one-dimensional row no
  wider than `Capacities.payload_width`; the core pads it with zeros. A wider or
  multi-dimensional row raises `ValueError` at trace time, and so does any version other than 1 or 2.
- `core/handlers.py` (`_realize_rec`, used by Done and by cancel) and `core/battery/fork.py`
  (`_keyed_draw`) call `realize_row` on the record, so both build the same context. The fork
  battery compares the full row for a version-2 model and the first two columns for version 1.
- No codec source change: the existing kind registry already registers a version-2 class
  with a `migrate` from version 1. No new declaration field, carry leaf or trace leaf.
- Test double: `tests.doubles.ToyEpochPhotometry`, a version-2 model whose row is
  `[start_tick, end_tick, acquisition_index, fraction, z]`, which needs a five-column kind.

## Tests

| Test | Tier |
| --- | --- |
| version-2 row is the context function of the record (all four fields); context fields | unit |
| too-wide row and unknown versions 0 and 3 refused (parametrized) | unit |
| version-2 study document round trip; version-1 reference migrates | unit |
| version-2 log row carries its context; delay moves it by the wait; host and compiled histories equal | medium (6.0 s cold) |
| cancelled version-2 row carries its logged start tick, end tick, acquisition index and a fraction below 1 | medium (2.4 s cold) |
| fork battery of a delayed version-2 study | medium (5.3 s cold) |

Negative controls: shifting the start tick, the end tick or the acquisition index inside the
fork battery's context makes `fork_check` fail. The version-1 history guard is the golden
test file.

## Receipts

- Program budget for `tests/core/test_realize_context.py`: 120 programs measured cold,
  budget 148.
- Default program (measured at merge time, not a standing test): the lowered jaxpr of one `transition` of `survey_1` is identical (same
  length and hash) before and after the change.
- Golden check: `tests/core/test_golden_histories.py -m ""` passes unchanged (124 passed, 353 s); no golden or pin file was touched.
- Quick tier: tests/core, tests/codec and tests/test_package.py pass (1499 of 1501; two unit tests of `test_core.py` and two medium tests of `test_compile_sharing.py` exceeded their time budgets in a two-worker run under load, and `test_core.py` passes with no violation when run alone).
- Test-work share: one test file, one test double and one budget line.
