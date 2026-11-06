---
orphan: true
---

# Slice 2 T2: numbered Agenda levels

Each phase of an `Agenda` now carries a level, an integer from 0 to 3, beside its
phase mask. A level names a fixed, nested mask over `EffectKind`, and admission
refuses with `AdmissionCode.FORBIDDEN` any effect whose bit is clear in the
intersection of the phase mask and the level mask. The core holds the levels as
numbers only. Every Slice 1 history is unchanged: all 112 golden digests match.

## Levels

`LEVEL_EFFECTS` (in `core.enums`, exported from `spaceodyssey.core`) holds the
four masks by level number:

| Level | Effects permitted |
|---|---|
| 0 | `NONE`, `CANCEL` |
| 1 | level 0 plus `OBSERVE`, `WAIT`, `QUERY`, `STOP` |
| 2 | the same as level 1 |
| 3 | level 2 plus `ALLOCATE`, which admission still answers `UNSUPPORTED` |

The default level is 3. Its mask holds every `EffectKind`, so the intersection
leaves each phase mask as it was and every existing study keeps its behaviour.

## What changed

- `Agenda` gains `levels`, an `I64` array with one level per policy phase. The
  constructor takes it as an optional fourth argument; when it is left out, every
  phase is at level 3. A level outside 0 to 3, or a count that differs from the
  number of policies, is refused at construction (indexing would otherwise clip an
  unknown level to a real mask).
- `Agenda.effective_effects()` returns the per-phase mask admission applies:
  `permitted_effects & LEVEL_EFFECTS[levels]`. The transaction's phase mask reads
  it in place of `permitted_effects`.
- The domain-word lint (`tools/check_domain_words.py`) denies the level role
  names `planner`, `allocator` and `integration`, so they cannot enter the core.
  The fourth role name is two common words and is not listed.
- `docs/core.md` describes the levels.

## Public API

Added: `LEVEL_EFFECTS`, `Agenda.levels`, `Agenda.effective_effects()`. `Agenda`
now has an explicit constructor with the signature
`Agenda(policies, phase_until, permitted_effects, levels=None)`; existing
three-argument calls are unchanged. A pytree of a study gains one integer leaf.

## Tests

Added, in `tests/core/test_agenda_levels.py` (unit tier, 7 cases):

- a level-1 phase emitting `Allocate` gets `FORBIDDEN` and the carry is
  unchanged; the same emission at level 3 gets `UNSUPPORTED` (one test, two
  cases);
- the masks equal the level table, are nested, and the default level permits
  every effect;
- the effective mask is the phase mask intersected with each phase's own level
  (three phases: a level cuts an open phase mask, a phase mask cuts an open
  level, and an effect both permit is admitted);
- an unknown level or a missing level is refused (three cases).

The lint's per-word negative control (`test_lint_catches_each_denied_token`) gains
three cases, one per new denied word, through the pinned word list in
`tests/core/lint_helpers.py`.

Retired: none. Net: 10 quick (7 new, 3 lint cases), 0 slow.

Deciding tests: `tests/core/test_golden_histories.py`,
`tests/core/test_corpus_invariants.py`, `tests/core/test_parity_values.py` and
`tests/test_slice1_gate.py` pass with the manifest unchanged.

## Timing

Quick tier (`-m "not slow" -n 2 --dist loadfile -p no:randomly`, through the
workspace compute queue, macOS arm64):

| | Tests | Wall | Peak RSS |
|---|---|---|---|
| Before (`ab85828`) | 1,701 passed | 95.2 s | 2.78 GB |
| After | 1,711 passed | 55.7 s | 3.13 GB |

The before run also reported one collection error in
`tests/core/test_acquisition_index.py` (a truncated entry in the shared JAX
persistent compilation cache, raised as a warning); that file is wholly in the
slow tier, so no quick test was lost. The two wall times differ through machine
load, not this change: the seven new tests sum to about 2 s (the slowest, the
level-1 `Allocate` case, 1.2 s with its compile) and the three lint cases to
under 0.1 s.

Full suite after: 2,767 passed in 1,360 s wall (`-n 2 --dist loadfile`), peak
footprint 7.6 GB. Its four budget overruns (`test_record_statuses`,
`test_profile`, two in `test_core`) are in files this task does not touch.

## Time spent

About one hour of agent time, of which 23 minutes was the full-suite run.
