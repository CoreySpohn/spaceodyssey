# Slice 1, T6: record

## Behaviour changes

- Six statuses. `RecordStatus` gains `NULL` (the acquisition ran and produced no usable
  value) and `UNRELEASED` (logged but held back). A held row is never released, never
  accepted and never visible in a snapshot (`visible_rows`, `h_release`). No handler
  writes either status yet; a process or a test writes them, and the toy observer folds
  `COMPLETED` rows only.
- `Record` and `RecordLog` gain `audience` (`Audience` flags the row is released to),
  copied from the channel at settlement and cancel and from the process event for process
  rows. `canonical_history` records `audience`, `revision` and `superseded_by` next to the
  integer columns it already carried.
- `Audience.SCORER` joins `OBSERVER` and `POLICY`. It is meaningful on a payload column
  only: a column whose audience is `SCORER` alone stays in the log and is zero in every
  snapshot and in every record the observer folds. `EVERYONE` is the union of the three.
- `RecordKind` and `PayloadColumn` (frozen dataclasses) declare a layout; a column with
  no audience, an empty one or an unknown bit fails at construction. `KindRegistry` is an
  in-process list of kinds; a kind's index is its record `kind`. `KindRegistry.lower`
  returns capacities with `payload_width` widened to the widest kind (the declared width
  is a floor) and the new static `Capacities.payload_audience` table, one row per kind.
  There are no entry points and no versions.
- `snapshot_view` masks payload columns by the audience of the view and reports
  `superseded_by` over the rows that audience can see. `h_release` and `replay_belief`
  fold a record masked for the observer, and the policy belief folds one masked for the
  policy. Capacities with no lowered registry leave every column readable; a kind index
  the registry lacks has no readable column.
- Revisions. The observer accepts every released row and folds the first accepted
  revision of an acquisition; the policy belief does the same over the rows released to
  the policy. A later revision is logged and accepted but not folded, and the earlier
  rows are marked `superseded_by` it (a revision that arrives first marks the older row
  that arrives after it). `dup_fold` still means one row released twice.
  `replay_belief` skips later revisions the same way.
- The toy study declares its kinds (`toy_registry`, `MEASUREMENT_KIND`, `AUXILIARY_KIND`):
  the noise variate, which let a policy recover the true signal from a released row, is
  now a scorer-only column. `survey_1` lowers the toy registry onto any capacities that
  have none, and its channel table has an unused fourth channel (id 13) when
  `n_channels` is 4.
- `studies/catalog.py` defines the second record kind: a catalog row on channel 13 that is
  released to the observer only, 150 ticks after the occupancy ends, and folded into
  auxiliary slot 1 by `CatalogObserver`. The core names nothing of it.

## Carry changes (shared with the release task)

- `Carry` is unchanged. `RecordLog` (and so `Carry.log`) gains the `audience` column.
- `Capacities` gains `payload_audience` (static, a tuple of tuples). `Trace` and `Study`
  are unchanged.

## Tests

- `test_record_kinds.py`: two kinds of different widths lower to one width, a kind or
  column with an undeclared audience fails at construction, registry indices, the toy
  registry, and column readability by audience.
- `test_record_statuses.py`: all six statuses logged by a scripted process, the held
  row never released or accepted, `NULL` released but not folded, executor agreement and
  replay, and the identities and provenance of a settled record.
- `test_revisions.py`: `test_revision_supersedes` (both orders, the observer-only case,
  executor agreement and replay), `test_record_folded_once` over five corpus cases, and
  the double release of one row.
- `test_scorer_columns.py`: `test_snapshot_hides_scorer_columns` over every corpus case
  on both executors (a spy policy reads every view the run hands out, and the final
  snapshots are checked too), `test_policy_cannot_recover_truth`, and the negative
  controls: the old layout is lowered onto the same studies and the same checks fail
  (`test_the_leaking_layout_is_caught_by_the_same_check`,
  `test_the_leaking_layout_lets_the_same_policy_recover_truth`).
- `test_second_kind.py`: `test_second_kind_needs_no_core_edit` (the core never names the
  catalog, `RecordKind("catalog", ...)` is built only in `studies/catalog.py`, and the
  kind registers through `KindRegistry.register` after the toy kinds, which are
  unchanged), the observer receiving the row at its release tick, the policy never
  seeing it, and executor agreement.

## Rebaselined cases

None. The scorer-only column keeps its position in the payload, so every logged payload,
hand history and parity comparison is unchanged; only what a snapshot shows changed.

## Deviations from the spec

- The second audience is the observer. A catalog row has audience `OBSERVER`: it is
  accepted at its release tick (`accepted_tick`) and its `released_tick` stays -1,
  because `released_tick` names release to the policy. No further audience flag was added.
- `NULL` and `UNRELEASED` are honoured by the release path but nothing in the core
  produces them yet; the channel contract (`realize` returns a value and a variate) did
  not change, so a channel cannot choose a status.
- The channel model still fills payload columns 0 to 2 (value, variate, fraction); a
  kind that a channel realizes must lay its columns out that way.
- `Capacities.payload_width` is a floor, not an exact width, so existing studies that
  asked for a wider payload keep their shapes.
- The fold-once rule keys on `acquisition_id`: a process that writes several rows must
  give each its own id, or give revisions of one acquisition the same id.

## Left for later tasks

- Pre-flight: flag capacities whose registry was never lowered (every column readable),
  and a record kind a channel declares but the registry lacks (no column readable).
- Whether a later revision refits or increments the belief stays with the observer kit.
- Release versions and snapshot fingerprints (the snapshot path now masks columns and
  recomputes `superseded_by`; the fingerprint should hash the masked view).
