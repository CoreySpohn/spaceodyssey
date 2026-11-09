# Slice 1, T7: release versions, snapshot fingerprints, host protocol

## Behaviour changes

- Release versions. `Carry.release_version` (int64[2], observer then policy) counts the
  releases each audience has received. A release adds one to the versions of the
  audiences that can read the row; a held row adds nothing, and a second release of one
  row is not counted twice. `Snapshot.release_version` carries the count of the snapshot's
  audience. `version_slot(audience)` maps an audience to its index.
- Snapshot fingerprints. `snapshot_fingerprint(view)` is an int64 digest of the integer
  content of a masked view (the integer columns of the visible rows, `n_rows`, the belief
  sample counts, `tick`, `n_dec`, `release_version`). It is taken from the view the audience
  is handed, so invisible rows, columns and events cannot move it; floats are excluded so
  executors that differ by rounding still agree. `Trace.snapshot_fp` records it for every
  decision (zero otherwise) and `canonical_history` records it as `snapshot_fp`, with the
  per-step `release_version` and the final `release_version_final`.
- `Query` is live. Admission no longer answers `UNSUPPORTED` for it (`Allocate` still
  does). A permitted Query charges and counts nothing and queues a `REQUEST` row
  (`EventKind.REQUEST`, 9) at the current tick in the admit stage, audience `POLICY`,
  stamped with the decision's snapshot fingerprint, release version and decision index. A
  full queue answers `NO_SLOT` without setting overflow; a phase that does not permit it
  answers `FORBIDDEN`.
- Host protocol (`core/service.py`). The `REQUEST` row is handled by moving into
  `Carry.request`, which holds the run (`active` is false while a request is pending).
  `commit_result` logs the host's answer and queues a `RESULT` row at the request's own tick
  and stage, one microstep later; the `RESULT` handler is the release handler, so the
  record is released at that tick and wakes the policy within the stage.
  `AbstractHostService` (a `kind` and `answer(request, view)`), `Request`, `read_request`
  and `answer_pending` are exported. `answer_pending` rebuilds the audience's snapshot for
  the request's decision and refuses to answer if its fingerprint differs from the
  request's.
- Executors. `HostExecutor.run(..., service=)` answers after the step that reached a
  request. `CompiledExecutor.run(..., service=, chunk=32)` scans in chunks and yields to
  the host between them; a step is taken only while the world has budget and no pending
  request, so the budget is the same `n_steps` the host executor spends. Without a
  service a reached request is never answered and the run reports the new
  `RunStatus.AWAITING`. The scan without a service is unchanged.
- Visibility. `visible_rows` also requires the row's own audience to include the reader,
  so a policy-only answer is absent from an observer snapshot. The observer still accepts
  every released row (`accepted_tick`), as before.
- `replay_belief` counts handled `RESULT` steps as releases.
- `studies/oracle.py`: a toy service (`ToyOracleService`) that answers with the asked
  subject's belief mean, count and the snapshot's release version, an `answer` record kind
  readable by the policy only, and `QueryPolicy`, which asks once and then follows its
  plan.

## Carry changes (shared with the pre-flight task)

- `Carry` gains `release_version` and `request` (a one-row `EventTable`).
- `EventTable` gains `snapshot_fp`, `version` and `decision`; `enqueue` takes them.
- `Trace` gains `snapshot_fp`, `event_fp` and `release_version`.
- `Snapshot` gains `release_version`. `admit` takes an optional `snapshot_fp`;
  `finish_trace` takes an optional `snapshot_fp`.

## Tests

- `test_host_service.py`: `test_query_result_same_tick` (result is the very next row, same
  tick and stage, released at that tick, wakes a decision, both rows carry the asking
  decision's fingerprint), admission of a Query (request row, nothing charged, full queue,
  forbidden phase), `test_host_service_parity` (compiled with chunks of 1, 3 and 32 against
  the host on three worlds: histories, fingerprints and versions equal, replay bitwise),
  truncated hosted runs agree, an unanswered request is reported as `AWAITING` on both
  executors, the answer reaches the policy snapshot and not the observer snapshot, each
  world is answered from its own snapshot, and the core never names the toy service.
- `test_snapshot_fingerprint.py`: `test_snapshot_fingerprint_recorded` (each recorded value
  is the fingerprint of the view that decision was handed, zero on steps with no decision,
  equal on the compiled executor), the fingerprint hashes the masked view and not the raw
  log (an invisible row and every float leave it unchanged while a raw-log digest moves; a
  visible row's identity moves it), it follows the decision index, the tick and the release
  version, and release versions count what each audience can read.
- `test_core.py`: the unsupported-effect test now covers `Allocate` only.
- `test_contracts.py`: `test_study_has_no_untyped_fields` now counts nine abstract bases;
  `AbstractHostService` is a plain host-side class, so its annotations are checked directly.
- `test_schema.py`: the hand-built rows of the compaction test carry a public audience.

## Rebaselined cases

None. No corpus study issues a Query, so every hand history, row table and parity
comparison is unchanged. The canonical history gains four integer entries
(`snapshot_fp`, `event_fp`, `release_version`, `release_version_final`), compared across
executors like the rest.

## Deviations from the spec

- A request row is moved into the carry when handled and the run waits there, instead of
  staying in the queue: this is what lets a compiled scan stop at the request and the
  host answer before any later row runs.
- The fingerprint excludes floating-point content (see above).
- `Query` fields reuse the effect table: `channel_index` names the service and `subject`
  the subject asked about.
- The `Result` record is logged when the host commits it and released when the `RESULT`
  row is handled; it is invisible in between, and a full log sets the overflow flag.
- A `RESULT` row's `channel` is the service id, a namespace separate from channel ids;
  replay matches rows on it, so a service id should not equal a channel id.

## Left for later tasks

- Pre-flight: flag a study whose policy can Query while the run has no service, and a
  service whose record kind the registry lacks.
- Stale-result policy for a service with modelled latency (a result that arrives after a
  newer release): the protocol answers in zero ticks today.
- More than one outstanding request, a bounded buffer of results per request, and a
  host-only updater with a compiled-capability rejection.
