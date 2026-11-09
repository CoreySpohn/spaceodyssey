# Validation checks

## Conformance

- d1: Fixed. Density checks visit every subject with a positive acquisition count,
  including subjects with no retained records or queued completions.
- d2: Fixed. Conformance runs count committed folds per acquisition for both beliefs.
  The invariant checks these counters directly, including auxiliary records that do
  not change sample counts.
- d3: Fixed. Only the LOG_ROWS capacity bit permits incomplete accounting. Stable-id
  and queue exhaustion retain balance and exact-density checks.
- d4: Fixed. Private-event probes compare public event columns including lane, channel,
  subject, acquisition index, audience, ordering and record metadata. Stable ids are
  ranked among public handled rows because private enqueues legitimately renumber them.
- d5: Fixed. Nonfinite final balances fail, including when log rows overflow; nonfinite
  conservation differences also fail.
- d6: Fixed. Canonical histories include next_sid, and the unaffordable-bundle case
  requires the original value of 3 after refusal.
- d7: Fixed. Canonical histories include record kind. The catalog case requires its
  registered kind and the corresponding contribution in the observer's auxiliary state.

## Battery and declarations

- c1: Fixed. Truth-swap cutoffs compare payloads masked for each reader independently;
  scorer-only differences cannot end the policy comparison.
- c2: Fixed. Policies are wrapped to capture the actual masked views used at every
  decision. The relations compare row counts, visible metadata and content after
  relabeling ids, shifting epoch ticks or delaying release ticks. Relabeling legitimately
  changes keyed draws, so each visible payload is checked against its own masked log
  row, while draw-independent content and belief counts are compared across runs.
  Fingerprints identify captured decisions, but are not compared across relations.
  Callback transport preserves 64-bit words even when a worker uses 32-bit mode;
  missing capture is distinguished from actual view divergence in the tests.
- c3: Fixed. Latency checks compare requested effect tables and admission codes at
  release-woken decisions as well as at other decisions.
- c4: Fixed. Null calibration requires a complete run and at least one accepted
  acquisition folded in every world; a truncated, unexercised check fails.
- c5: Fixed. Fork-prefix records are compared bitwise across every column.
- c6: Fixed. The causality lower bound accounts for cancellation during an occupancy
  or its own lane transition, including the minimum one-tick cancellation release
  delay. An unrelated lane's transition cannot relax that bound.
- c7: Fixed. A declared need must have a decision in the phase that declared it.

## Validation and baseline changes

The continuation inherited the fixes and property-named regression tests together.
Their failure conditions were checked against the original diff without stashing or
reverting the inherited work. The additional causality-bound regression was run
failing before the lane lookup was corrected. Focused validation results follow below.

- Initial targeted regressions: `28 passed in 270.74s (0:04:30)`.
- Additional lane-bound regression before correction: `1 failed in 1.35s`;
  it passes in the subsequent focused runs.
- Battery, pre-flight, executor and parity files: `3 failed, 153 passed in
  734.05s (0:12:14)`. The three positive relation tests exposed callback narrowing
  of 64-bit values. After preserving their words, all affected files (including
  the stronger negative controls and host checks) pass: `28 passed in 307.01s
  (0:05:07)`.
- Conformance invariants, conformance guards and the explicit latency effect/code
  assertions: `182 passed in 701.14s (0:11:41)`.
- All required lint commands passed. Formatting changed no files on the final run;
  all changed Python files are ASCII, and the changed-file internal-reference scan
  passed as well.

The only changes in `core/executors/base.py` are two canonical-history entries:
`kind` from each retained record and the final `next_sid`. They support the two new
case assertions; the parity comparator and the history class are unchanged.

No hand history was re-baselined. The causality failure fixture now declares a decision
at tick 19 instead of tick 50: the channel's 20-tick latency makes the former impossible
even after cancellation, whereas an early partial record can satisfy the latter.
