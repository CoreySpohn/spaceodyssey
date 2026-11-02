# Slice 1, T3: keys

## Behaviour changes

- Policy draws are keyed by meaning: `policy_key(root, phase, tick, ordinal, lineage=...)`
  folds the purpose, the phase, the lineage, the decision tick and the ordinal of the
  decision within its tick. The old key was the decision count.
- `Carry` gains `dec_tick` (tick of the last decision, -1 at start) and `dec_ordinal`
  (decisions already made at that tick). `close_decision` advances them; `decision_key`
  reads them. Both executors use `decision_key`.
- `Coupling` declares, per purpose, the coordinates whose changes leave that purpose's draws
  alone (folded as zero). The default shares nothing.
- `asset` and `branch` join the key coordinates (nine coordinates plus `revision` and
  `status`). Both are folded only when nonzero, so a key that does not use them is bit for
  bit the key built before.

## Measurement draws

Unchanged: every measurement variate equals its earlier value (the corpus draw tests pass
unmodified).

## Rebaselined cases

Policy draws changed by design. No hand history, parity comparison or corpus case needed
rebaselining: the corpus histories do not record the policy variate and no policy in the
corpus branches on it, so the 22 cases agree across executors and policy forms with their
existing hand histories.

Test expectations that changed:

| Test | Old | New | Reason |
|---|---|---|---|
| `test_policy_draws_shift_with_inserted_wait` | policy draws changed (> 0) by release identity | removed; replaced by `test_inserted_wait_changes_no_policy_draw` (0 changed) | draws are addressed by decision tick and ordinal |
| `policy_draws_changed` (helper) | compared by the release that woke a decision | compares by (decision tick, ordinal within tick) | matches the new key |
