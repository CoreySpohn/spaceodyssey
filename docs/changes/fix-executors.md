# Executor corrections

## a2: release identity during replay

Fixed. Trace now records the handled event's log slot, assigned by
transaction.advance. Replay selects that exact row in release order, checks its
validity and metadata, and raises ValueError for an unmatched release. The first
released revision is folded once per acquisition, including revisions released
out of log order at one tick. Held records remain excluded. Regression tests cover
the reordered revisions, inconsistent metadata, and invalid or unused log slots.

## a3: policy coupling

Fixed. Decision keys now pass the study's coupling to the policy key builder.
Tests exercise phase sharing and default phase separation on both executors and
both host policy forms. No default draw changed.

## a4: parity values

Fixed. Integer equality is checked exactly, with Python integer arithmetic for
the difference diagnostic so signed and unsigned 64-bit endpoints cannot overflow.
Different dtypes or shapes fail parity. Floating comparisons require matching NaN
positions, preserve equal infinities, and report infinite error for unmatched
nonfinite values.

## a5: empty step budget

Fixed. The host returns its initialized carry and a correctly typed empty trace
when the step budget is zero. Its batched result matches the compiled executor,
including TRUNCATED status and all array shapes and dtypes.

## a6: canonical identities

Fixed. Canonical integer histories now include handled microsteps, handled stable
event ids, and valid record acquisition ids. Independent corruption of each is
rejected by parity.

## History baselines

No cases were re-baselined. The added fields extend the integer map; existing
hand-written event tuples and row tuples retain their meaning and values.

## Validation

Before fixes: 14 failed, 11 passed in the new value, status, key, and identity
checks. The two new replay checks separately failed as expected.

After fixes: 150 passed across test_parity_values.py, test_run_status.py,
test_executors.py, test_schema.py, test_keys.py, and test_parity_corpus.py.
Formatting, Ruff, domain-word, core-import, and internal-reference checks passed.

Replay regressions before the fix: 2 failed in 4.50s, confirming the wrong
revision fold and the missing error for an unmatched release.

After the replay fix: 169 passed in 651.19s (0:10:51) across
test_replay_log.py, test_parity_corpus.py, test_parity_values.py, test_schema.py,
test_executors.py, test_revisions.py, test_record_statuses.py, and
test_host_service.py. Formatting, Ruff, domain-word, core-import, and
internal-reference checks passed.
