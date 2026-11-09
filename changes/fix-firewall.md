# Audience isolation and host sequencing

## Changes

- b1: fixed. Service requests expose only semantic question coordinates; internal
  stage and microstep stay on the queued event. Private events no longer change
  the request or an answer derived from it.
- b2: fixed. The integer fingerprint chains each array element into its preceding
  hash state and includes the column tag and length. Floats remain excluded.
  Exchanged revision contributions now change the fingerprint, and a pending
  question rejects the altered view before calling its service.
- b3/a1: fixed. Queries reserve the next microstep for their results and defer
  sibling events, including effects listed before the Query. Committing an answer
  moves remaining peers together behind it, preserving their order. Campaign
  drain completion includes queued requests, pending requests and queued results.
  A second Query receives `NO_SLOT` while the first is outstanding. This enforces
  the protocol's single-request capacity and prevents the first answer from making
  an already-admitted second question's snapshot stale.
- b4: fixed. Fold inputs hide queue-derived acquisition identities, and the policy
  fold computes supersession using only policy-released revisions. Raw identities
  remain in the stored log for accounting.
- b5: fixed. Each audience's snapshot carries its own belief.
- b6: fixed. Releases use the intersection of the event and record audiences for
  folds, acceptance/release ticks, supersession and release versions.
- b7: fixed. Only a newly readable policy release raises a policy wake. A held row
  or an audience mismatch leaves the policy wake and release version unchanged.

## History changes

Fingerprint integers change because the digest algorithm changed. Observer snapshot
beliefs now reflect observer folds. Policy-only rows no longer receive an observer
acceptance tick or contribute to its belief. Their policy snapshots and fingerprints
therefore also reflect an acceptance tick of -1. These changes enforce the declared
column and row audiences. No stored hand history was re-baselined.

Mixed Query decisions now handle Request, Result, then their sibling events; those
siblings may have later microsteps. A run awaiting a host answer cannot complete its
campaign drain. Private queue positions are no longer part of the service Request API.

## Verification

Every assigned property was first reproduced by a failing regression in
`tests/core/test_audience_firewall.py` or `tests/core/test_query_ordering.py`.
The initial ordering run failed all 15 cases. After fixing the record fixture's
payload width, all 11 remaining audience assertions failed for their intended
reasons; the private request assertion had already failed before its fix.

The additional single-request capacity regression failed in all three executor
configurations before admission was restricted.

Long transaction runs were interrupted while JAX repeatedly compiled the test
helper's Python step indices as static arguments. A diagnostic stack identified
compilation inside `run_until_quiet`. The three shared step loops now pass the
same indices as JAX scalars, preserving every assertion and step value while
reusing the compiled transition. The digest also uses one scan over tagged fields
instead of a separate scan per column. No test expectation was weakened.

Focused passing results on the final implementation:

- Core handlers and transactions, admission, wake coalescing, run status and stop
  dispositions: `85 passed in 413.07s (0:06:53)`.
- Audience isolation, Query ordering, host service, fingerprint, record kinds,
  statuses, revisions, second kind, schema, leaks, scorer columns and replay:
  `199 passed in 753.52s (0:12:33)`.

Ruff check and format, domain vocabulary, core imports and internal-reference
checks all passed. All changed Python files are ASCII. Tests ran through the
prescribed worktree runner, with one process and only the focused files above.
The complete suite is left to the coordinator.
