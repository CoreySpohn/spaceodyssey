# Conformance invariant evidence

## What changed

- Decision views: each policy, wrapped by the battery's `DecisionViews`, records the
  exact view `decide` receives, on both executors, and the host's Python form records
  the view `decide_py` receives. The runner keeps one view per woken step and the
  hidden-column invariant judges all of them. A missing, short or `None` capture fails.
  The host no longer rebuilds a view for the check.
- Admission counters: the trace's admitted observe rows must sum to the counters, every
  admitted acquisition (named by handled or queued completions, or by a retained
  partial or failed record) must hold an index in `[0, count)` of its subject with no
  repeat, and without log overflow the indices must be exactly `0 .. count - 1`. A zero
  counter beside admitted acquisitions fails.
- Folds: per audience, the committed folds the counting observer recorded must equal,
  key for key, the acquisitions with an accepted (observer) or released (policy) row,
  one fold each.
- Conservation: the final balance, the ledger counters and the retained records plus
  queued settlements plus refused completions must agree, with or without overflow.

## Canonical fields

`Ledger.debited` and `Ledger.refunded` (F64[balance rows, resources]) are new reserved
accounting counters, written beside `balance` in `_commit_observe` and `_cancel_lane`
and read by no rule. They are not part of the canonical history, so hand event and row
histories are unchanged; parity is held because both executors run the same
transaction code and a test compares the counters across executors.

## Residual limits

- A cancellation whose record the log refused cannot be priced from evidence that
  survives; such a run fails the residual check instead of passing.
- The counters share their update with the balance, so a mutant that edits balance and
  counters alike is caught only by the retained-record and trace accounts.
- A forged queue or trace that agrees with a forged counter is out of reach.
