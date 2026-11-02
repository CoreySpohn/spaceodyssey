# Executors

The transaction core runs under two executors that share one transition. Both take a
study, a batch of worlds (truths and root keys) and a step budget, and both return a
`RunResult`: the final carry and a per-step trace, with a leading world axis from
`run_worlds`.

## Tiers

| Executor | Program form | Use |
| --- | --- | --- |
| `HostExecutor` | A Python loop over steps; each stage is its own compiled call. One world per `run`; `run_worlds` loops over worlds. | Reference runs, debugging, Python-form policies, and the recorded log that `replay_belief` reads. |
| `CompiledExecutor` | One compiled `lax.scan` over steps, `vmap`ped over worlds. | Batches of worlds. |

The two produce the same canonical history: integers (ticks, stages, event kinds, lanes,
subjects, acquisition indices, admission codes, ties, pending wakes, statuses, counts)
agree exactly, and floats (payloads, costs, balances, beliefs) agree to 1e-12 in
absolute difference, the band the parity tests require. A float difference above 1e-9
or any integer difference is a failure.

## Run status

A run ends in exactly one `RunStatus`, one per world in a batch:

| Status | Meaning |
| --- | --- |
| `COMPLETE` | The run reached its own end. |
| `IDLE` | The event queue emptied before the study ended. |
| `TRUNCATED` | The step budget ran out first. |
| `STALLED` | A row reached the microstep guard. |
| `OVERFLOW` | A static capacity ran out; `Carry.overflow_mask` holds the `Capacity` bits for what did (`QUEUE_SLOTS`, `STABLE_IDS`, `LOG_ROWS`). |
| `AWAITING` | A request to the host is still unanswered, so nothing after it ran. |

Only `COMPLETE` is a finished run. After an overflow, a stall or an exhausted step
budget, the status says so instead of the run looking finished.

## Host protocol

A policy asks the host a question with a `Query` effect. The effect names a service in
`channel_index` and a subject in `subject`; admission charges nothing and queues a
request row at the current tick in the admit stage, addressed to the policy audience and
stamped with the fingerprint of the snapshot the decision read, that snapshot's release
version and the decision's index.

The protocol admits one outstanding question per world. Additional Queries receive
`AdmissionCode.NO_SLOT` until its Result releases, including a second Query in the
same decision.

When the transaction reaches the request row it moves it into `Carry.request` and stops:
no later row runs until the host answers. The host's answer is a record of the service's
kind, readable by the audience of the snapshot it was asked on. `commit_result` logs the
record and queues a `Result` row at the request's own tick and stage, the next microstep,
so no other work separates them. Sibling effects from that decision run after the
answer, regardless of their order in the effect table. Remaining rows at that tick and
stage move together behind the answer, preserving their relative order. A campaign
drain waits for queued requests, pending requests and queued results. The transaction
handles the `Result` row through the ordinary release rules: the record is released
at that tick and the policy wakes in the
same stage. The request and the `Result` rows both carry the snapshot fingerprint, and
both appear in the history.

A service is an `AbstractHostService`: a `kind` (the record kind index of its answers) and
`answer(request, view)`, which receives the request and the audience's masked snapshot
as numpy arrays and returns the payload columns. The service-facing request carries
the question's semantic coordinates, not its internal stage or microstep, which can
depend on private events. It sees nothing else. Before answering, the executor rebuilds
the snapshot and checks that its fingerprint equals the one on the
request, so a snapshot that changed under a pending question is an error, not a quiet
difference.

Both executors take a `service`:

- `HostExecutor.run(..., service=svc)` answers a pending request after the step that
  reached it.
- `CompiledExecutor.run(..., service=svc, chunk=32)` runs the scan in chunks and yields to
  the host between them. A world that reached a request takes no steps until it is
  answered, and a step it does not take spends none of its budget, so a world takes
  `n_steps` steps in all, as on the host executor, for any `chunk`. The trace has empty
  rows where a world waited. A run without a service that reaches a request reports
  `AWAITING`.

Histories agree across executors for any chunk size, including the request and result
rows and the fingerprints.

## Snapshot fingerprints and release versions

A release version counts the releases an audience has received: `Carry.release_version`
holds one count for the observer and one for the policy, and a snapshot reports its
audience's count. A release moves only the versions of the audiences it reaches, by one,
so a row the policy cannot read leaves the policy's version alone.

Each snapshot carries the belief of its own audience. Each decision records the
fingerprint of the snapshot it read (`Trace.snapshot_fp`, zero where no decision ran;
`snapshot_fp` in the history). The fingerprint is a 64-bit digest of
the integer content of the masked view: every integer column of the visible rows, the row
count, the belief sample counts, the tick, the decision index and the release version.
The digest chains each integer into the previous hash state in column and row order;
exchanging integer contributions cannot preserve it by commutative summation.
Floating-point content is left out on purpose, since it is computed from the prefix the
integers name and executors may differ in it by rounding. Because the digest is taken
from the view the policy was handed, a row, column or event the audience cannot see leaves
it unchanged. A request or result row records the fingerprint it names in `event_fp`.

## Replay

`replay_belief(study, carry, trace)` rebuilds the observer belief of one world from its
recorded log: each logged row is folded once, in the order the run released rows,
through the study's observer. Rows released by a host result count as releases. The
result equals the belief the run carried, bit for bit.

## Handler lowering under `vmap`

The transition picks a handler with `lax.switch` on the event kind. How that lowers
decides what a step costs:

- When the event kind is the same for every world, as for one world, for worlds that
  share an event stream, or for a sequential `lax.map` over worlds, the switch lowers
  to a conditional and only the taken handler runs. Cost is nearly flat in the number
  of handlers.
- When the event kind differs between worlds under `vmap`, JAX turns the switch into a
  select over all branches, so every handler runs on every step and cost grows linearly
  with the number of handlers. In a CPU microbenchmark with equal-sized handlers the
  batched form was about 4.7 times slower at eight handlers than at two, against about
  1.3 times for the conditional form.

The compiled executor is the second case as soon as the worlds differ, because each
world has its own event queue. A step on a restricted path such as a settlement-only
transition, whose event index is shared by all worlds, can still lower to a conditional,
while the full transition over the same worlds lowers to a select. A study with heavy
handlers therefore pays for all of them on every step under the compiled executor.
Skipping handlers needs worlds grouped by event kind or a `lax.map` over worlds.

Timings here are CPU only.
