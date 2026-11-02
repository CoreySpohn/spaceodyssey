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

Only `COMPLETE` is a finished run. After an overflow, a stall or an exhausted step
budget, the status says so instead of the run looking finished.

## Replay

`replay_belief(study, carry, trace)` rebuilds the observer belief of one world from its
recorded log: each logged row is folded once, in the order the run released rows,
through the study's observer. The result equals the belief the run carried, bit for bit.

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
