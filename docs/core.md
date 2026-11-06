# Transaction core

`spaceodyssey.core` is a JAX/Equinox discrete-event simulation core for observing
campaigns. It owns event ordering, admission, resource accounting, record release
and the boundary between simulated truth and a policy's information. A study
supplies its world source, channel model, observer, policies and objective.
Astronomical models and study names live outside the core, in
`spaceodyssey.studies` or application adapters.

Import the core explicitly, for example `from spaceodyssey import core`.
The core requires JAX 64-bit mode before constructing capacities or studies; it
never changes that process setting itself. Shapes are fixed by `Capacities`. Changing those shapes
retraces compiled functions.

## Layers and contracts

| Layer | Responsibility |
| --- | --- |
| Keys | `key_for`, `draw_key` and `policy_key` address randomness by semantic coordinates instead of call order. Measurement identities include subject, channel and acquisition index; policy keys use phase, tick and the decision ordinal within that tick. `Coupling` declares shared coordinates and `Fork` branches acquisitions from a chosen index. |
| Transaction | `advance` handles one queued event; `decide` reads a policy snapshot; `admit` checks effects; `close_decision` commits policy state. `transition` composes these operations and the trace. |
| Records | `RecordKind` and `PayloadColumn` declare payloads and their readers. `KindRegistry.lower` puts those audiences into capacities. `RecordLog` retains provenance, release times and revisions. |
| Executors | `HostExecutor` loops in Python over compiled stages. `CompiledExecutor` scans over steps and batches worlds with `vmap`. Both return `RunResult` and use the same transaction rules. |
| Host protocol | A `Query` queues a request tied to a snapshot fingerprint and release version. A service receives only the masked snapshot and semantic request coordinates; its answer becomes a released record. |
| Pre-flight | `preflight(study, executor=...)` checks declarations, table wiring, information boundaries, causal needs and executor capabilities without running a simulation. |
| Battery | Generated checks exercise null credit, truth swaps, release latency, forks, relabeling, epoch shifts, unused assets and declared behavior on actual runs. |
| Conformance | `run_case` runs a `CorpusCase` on an executor and collects its histories; `check_run` evaluates the generic accounting and ordering invariants on one run. Comparing a run against its hand history, and any case-specific assertion, is a separate explicit step. The concrete toy cases and hand histories live in `studies.conformance`. |

Events are ordered by integer `(tick, stage, micro, tie, sid)`. A channel declares
whether a fault or a coincident settlement wins. Wakes coalesce so that a decision
sees all records released ahead of it at that tick. Observe effects in one group
are admitted together or refused together; refusal consumes no acquisition index.
Admission checks funds, lane availability, the horizon and queue capacity.

A policy implements `AbstractPolicy.decide(pstate, view, study, key)` and returns
an `EffectTable`, its next state and a recorded random draw. Its `Snapshot` contains
the policy's belief, visible records, tick, decision count and release version.
It receives no truth argument. Unreleased rows and unreadable payload columns are
zeroed, and queue-derived acquisition identities are hidden. Observer and policy
beliefs are folded separately from the records each audience receives. Later
revisions are logged and marked as superseding earlier rows, but are not folded
again for the same acquisition. See [Records](records.md) for the full contract.

The host executor supports Python-form policies through `decide_py`. The compiled
executor requires array-compatible decisions, including fixed output shapes and
no Python branches on traced values. `studies.survey_25.NetworkPolicy` demonstrates
an `eqx.nn.MLP` inside the scan: released means and counts, normalized tick and
normalized decision count produce subject/channel logits. The network stays
fixed during the run, and a separate decision limit ends the campaign with a drain.
This example has no training or external agent service.

For host services, both executors pause a world at its request until the answer is
committed. The compiled executor yields between scan chunks. A missing service
leaves the run `AWAITING`; only one request can be outstanding per world. Results
release at the request's tick and stage, at the next microstep, before sibling
effects. See [Executors](executors.md) for this protocol, replay, statuses and
the cost of divergent handler branches under `vmap`.

## Composing a study

1. Choose `Capacities` for subjects, assets, lanes, channels, event slots, log rows,
   effects and policy state. Register record kinds and lower their payload layout.
2. Supply `ChannelTable`, `AssetTable`, initial budgets, a `Clock` and a horizon.
   Channels name occupancy, release latency, costs, audiences and coincidence order.
3. Implement the kind contracts: `AbstractWorldSource`, `AbstractChannelModel`,
   `AbstractProcess`, `AbstractObserver`, `AbstractPolicy`, `AbstractDesign` and
   `AbstractObjective`. An observer keeps two belief types: the full belief
   (`Carry.belief`) and the policy summary (`Carry.pbelief`), the only belief a
   policy reads. It declares both in a `BeliefLayout` (`observer.layout(caps)`):
   their shapes, the integer count leaf the snapshot fingerprint reads, and the
   projection the trace records. The default is `MeanBelief` for both types, and a
   `Study` refuses an observer whose layout does not describe what it initializes.
   Put phase policies, boundaries and permitted effects into an `Agenda`, then
   compose a `Study`. Each phase also carries a level from 0 to 3, whose mask
   (`LEVEL_EFFECTS`) admission intersects with the phase mask; the levels are
   nested, and the default, level 3, permits every effect.
4. Run pre-flight for the intended executor and resolve its findings. Declarations
   are assertions by the components, so a clean result does not prove scientific
   correctness or detect an undeclared dependency hidden in a closure.
5. Provide truths and independent root keys, execute with a step budget, and check
   `RunResult.complete` before interpreting an objective. `OVERFLOW`, `STALLED`,
   `IDLE`, `TRUNCATED` and `AWAITING` are not complete runs.
6. Use the battery checks whose stated assumptions match the study, then compare
   executors through canonical histories. `histories_equal(...).go` requires
   exact integer equality and an absolute float difference at most `1e-12`.

Conformance checks include monotone event ordering, exactly-once settlement, one
fold per acquisition (the committed folds of each audience must equal the
acquisitions it accepted or released, so a duplicate or a missing fold fails; log
replay separately rebuilds the belief and compares it), dense acquisition
indices, conserved balances, release timing, masked
snapshots and private-event wake isolation. These checks protect execution
semantics; each study still needs an independent reference for its objective.
The toy gradient tests separately cover reverse mode on a frozen discrete path,
REINFORCE against enumeration, and a smoothed pathwise gradient against an analytic
reference. They do not make hard argmax decisions differentiable.

## Worked example: a hand-counted search

`survey_1` supplies a fixed five-acquisition plan. This example narrows its tables
to one asset, one lane and one channel, with two subjects. A deterministic channel
returns the known signal and retains `noise=0.5` as the objective's declared
uncertainty scale. This deliberately removes measurement randomness so the count
can be calculated without running either executor.

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.core import Capacities, CompiledExecutor, HostExecutor
from spaceodyssey.core.executors.base import histories_equal, index_world
from spaceodyssey.core.preflight import preflight
from spaceodyssey.studies.toy import ToyPhotometry, ToyTruth, survey_1


class ToyKnownSignal(ToyPhotometry):
    """Known values with a declared uncertainty used only for scoring."""

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        return design.gain * truth.signal[subject], jnp.asarray(0.0, jnp.float64)


caps = Capacities(n_assets=1, n_lanes=1, n_channels=1, n_subjects=2, n_an=2)
study = survey_1(caps=caps, channel_model=ToyKnownSignal(noise=jnp.array([0.5])))
assert not preflight(study, executor="compiled")
truths = ToyTruth(signal=jnp.array([[1.0, 1.0]]))  # leading world axis
roots = jr.PRNGKey(0)[None]

host = HostExecutor().run_worlds(study, truths, roots, n_steps=40)
compiled = CompiledExecutor().run_worlds(study, truths, roots, n_steps=40)
assert bool(host.complete[0]) and bool(compiled.complete[0])
assert histories_equal(host.history(0), compiled.history(0)).go
for result in (host, compiled):
    carry = index_world(result.carry, 0)
    assert int(study.objective.evaluate(carry, study)) == 1
```

The plan observes subjects `(0, 1, 0, 1, 0)` on channel 10. All five acquisitions
fit the budgets: each costs one unit from the asset, program and shared resource
balances, initially 10, 100 and 100. Each release follows 10 transition ticks,
100 occupancy ticks and 20 latency ticks, so the last release is at tick 650,
before the horizon of 1000. Counts are therefore `(3, 2)` and means are `(1, 1)`.
The signal-to-noise ratios are `sqrt(3)/0.5` and `sqrt(2)/0.5`. Only the first
exceeds the objective threshold of 3: **the hand count is one**.

`tests/studies/test_survey_1.py` checks this count and full history parity.
`tests/studies/test_survey_25.py` captures the snapshots actually used by each
executor and compares every effect column with eager evaluations of the same
frozen network.
