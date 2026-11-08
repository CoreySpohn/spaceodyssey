"""A toy allocation committee, the level-3 policy that spends its awards, and a replay.

The committee is a host service. A level-3 array policy asks it a `Query` at fixed
ticks; the service reads the masked snapshot of that decision, splits a pool between two
programs in proportion to the proposals released to the policy so far, and answers with
an award record. The policy reads the award record and emits one `Allocate` effect per
funded program (amount, window, recipient row, program), then observes with the money.

`replay_allocation` reruns the committee and the policy on the masked views the run
actually showed the policy, with their semantic keys, and compares the award rows, the
`Allocate` effects and their admission outcomes with the recorded run. A record dropped
from the run, or a proposal the committee should not have read, makes it fail.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import EventKind, Stage, enqueue
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    F64,
    I64,
    AdmissionCode,
    Audience,
    Disposition,
    EffectKind,
    RecordStatus,
    RemainderRule,
    Scope,
)
from spaceodyssey.core.keys import policy_key
from spaceodyssey.core.kinds import AbstractPolicy
from spaceodyssey.core.service import AbstractHostService, Request
from spaceodyssey.core.snapshot import effects_none
from spaceodyssey.studies import scripted, toy

PROPOSAL_KIND = toy.RecordKind(
    "proposal",
    (
        toy.PayloadColumn("request", Audience.POLICY),
        toy.PayloadColumn("program", Audience.POLICY),
    ),
)
AWARD_KIND = toy.RecordKind(
    "award",
    (
        toy.PayloadColumn("round", Audience.POLICY),
        toy.PayloadColumn("window_from", Audience.POLICY),
        toy.PayloadColumn("window_until", Audience.POLICY),
        toy.PayloadColumn("amount_0", Audience.POLICY),
        toy.PayloadColumn("amount_1", Audience.POLICY),
    ),
)
# The toy registry holds three kinds, so these take the next two indices.
PROPOSAL_KIND_INDEX, AWARD_KIND_INDEX = 3, 4

PROGRAMS = (0, 1)
PROGRAM_COLUMN = 1  # the resource whose scope is the program: awards fund this one
QUERY_TICKS = (200, 500)  # when the policy asks the committee
FINISH_TICK = 750  # when the policy drains the campaign, after the last award
POOL = 12.0  # what the committee splits at each query
QUANTUM = 0.25  # awards are whole multiples of this
WINDOW = 250  # ticks an award stays open
N_EFFECTS = 5  # two awards, two observations and a wait
SERVICE = 0  # the channel index a `Query` names
OBSERVE_CHANNELS = (10, 11)  # the channel ids the two programs observe on

# (write tick, program, request, audience, release tick): what each program asks for,
# and who sees it, when. The proposal at 120 reaches the observer only; the one written
# at 150 sits in the log unreleased until 700, after both queries. Neither may move an
# award.
DEFAULT_PROPOSALS = (
    (50, 0, 6.0, Audience.POLICY, 50),
    (80, 1, 3.0, Audience.POLICY, 80),
    (120, 0, 40.0, Audience.OBSERVER, 120),
    (150, 1, 90.0, Audience.POLICY, 700),
    (300, 1, 5.0, Audience.POLICY, 300),
    (400, 0, 2.0, Audience.POLICY, 400),
)


def allocator_registry():
    """The toy registry with the proposal and award kinds added."""
    registry = toy.toy_registry()
    if registry.register(PROPOSAL_KIND) != PROPOSAL_KIND_INDEX:
        raise AssertionError("the proposal kind moved")
    if registry.register(AWARD_KIND) != AWARD_KIND_INDEX:
        raise AssertionError("the award kind moved")
    return registry


def split_pool(requests):
    """The pool divided in proportion to `requests`, in whole quanta rounded down."""
    requests = np.asarray(requests, dtype=np.float64)
    total = float(requests.sum())
    if total <= 0.0:
        return np.zeros_like(requests)
    return np.floor(POOL * requests / total / QUANTUM) * QUANTUM


class ToyCommittee(AbstractHostService):
    """Award the pool in proportion to the proposals the snapshot shows.

    It reads the policy's masked snapshot and nothing else: the proposal rows in the
    log, and the count of earlier award rows for the round number. The answer is
    `(round, window from, window until, amount per program)`.
    """

    def __init__(self, kind=AWARD_KIND_INDEX):
        """Answer with records of registry index `kind`."""
        self.kind = kind
        self.requests = []

    def requested(self, view):
        """The total request of each program among the proposals in `view`."""
        log, n = view.log, int(view.n_rows)
        kind = np.asarray(log.kind)[:n]
        pay = np.asarray(log.payload)[:n]
        seen = np.asarray(log.valid)[:n] & (kind == PROPOSAL_KIND_INDEX)
        return np.asarray(
            [pay[seen & (pay[:, 1] == p), 0].sum() for p in PROGRAMS], dtype=np.float64
        )

    def answer(self, request, view, layout):
        """Return the round's award row for the snapshot the committee was asked on."""
        self.requests.append(request)
        n = int(view.n_rows)
        kind = np.asarray(view.log.kind)[:n]
        done = int(np.sum(np.asarray(view.log.valid)[:n] & (kind == self.kind)))
        tick = int(view.tick)
        amounts = split_pool(self.requested(view))
        return np.asarray([done, tick, tick + WINDOW, *amounts], dtype=np.float64)


class CarryCommittee(ToyCommittee):
    """The negative control: it reads the carry's log, released or not.

    The service is handed only the masked snapshot. This committee also holds the carry
    (see `hold`) and totals every proposal row in its log written by the query tick,
    whatever the row's audience and release tick. A proposal that is still unreleased
    to the policy, or that reached the observer only, moves its awards.
    """

    def __init__(self, kind=AWARD_KIND_INDEX):
        """Start without a carry; call `hold` before each answer."""
        super().__init__(kind)
        self.carry = None

    def hold(self, carry):
        """Keep the carry whose log the next answers read."""
        self.carry = carry

    def requested(self, view):
        """The total request of each program among the proposals in the log."""
        log = self.carry.log
        pay = np.asarray(log.payload)
        seen = (
            np.asarray(log.valid)
            & (np.asarray(log.kind) == PROPOSAL_KIND_INDEX)
            & (np.asarray(log.start_tick) <= int(view.tick))
        )
        return np.asarray(
            [pay[seen & (pay[:, 1] == p), 0].sum() for p in PROGRAMS], dtype=np.float64
        )


class ProposalRecords(scripted.ScriptedRecords):
    """Scripted proposal records, each released to its own audience.

    Fire i writes a proposal of `values[i]` for program `programs[i]` at `ticks[i]`; the
    record reaches the audience in `audiences[i]` only, at `releases[i]`. Until then it
    is in the log and in no snapshot.
    """

    programs: tuple[int, ...] = eqx.field(default=(), static=True)
    audiences: tuple[int, ...] = eqx.field(default=(), static=True)
    releases: tuple[int, ...] = eqx.field(default=(), static=True)

    def seed(self, carry, caps):
        """Queue one process event per proposal, with its own audience."""
        for i, tick in enumerate(self.ticks):
            carry = enqueue(
                carry,
                True,
                kind=EventKind.PROCESS,
                tick=tick,
                stage=Stage.RELEASE,
                group=i,
                audience=self.audiences[i],
                caps=caps,
            )
        return carry

    def handle(self, carry, ev, study):
        """Write the proposal of this fire: its request and its program."""
        carry, rec = super().handle(carry, ev, study)
        program = jnp.asarray(self.programs, F64)[ev.group]
        release = jnp.asarray(self.releases, I64)[ev.group]
        return carry, dataclasses.replace(
            rec, payload=rec.payload.at[1].set(program), available_tick=release
        )


def _proposal_process(proposals):
    n = len(proposals)
    return ProposalRecords(
        ticks=tuple(p[0] for p in proposals),
        values=tuple(float(p[2]) for p in proposals),
        ids=tuple(range(n)),
        revisions=(0,) * n,
        statuses=(int(RecordStatus.COMPLETED),) * n,
        kinds=(PROPOSAL_KIND_INDEX,) * n,
        programs=tuple(int(p[1]) for p in proposals),
        audiences=tuple(int(p[3]) for p in proposals),
        releases=tuple(int(p[4]) for p in proposals),
    )


class AllocationPolicy(AbstractPolicy):
    """A level-3 array policy: ask at fixed ticks, turn each award into `Allocate`.

    `pstate[0]` counts the questions asked and `pstate[1]` the awards spent. At each
    query tick it emits a `Query`. When the award record of the round it asked is in its
    view it emits an `Allocate` for each program with a positive amount, an `Observe`
    on that program's channel (paid from the award), and a `Wait` to the next query or
    to the end; after the last award it drains the campaign.
    """

    def declare(self):
        """The programs, effects and the service channel this policy uses."""
        return Declaration(
            programs=PROGRAMS,
            effects=frozenset(
                {EffectKind.ALLOCATE, EffectKind.QUERY, EffectKind.OBSERVE}
            ),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
            channels=(SERVICE,),
        )

    def decide(self, pstate, view, study, key):
        """Emit the effects of this decision from the masked view alone."""
        caps = study.caps
        n_q = len(QUERY_TICKS)
        qt = jnp.asarray((*QUERY_TICKS, FINISH_TICK), I64)
        asked, done = pstate[0].astype(I64), pstate[1].astype(I64)
        log = view.log
        seen = (jnp.arange(log.kind.shape[0]) < view.n_rows) & log.valid
        mine = seen & (log.kind == AWARD_KIND_INDEX) & (log.payload[:, 0] == done)
        spend = mine.any() & (asked > done)
        award = log.payload[jnp.argmax(mine)]
        ask = ~spend & (asked == done) & (asked < n_q) & (view.tick >= qt[asked % n_q])
        final = ~spend & ~ask & (done == n_q) & (view.tick >= FINISH_TICK)
        first = ~spend & ~ask & ~final & (view.n_dec == 0)
        amounts = award[3 : 3 + len(PROGRAMS)]
        e = N_EFFECTS
        none = jnp.full(e, EffectKind.NONE, I64)
        z = jnp.zeros(e, I64)
        funded = [spend & (amounts[p] > 0.0) for p in PROGRAMS]
        kind = (
            none.at[0]
            .set(
                jnp.select(
                    [funded[0], ask, final, first],
                    [
                        EffectKind.ALLOCATE,
                        EffectKind.QUERY,
                        EffectKind.STOP,
                        EffectKind.WAIT,
                    ],
                    EffectKind.NONE,
                )
            )
            .at[1]
            .set(jnp.where(funded[1], EffectKind.ALLOCATE, EffectKind.NONE))
            .at[2]
            .set(jnp.where(spend, EffectKind.OBSERVE, EffectKind.NONE))
            .at[3]
            .set(jnp.where(spend, EffectKind.OBSERVE, EffectKind.NONE))
            .at[4]
            .set(jnp.where(spend, EffectKind.WAIT, EffectKind.NONE))
        )
        channel = z.at[2].set(_channel(study, 0)).at[3].set(_channel(study, 1))
        rows = caps.n_assets + jnp.asarray(PROGRAMS, I64)
        amount = jnp.zeros((e, caps.n_resources), F64)
        for slot, p in enumerate(PROGRAMS):
            amount = amount.at[slot, PROGRAM_COLUMN].set(amounts[p])
        wake = qt[jnp.minimum(done + 1, n_q)]
        eff = dataclasses.replace(
            effects_none(caps),
            kind=kind,
            subject=z.at[3].set(1),
            channel_index=channel.at[0].set(jnp.where(ask, SERVICE, 0)),
            group=view.n_dec * e + jnp.arange(e, dtype=I64),
            tick=z.at[0].set(jnp.where(first, qt[0], 0)).at[4].set(wake),
            scope=z.at[0].set(jnp.where(final, Scope.CAMPAIGN, 0)),
            disposition=z.at[0].set(jnp.where(final, Disposition.DRAIN, 0)),
            program=z.at[1].set(1).at[3].set(1),
            amount=amount,
            window_from=jnp.where(kind == EffectKind.ALLOCATE, award[1], 0).astype(I64),
            window_until=jnp.where(kind == EffectKind.ALLOCATE, award[2], 0).astype(
                I64
            ),
            recipient=z.at[0].set(rows[0]).at[1].set(rows[1]),
            remainder=jnp.where(
                kind == EffectKind.ALLOCATE, int(RemainderRule.FORFEIT), 0
            ).astype(I64),
        )
        new = jnp.zeros_like(pstate).at[0].set(asked + ask).at[1].set(done + spend)
        return eff, new, jax.random.uniform(key, dtype=F64)


def _channel(study, i):
    return jnp.argmax(study.channels.id == OBSERVE_CHANNELS[i]).astype(I64)


def allocator_study(proposals=DEFAULT_PROPOSALS, **overrides):
    """The two-program toy study with awards on, the proposals and the policy.

    The program rows hold no credit of the program resource, so the awards pay for
    every observation. Returns the study; answer it with `ToyCommittee()`.
    """
    caps = allocator_registry().lower(
        overrides.pop(
            "caps", Capacities(n_programs=2, n_effects=N_EFFECTS, awards=True)
        )
    )
    study = toy.survey_1(
        caps=caps,
        policies=(AllocationPolicy(),),
        process=_proposal_process(proposals),
        **overrides,
    )
    budget = study.budget
    for p in PROGRAMS:
        budget = budget.at[caps.n_assets + p, PROGRAM_COLUMN].set(0.0)
    return dataclasses.replace(study, budget=budget)


@dataclasses.dataclass(frozen=True)
class AllocatorRun:
    """One world of a run, as `replay_allocation` reads it.

    `decisions` is the `(view, effects)` of every executed decision, as the policy saw
    and answered them; `carry` and `trace` are the world's final state and trace.
    """

    carry: object
    trace: object
    decisions: tuple
    root: object


@dataclasses.dataclass(frozen=True)
class ReplayReport:
    """What a replay compared and what differed."""

    failures: tuple
    n_decisions: int
    n_awards: int
    n_allocations: int

    @property
    def ok(self):
        """Whether the replay matched the run everywhere."""
        return not self.failures


EFFECT_COLUMNS = (
    "kind",
    "subject",
    "channel_index",
    "group",
    "tick",
    "scope",
    "disposition",
    "program",
    "amount",
    "window_from",
    "window_until",
    "recipient",
    "remainder",
    "realloc_to",
)


def release_steps(trace, n_log):
    """The executor step at which each log row was released to the policy, else -1.

    A release or a result row handled with the policy audience names the log slot it
    releases; the first such step wins. A step's decision sees what the step released.
    """
    out = np.full(n_log, -1, dtype=np.int64)
    kind, audience = np.asarray(trace.kind), np.asarray(trace.audience)
    slot = np.asarray(trace.log_slot)
    released = (
        np.asarray(trace.handled)
        & ((kind == EventKind.RELEASE) | (kind == EventKind.RESULT))
        & ((audience & int(Audience.POLICY)) != 0)
        & (slot >= 0)
        & (slot < n_log)
    )
    for step in np.flatnonzero(released):
        if out[slot[step]] < 0:
            out[slot[step]] = step
    return out


def released_rows(log, kind, step, tick, width, released_step):
    """Sorted payloads (first `width` columns) of the `kind` rows released by `step`.

    A row counts when it is valid, addressed to the policy, stamped released by `tick`,
    and released at a step no later than `step` (`released_step`, per log row).
    """
    audience = np.asarray(log.audience)
    stamp = np.asarray(log.released_tick)
    at = np.asarray(released_step)
    keep = (
        np.asarray(log.valid)
        & (np.asarray(log.kind) == kind)
        & ((audience & int(Audience.POLICY)) != 0)
        & (stamp >= 0)
        & (stamp <= tick)
        & (at >= 0)
        & (at <= step)
    )
    return sorted(map(tuple, np.asarray(log.payload)[keep][:, :width].tolist()))


def viewed_rows(view, kind, width):
    """Sorted payloads (first `width` columns) of the `kind` rows a view shows."""
    n = int(view.n_rows)
    keep = np.asarray(view.log.valid)[:n] & (np.asarray(view.log.kind)[:n] == kind)
    pay = np.asarray(view.log.payload)[:n][keep][:, :width]
    return sorted(map(tuple, pay.tolist()))


ROW_KINDS = ((PROPOSAL_KIND_INDEX, 2, "proposals"), (AWARD_KIND_INDEX, 5, "awards"))


def visibility_failures(view, log, step, tick, released_step):
    """How a decision view differs from the records released to the policy by `step`.

    The proposals and the awards in the view must be exactly those the log shows as
    released to the policy audience by that executor step (see `released_rows`).
    """
    return [
        f"the view's {name} are not the released ones"
        for kind, width, name in ROW_KINDS
        if viewed_rows(view, kind, width)
        != released_rows(log, kind, step, tick, width, released_step)
    ]


def award_failures(committee, asked, recorded, layout=None):
    """How the committee's answers differ from the recorded award rows.

    `asked` is `(request, view)` for each query in order; `recorded` is
    `(payload row, start tick)` for each award record in order.
    """
    fails = []
    for k, (request, view) in enumerate(asked):
        if k >= len(recorded):
            fails.append(f"query {k}: no recorded award")
            continue
        row, start = recorded[k]
        given = np.asarray(committee.answer(request, view, layout), np.float64)
        payload = np.zeros(len(row))
        payload[: len(given)] = given
        if not np.array_equal(payload, np.asarray(row)):
            fails.append(f"query {k}: the award row differs")
        if int(start) != int(request.tick):
            fails.append(f"query {k}: the award row is at another tick")
    if len(recorded) > len(asked):
        fails.append(f"{len(recorded)} recorded awards for {len(asked)} queries")
    return fails


def allocation_failures(n_assets, n_resources, awards, allocates):
    """How the recorded `Allocate` effects differ from what the awards call for.

    `awards` are the award payload rows in order. Each funded program of each award must
    be allocated exactly: its amount in the program resource, the award's two window
    bounds, the program's balance row, the program and the `FORFEIT` rule. `allocates`
    is `(amount, from, until, recipient, program, remainder)` for each recorded
    `Allocate` effect, in order.
    """
    want = []
    for row in awards:
        for p in PROGRAMS:
            if row[3 + p] > 0.0:
                amount = np.zeros(n_resources)
                amount[PROGRAM_COLUMN] = row[3 + p]
                want.append(
                    (
                        amount.tolist(),
                        int(row[1]),
                        int(row[2]),
                        n_assets + p,
                        p,
                        int(RemainderRule.FORFEIT),
                    )
                )
    have = [
        (np.asarray(a).tolist(), int(f), int(u), int(r), int(p), int(m))
        for a, f, u, r, p, m in allocates
    ]
    if len(have) != len(want):
        return [f"{len(have)} Allocate effects for {len(want)} funded awards"]
    return [
        f"Allocate {i} does not follow its award"
        for i, (h, w) in enumerate(zip(have, want, strict=True))
        if h != w
    ]


def replay_allocation(study, run, committee=None):
    """Replay the committee and the policy on the run's own decision views.

    For each executed decision it checks that the proposals and awards in the view are
    the records released to the policy by then, reruns the policy on the view with the
    key of that decision, and compares its effects, its draw and its retained `Allocate`
    rows with the run. Each replayed `Allocate` is given the admission code the window
    rule implies (one open window per row) and compared with the trace. For each `Query`
    it reruns the committee on the view that asked. After the walk it compares those
    answers with the award records in the log, checks that each recorded `Allocate`
    follows its award, and checks the ledger's granted totals against the admitted
    awards. Returns a `ReplayReport`; it never raises on a mismatch.
    """
    committee = ToyCommittee() if committee is None else committee
    if hasattr(committee, "hold"):
        committee.hold(run.carry)
    caps = study.caps
    layout = study.observer.layout(caps)
    (policy,) = study.agenda.policies
    carry, trace = run.carry, run.trace
    steps = np.flatnonzero(np.asarray(trace.handled) & np.asarray(trace.wake))
    if len(steps) != len(run.decisions):
        message = f"{len(run.decisions)} views for {len(steps)} decisions"
        return ReplayReport((message,), 0, 0, 0)
    log = carry.log
    rows = np.flatnonzero(
        np.asarray(log.valid) & (np.asarray(log.kind) == committee.kind)
    )
    recorded = [(np.asarray(log.payload)[r], int(log.start_tick[r])) for r in rows]
    released = release_steps(trace, len(np.asarray(log.kind)))
    fails, asked, allocates = [], [], []
    pstate = jnp.zeros(caps.n_pstate, F64)
    open_until = {}
    granted = np.zeros((caps.n_assets + caps.n_programs + 1, caps.n_resources))
    previous, ordinal = None, 0
    for d, (step, (view, taken)) in enumerate(zip(steps, run.decisions, strict=True)):
        tick = int(trace.tick[step])
        ordinal = ordinal + 1 if previous == tick else 0
        previous = tick
        fails += [
            f"decision {d}: {m}"
            for m in visibility_failures(view, carry.log, step, tick, released)
        ]
        phase = int((tick >= np.asarray(study.agenda.phase_until[:-1])).sum())
        key = policy_key(run.root, phase, tick, ordinal)
        eff, pstate, u = policy.decide(pstate, view, study, key)
        for name in EFFECT_COLUMNS:
            if not np.array_equal(
                np.asarray(getattr(eff, name)), np.asarray(getattr(taken, name))
            ):
                fails.append(f"decision {d}: effect column {name} differs")
        if float(u) != float(trace.pol_u[step]):
            fails.append(f"decision {d}: the policy draw differs")
        kinds = np.asarray(taken.kind)
        for slot in np.flatnonzero(kinds == EffectKind.ALLOCATE):
            allocates.append(
                (
                    np.asarray(taken.amount[slot]),
                    taken.window_from[slot],
                    taken.window_until[slot],
                    taken.recipient[slot],
                    taken.program[slot],
                    taken.remainder[slot],
                )
            )
        for slot in np.flatnonzero(np.asarray(eff.kind) == EffectKind.ALLOCATE):
            row = int(eff.recipient[slot])
            start, until = int(eff.window_from[slot]), int(eff.window_until[slot])
            kept = (
                int(trace.alloc_from[step, slot]),
                int(trace.alloc_until[step, slot]),
                int(trace.alloc_recipient[step, slot]),
            )
            if not (
                np.array_equal(
                    np.asarray(trace.alloc_amount[step, slot]),
                    np.asarray(eff.amount[slot]),
                )
                and kept == (start, until, row)
            ):
                fails.append(f"decision {d} slot {slot}: the trace row differs")
            opened = open_until.get(row, -1) > tick
            want = AdmissionCode.FORBIDDEN if opened else AdmissionCode.OK
            if int(trace.admit_code[step, slot]) != int(want):
                fails.append(f"decision {d} slot {slot}: the admission code differs")
            if not opened:
                open_until[row] = until
                granted[row] += np.asarray(eff.amount[slot])
        if (kinds == EffectKind.QUERY).any():
            q = int(np.argmax(kinds == EffectKind.QUERY))
            request = Request(
                tick=tick,
                service=int(taken.channel_index[q]),
                subject=int(taken.subject[q]),
                audience=int(Audience.POLICY),
                snapshot_fp=int(trace.snapshot_fp[step]),
                release_version=int(view.release_version),
                decision=int(view.n_dec),
                group=int(taken.group[q]),
            )
            asked.append((request, view))
    fails += award_failures(committee, asked, recorded, layout)
    fails += allocation_failures(
        caps.n_assets, caps.n_resources, [r for r, _ in recorded], allocates
    )
    if not np.allclose(np.asarray(carry.ledger.granted), granted):
        fails.append("the ledger's granted totals differ from the admitted awards")
    return ReplayReport(tuple(fails), len(steps), len(asked), len(allocates))


__all__ = [
    "AWARD_KIND",
    "DEFAULT_PROPOSALS",
    "PROPOSAL_KIND",
    "AllocationPolicy",
    "AllocatorRun",
    "CarryCommittee",
    "ProposalRecords",
    "ReplayReport",
    "ToyCommittee",
    "allocation_failures",
    "allocator_registry",
    "allocator_study",
    "award_failures",
    "release_steps",
    "released_rows",
    "replay_allocation",
    "viewed_rows",
    "visibility_failures",
]
