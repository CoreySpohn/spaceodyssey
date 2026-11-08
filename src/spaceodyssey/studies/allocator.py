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

# (tick, program, request, audience): what each program asks for, and who sees it.
# The proposal at 120 reaches the observer only, and the one at 700 comes after the last
# query, so neither may move an award.
DEFAULT_PROPOSALS = (
    (50, 0, 6.0, Audience.POLICY),
    (80, 1, 3.0, Audience.POLICY),
    (120, 0, 40.0, Audience.OBSERVER),
    (300, 1, 5.0, Audience.POLICY),
    (400, 0, 2.0, Audience.POLICY),
    (700, 1, 90.0, Audience.POLICY),
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


class OmniscientCommittee(ToyCommittee):
    """The negative control: it reads every proposal of the script, released or not.

    It takes the proposals from the study's own table instead of the snapshot, so a
    proposal that was never released to the policy, or reached the observer only, moves
    its awards.
    """

    def __init__(self, proposals=DEFAULT_PROPOSALS, kind=AWARD_KIND_INDEX):
        """Hold the script of proposals the study was built from."""
        super().__init__(kind)
        self.proposals = tuple(proposals)

    def requested(self, view):
        """The total request of each program over the whole script, seen or not."""
        return np.asarray(
            [sum(r for _, p, r, _ in self.proposals if p == q) for q in PROGRAMS],
            dtype=np.float64,
        )


class ProposalRecords(scripted.ScriptedRecords):
    """Scripted proposal records, each released to its own audience.

    Fire i writes a proposal of `requests[i]` for program `programs[i]` at `ticks[i]`;
    the record reaches the audience in `audiences[i]` only.
    """

    programs: tuple[int, ...] = eqx.field(default=(), static=True)
    audiences: tuple[int, ...] = eqx.field(default=(), static=True)

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
        return carry, dataclasses.replace(rec, payload=rec.payload.at[1].set(program))


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


def _recorded_proposals(carry, tick):
    """Payloads of the proposal records released to the policy by `tick`."""
    log = carry.log
    keep = (
        np.asarray(log.valid)
        & (np.asarray(log.kind) == PROPOSAL_KIND_INDEX)
        & ((np.asarray(log.audience) & int(Audience.POLICY)) != 0)
        & (np.asarray(log.released_tick) >= 0)
        & (np.asarray(log.released_tick) <= tick)
    )
    pay = np.asarray(log.payload)[keep][:, :2]
    return sorted(map(tuple, pay.tolist()))


def _viewed_proposals(view):
    n = int(view.n_rows)
    keep = np.asarray(view.log.valid)[:n] & (
        np.asarray(view.log.kind)[:n] == PROPOSAL_KIND_INDEX
    )
    pay = np.asarray(view.log.payload)[:n][keep][:, :2]
    return sorted(map(tuple, pay.tolist()))


def replay_allocation(study, run, committee=None):
    """Replay the committee and the policy on the run's own decision views.

    For each executed decision it reruns the policy on the recorded view, with the key
    of that decision, and compares its effects, its draw and its retained `Allocate`
    rows with the run; for each `Query` it reruns the committee on the view that asked
    and compares its answer with the award record in the log, after checking that the
    proposals in the view are the recorded proposals released by then. Each replayed
    `Allocate` is also given the admission code the window rule implies (one open window
    per row) and the ledger's granted totals are checked against the admitted ones.
    Returns a `ReplayReport`; it never raises on a mismatch.
    """
    committee = ToyCommittee() if committee is None else committee
    caps = study.caps
    layout = study.observer.layout(caps)
    (policy,) = study.agenda.policies
    carry, trace = run.carry, run.trace
    steps = np.flatnonzero(np.asarray(trace.handled) & np.asarray(trace.wake))
    fails = []
    if len(steps) != len(run.decisions):
        return ReplayReport(
            (f"{len(run.decisions)} views for {len(steps)} decisions",), 0, 0, 0
        )
    log = carry.log
    recorded = np.flatnonzero(
        np.asarray(log.valid) & (np.asarray(log.kind) == committee.kind)
    )
    pstate = jnp.zeros(caps.n_pstate, F64)
    open_until = {}
    granted = np.zeros((caps.n_assets + caps.n_programs + 1, caps.n_resources))
    answers = n_alloc = 0
    previous, ordinal = None, 0
    for d, (step, (view, recorded_eff)) in enumerate(
        zip(steps, run.decisions, strict=True)
    ):
        tick = int(trace.tick[step])
        ordinal = ordinal + 1 if previous == tick else 0
        previous = tick
        phase = int((tick >= np.asarray(study.agenda.phase_until[:-1])).sum())
        key = policy_key(run.root, phase, tick, ordinal)
        eff, pstate, u = policy.decide(pstate, view, study, key)
        for name in EFFECT_COLUMNS:
            want, have = getattr(eff, name), getattr(recorded_eff, name)
            if not np.array_equal(np.asarray(want), np.asarray(have)):
                fails.append(f"decision {d}: effect column {name} differs")
        if float(u) != float(trace.pol_u[step]):
            fails.append(f"decision {d}: the policy draw differs")
        kinds = np.asarray(recorded_eff.kind)
        # the retained Allocate rows say what the admission saw
        for slot in np.flatnonzero(np.asarray(eff.kind) == EffectKind.ALLOCATE):
            n_alloc += 1
            row = int(eff.recipient[slot])
            start, until = int(eff.window_from[slot]), int(eff.window_until[slot])
            kept = (
                np.asarray(trace.alloc_amount[step, slot]),
                int(trace.alloc_from[step, slot]),
                int(trace.alloc_until[step, slot]),
                int(trace.alloc_recipient[step, slot]),
            )
            if not (
                np.array_equal(kept[0], np.asarray(eff.amount[slot]))
                and kept[1:] == (start, until, row)
            ):
                fails.append(f"decision {d} slot {slot}: the trace row differs")
            opened = open_until.get(row, -1) > tick
            want_code = AdmissionCode.FORBIDDEN if opened else AdmissionCode.OK
            if int(trace.admit_code[step, slot]) != int(want_code):
                fails.append(f"decision {d} slot {slot}: the admission code differs")
            if not opened:
                open_until[row] = until
                granted[row] += np.asarray(eff.amount[slot])
        if (kinds == EffectKind.QUERY).any():
            request = Request(
                tick=tick,
                service=int(
                    recorded_eff.channel_index[np.argmax(kinds == EffectKind.QUERY)]
                ),
                subject=int(recorded_eff.subject[np.argmax(kinds == EffectKind.QUERY)]),
                audience=int(Audience.POLICY),
                snapshot_fp=int(trace.snapshot_fp[step]),
                release_version=int(view.release_version),
                decision=int(view.n_dec),
                group=int(recorded_eff.group[np.argmax(kinds == EffectKind.QUERY)]),
            )
            if _viewed_proposals(view) != _recorded_proposals(carry, tick):
                fails.append(
                    f"decision {d}: the view's proposals are not the released ones"
                )
            payload = np.zeros(caps.payload_width)
            given = np.asarray(committee.answer(request, view, layout), np.float64)
            payload[: len(given)] = given
            if answers >= len(recorded):
                fails.append(f"decision {d}: no recorded award for this query")
            else:
                slot = recorded[answers]
                if not np.array_equal(payload, np.asarray(log.payload)[slot]):
                    fails.append(f"decision {d}: the award row differs")
                if int(log.start_tick[slot]) != tick:
                    fails.append(f"decision {d}: the award row is at another tick")
            answers += 1
    if answers != len(recorded):
        fails.append(f"{len(recorded)} recorded awards for {answers} queries")
    if not np.allclose(np.asarray(carry.ledger.granted), granted):
        fails.append("the ledger's granted totals differ from the admitted awards")
    return ReplayReport(tuple(fails), len(steps), answers, n_alloc)


__all__ = [
    "AWARD_KIND",
    "DEFAULT_PROPOSALS",
    "PROPOSAL_KIND",
    "AllocationPolicy",
    "AllocatorRun",
    "OmniscientCommittee",
    "ProposalRecords",
    "ReplayReport",
    "ToyCommittee",
    "allocator_registry",
    "allocator_study",
    "replay_allocation",
]
