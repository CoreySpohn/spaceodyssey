"""A conformance case and the machinery that runs one on either executor.

A `CorpusCase` is a named study with an optional hand-written history. `run_case`
runs it on one executor and returns one `CaseRun` per world: the final carry, the
trace, the canonical history, the decision views the host executor saw, and the wake
counts of the same study with private events added. The invariants in
`spaceodyssey.core.conformance.invariants` read a `CaseRun`.

The cases themselves are studies, so they are built outside the core (see
`spaceodyssey.studies.conformance`).
"""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core.battery.views import DecisionViews
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.conformance.folds import CountingObserver, fold_counts, plain
from spaceodyssey.core.enums import Audience, EventKind, Purpose, Stage
from spaceodyssey.core.events import enqueue, where_tree
from spaceodyssey.core.executors.base import canonical_history, index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.kinds import AbstractProcess
from spaceodyssey.core.study import Study
from spaceodyssey.core.transaction import wake_by_audience

N_STEPS = 40
PROBE_SCOPE = 99
EXECUTORS = ("host", "compiled")


@dataclasses.dataclass(frozen=True)
class CorpusCase:
    """One study of the corpus.

    `make_study` builds it. `hand_history` pins its handled rows as (tick, stage, kind)
    in handled order, and `hand_rows` the per-row (lane, subject, acquisition_index,
    n_written, wake), when given. `invariants` are checks on the canonical history
    that belong to this case alone. A study that asks the host a question sets
    `make_service`, which builds a fresh answerer for each run; a study that
    checkpoints sets `make_refit`, which builds the refit boundary the same way.
    """

    name: str
    make_study: Callable[[], Study]
    n_seeds: int
    hand_history: tuple | None
    invariants: tuple
    policy_forms: tuple = ("array", "python")
    hand_rows: tuple | None = None
    make_service: Callable | None = None
    make_refit: Callable | None = None


class PrivateProbe(AbstractProcess):
    """Wrap a process and add private events that write nothing.

    The extra events fire every `period_ticks` for `n_fires` fires, in the release
    stage, for the observer only. A policy that wakes only on what its audience can
    see decides exactly as often with the probe as without it.
    """

    inner: AbstractProcess
    period_ticks: int = eqx.field(default=37, static=True)
    n_fires: int = eqx.field(default=10, static=True)

    def seed(self, carry, caps):
        """Seed the wrapped process, then the private events."""
        carry = self.inner.seed(carry, caps)
        for i in range(self.n_fires):
            carry = enqueue(
                carry,
                True,
                kind=EventKind.PROCESS,
                tick=self.period_ticks * (i + 1),
                stage=Stage.RELEASE,
                group=i,
                audience=Audience.OBSERVER,
                scope=PROBE_SCOPE,
                caps=caps,
            )
        return carry

    def handle(self, carry, ev, study):
        """Hand the event to the wrapped process unless it is a probe event."""
        probe = ev.scope == PROBE_SCOPE
        c1, rec = self.inner.handle(carry, ev, study)
        c1 = where_tree(probe, carry, c1)
        return c1, dataclasses.replace(rec, valid=rec.valid & ~probe)


@dataclasses.dataclass(frozen=True)
class CaseRun:
    """One world of one case on one executor.

    `probe` is (wakes, decisions, events) of the same world with private events added,
    or None when it was not run; `events` are the handled public event rows
    other than the observer-only process events (see `public_events`). `views` are the
    exact views `policy.decide` received, one per woken step in handled order, captured
    inside the policy on either executor; an entry is None when none was captured for
    that step. `form_views` maps each other host policy form of the case (`python`) to
    the views `decide_py` received. `reference_caps` is the layout that judges what a
    snapshot may show; it defaults to the study's own.
    """

    case: CorpusCase
    executor: str
    world: int
    study: Study
    carry: object
    trace: object
    history: object
    views: tuple = ()
    probe: tuple | None = None
    reference_caps: object = None
    folds: tuple = ()
    form_views: dict = dataclasses.field(default_factory=dict)

    @property
    def name(self):
        """The case name."""
        return self.case.name

    @property
    def wakes(self):
        """Handled steps that woke a decision."""
        return wake_count(self.trace)

    @property
    def decisions(self):
        """Decisions made."""
        return int(self.carry.n_dec)


def public_events(trace):
    """Public event columns, with stable ids ranked among handled public rows."""
    handled = np.asarray(trace.handled)
    kind = np.asarray(trace.kind)
    private = (kind == EventKind.PROCESS) & (
        np.asarray(trace.audience) == Audience.OBSERVER
    )
    keep = handled & ~private
    # Stable ids count private enqueues; compare their public relative order instead.
    # A coalesced wake can move onto a private row, so it is checked by count above.
    names = (
        "tick",
        "stage",
        "micro",
        "tie",
        "kind",
        "lane",
        "channel",
        "subject",
        "acquisition_index",
        "audience",
        "n_written",
        "rec_status",
        "event_fp",
    )
    ids = np.asarray(trace.sid)[keep]
    ranks = np.searchsorted(np.unique(ids), ids).tolist()
    return tuple(
        zip(
            *(np.asarray(getattr(trace, name))[keep].tolist() for name in names),
            ranks,
            strict=True,
        )
    )


def wake_count(trace):
    """Number of handled steps that woke the policy."""
    handled = np.asarray(trace.handled)
    return int((np.asarray(trace.wake) & handled).sum())


def make_worlds(study, n_worlds, root0=None):
    """Draw `n_worlds` truths and roots for a study from its own world source."""
    root0 = jr.PRNGKey(0) if root0 is None else root0
    roots = jax.vmap(lambda i: jr.fold_in(root0, i))(jnp.arange(n_worlds))
    truths = jax.vmap(
        lambda r: study.world_source.draw(jr.fold_in(r, Purpose.WORLD), study.caps)
    )(roots)
    return truths, roots


def _service(case):
    return None if case.make_service is None else case.make_service()


def _refit(case):
    return None if case.make_refit is None else case.make_refit()


def _execute(
    case, study, executor, truths, roots, n_steps, wake_fn, want_views, form="array"
):
    """Run every world; returns [(carry, trace, views)] for the executor named.

    With `want_views` each policy records the view it is handed, and `views` holds, per
    woken step, that view (None where nothing was recorded). `form` selects the host
    policy form; the compiled executor has only the array form.
    """
    n_worlds = roots.shape[0]
    capture = DecisionViews()
    run_study = capture.instrument(study) if want_views else study

    def views_of(w, trace):
        if not want_views:
            return ()
        got = capture.read(study, roots[w], trace)
        return tuple(None if g is None else g[0] for g in got)

    if executor == "compiled":
        run = CompiledExecutor().run(
            run_study,
            truths,
            roots,
            n_steps,
            wake_fn=wake_fn,
            service=_service(case),
            refit=_refit(case),
        )
        out = []
        for w in range(n_worlds):
            trace = index_world(run.trace, w)
            out.append((index_world(run.carry, w), trace, views_of(w, trace)))
        return out
    out = []
    for w in range(n_worlds):
        carry, trace = HostExecutor().run(
            run_study,
            index_world(truths, w),
            roots[w],
            n_steps,
            policy_form=form,
            wake_fn=wake_fn,
            service=_service(case),
            refit=_refit(case),
        )
        out.append((carry, trace, views_of(w, trace)))
    return out


def probed(study):
    """The study with a `PrivateProbe` around its process."""
    return eqx.tree_at(lambda s: s.process, study, PrivateProbe(inner=study.process))


def run_case(
    case,
    executor,
    *,
    n_worlds=2,
    n_steps=N_STEPS,
    probe=True,
    probe_wake_fn=wake_by_audience,
):
    """Run `case` on `executor` ("host" or "compiled"); one `CaseRun` per world.

    With `probe` the same worlds also run with private events added (under
    `probe_wake_fn`, which a negative control replaces), and their wake and decision
    counts are kept on the run.
    """
    if executor not in EXECUTORS:
        raise ValueError(f"executor must be one of {EXECUTORS}")
    require_x64()
    study = case.make_study()
    truths, roots = make_worlds(study, n_worlds)
    counted = dataclasses.replace(
        study, observer=CountingObserver(inner=study.observer)
    )
    main = _execute(
        case, counted, executor, truths, roots, n_steps, wake_by_audience, True
    )
    other_forms = (
        [f for f in case.policy_forms if f != "array"] if executor == "host" else []
    )
    form_runs = {
        form: _execute(
            case,
            counted,
            executor,
            truths,
            roots,
            n_steps,
            wake_by_audience,
            True,
            form,
        )
        for form in other_forms
    }
    probes = [None] * n_worlds
    if probe:
        extra = _execute(
            case, probed(study), executor, truths, roots, n_steps, probe_wake_fn, False
        )
        probes = [(wake_count(t), int(c.n_dec), public_events(t)) for c, t, _ in extra]
    layout = study.observer.layout(study.caps)
    out = []
    for w, (counted_carry, trace, views) in enumerate(main):
        carry = dataclasses.replace(
            counted_carry,
            belief=plain(counted_carry.belief),
            pbelief=plain(counted_carry.pbelief),
        )
        out.append(
            CaseRun(
                case=case,
                executor=executor,
                world=w,
                study=study,
                carry=carry,
                trace=trace,
                history=canonical_history(carry, trace, layout),
                views=views,
                folds=tuple(
                    tuple(np.asarray(x) for x in fold_counts(b))
                    for b in (counted_carry.belief, counted_carry.pbelief)
                ),
                form_views={form: runs[w][2] for form, runs in form_runs.items()},
                probe=probes[w],
            )
        )
    return out
