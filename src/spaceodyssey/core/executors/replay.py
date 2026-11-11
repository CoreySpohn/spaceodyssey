"""Replay a recorded run log into an observer belief."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import refresh
from spaceodyssey.core.carry import _tree_set
from spaceodyssey.core.enums import I64, Audience, Capacity, EventKind, RecordStatus
from spaceodyssey.core.events import blank_row
from spaceodyssey.core.executors.base import (
    KILL_TOLERANCE,
    belief_leaves,
)
from spaceodyssey.core.keys import fold_key
from spaceodyssey.core.lift import lifting_jit
from spaceodyssey.core.records import (
    AWARD_COLUMNS,
    BUNDLE_COLUMNS,
    LOG_INT_FIELDS,
    OPS_COLUMNS,
    REFUND_COLUMNS,
    Record,
    masked_record,
)
from spaceodyssey.core.refresh import expired_subjects, refresh_enabled
from spaceodyssey.core.snapshot import named_leaves


@lifting_jit
def _fold(study, belief, rec, key=None):
    return study.observer.analyze(study.observer.fold(belief, rec, study.caps, key=key))


def _row(log, i):
    fields = {n: jnp.asarray(getattr(log, n)[i]) for n in LOG_INT_FIELDS}
    fields.update({n: jnp.asarray(getattr(log, n)[i]) for n in ("cost", "charged")})
    fields["payload"] = jnp.asarray(log.payload[i])
    fields["valid"] = jnp.asarray(True)
    if log.program is not None:
        fields["program"] = jnp.asarray(log.program[i])
    for n in (*REFUND_COLUMNS, *AWARD_COLUMNS, *BUNDLE_COLUMNS, *OPS_COLUMNS):
        if getattr(log, n) is not None:
            fields[n] = jnp.asarray(getattr(log, n)[i])
    return Record(**fields)  # the full row; callers mask it for their audience


def _fold_time_row(full, accepted, released):
    """The row before this release stamped its ticks or later revisions replaced it."""
    last = jnp.asarray(-1, I64)
    return dataclasses.replace(
        full,
        accepted_tick=full.accepted_tick if accepted else last,
        released_tick=full.released_tick if released else last,
        superseded_by=last,
    )


def _numpy_log(carry):
    """The carry's log with its integer columns and `valid` as numpy arrays."""
    return dataclasses.replace(
        carry.log,
        **{n: np.asarray(vars(carry.log)[n]) for n in (*LOG_INT_FIELDS, "valid")},
    )


def _check_release(log, i, tick, subject, acquisition_index, channel):
    """Raise `ValueError` unless the release names the log row at slot `i`."""
    if not (
        0 <= i < len(log.valid)
        and log.valid[i]
        and log.subject[i] == subject
        and log.acquisition_index[i] == acquisition_index
        and log.channel[i] == channel
    ):
        raise ValueError(f"release at tick {tick} matches no log row at slot {i}")


def replay_belief(study, carry, trace, root=None):
    """Rebuild the observer belief of one world from its recorded log alone.

    Reads only the record log and the order in which the run released rows (the
    handled release and result steps of the trace). Each acquisition is folded once, in
    release order, through the study's observer; later revisions are not folded. With
    the executor's own observer on the same rows in the same order the result equals
    `carry.belief` bit for bit. Each fold gets the record's fold key under the
    world's `root` key (`keys.fold_key`, as in the run); without `root` the folds
    get no key, which only an observer that draws nothing accepts. A release that
    names no matching log row raises `ValueError`.
    """
    log = dataclasses.replace(
        carry.log,
        **{n: np.asarray(getattr(carry.log, n)) for n in (*LOG_INT_FIELDS, "valid")},
    )
    kind = np.asarray(trace.kind)
    released = (kind == EventKind.RELEASE) | (kind == EventKind.RESULT)
    handled = np.asarray(trace.handled) & released
    steps = zip(
        *(
            np.asarray(getattr(trace, n))[handled].tolist()
            for n in (
                "log_slot",
                "tick",
                "subject",
                "acquisition_index",
                "channel",
                "audience",
            )
        ),
        strict=True,
    )
    belief = study.observer.init(study.caps)
    used = np.zeros(log.valid.shape[0], dtype=bool)
    released = set()
    folded = set()
    for i, tick, subject, acquisition_index, channel, audience in steps:
        _check_release(log, i, tick, subject, acquisition_index, channel)
        if log.status[i] == RecordStatus.UNRELEASED:
            continue
        if int(log.kind[i]) in study.caps.derived_kinds:
            # a derived row (a product or a declared summary) is never folded
            continue
        reach = audience & int(log.audience[i])
        was_released = i in released
        if reach & Audience.POLICY:
            released.add(i)
        if not reach & Audience.OBSERVER:
            # a row the observer may not read is never folded into its belief
            continue
        if not used[i] and log.accepted_tick[i] != tick:
            raise ValueError(
                f"release at tick {tick} matches no accepted row at slot {i}"
            )
        used[i] = True
        acquisition = int(log.acquisition_id[i])
        if acquisition in folded:
            continue
        folded.add(acquisition)
        full = _fold_time_row(_row(carry.log, i), False, was_released)
        key = None if root is None else fold_key(root, full, study)
        belief = _fold(
            study, belief, masked_record(full, study.caps, Audience.OBSERVER), key
        )
    return belief


# ---- keyed replay with refresh ----------------------------------------------------


class Replayed(eqx.Module):
    """What `replay_run` rebuilt and checked.

    `belief` and `pbelief` are the replayed observer belief and policy summary,
    `checkpoints` counts the checkpoints that completed and `products` the product
    rows checked against the log. A checkpoint the core refused to install (too many
    rows, no room in the log or the internal ids) is in `refused` as `(ordinal, bits)`,
    the `Capacity` bits the replayed install set; a hold the run never completed
    (a budget cut) is `unfinished`; `overflow_mask` is the run's own final mask, which
    also holds refusals the replay does not meet, such as an invalid correction.
    """

    belief: object
    pbelief: object
    checkpoints: int = eqx.field(static=True)
    products: int = eqx.field(static=True)
    refused: tuple = eqx.field(static=True, default=())
    unfinished: bool = eqx.field(static=True, default=False)
    overflow_mask: int = eqx.field(static=True, default=0)


@lifting_jit
def _fold_summary(study, pbelief, rec, key):
    return study.observer.fold_summary(pbelief, rec, study.caps, key=key)


def _request(view, root, study):
    """The refit request of the frozen replay world: the run's own gather."""
    return refresh._gather(view, root, study)


def _visible(carry, n_log, accepted, released):
    """Log columns as the run held them at a checkpoint, from the trace order alone.

    A row exists once it is written (`slot < n_log`) and counts for an audience once a
    handled release step reached that audience. `superseded_by` is never read: the
    revision set recomputes replacement from the revisions the audience has received.
    """
    log = carry.log
    size = log.valid.shape[0]
    mask = lambda slots: np.isin(np.arange(size), list(slots))  # noqa: E731
    last = jnp.asarray(-1, I64)
    return {
        "log.valid": jnp.asarray(log.valid) & jnp.asarray(np.arange(size) < n_log),
        "log.accepted_tick": jnp.where(mask(accepted), log.accepted_tick, last),
        "log.released_tick": jnp.where(mask(released), log.released_tick, last),
    }


def _row_unequal(a, b):
    """Whether two arrays differ: ints exactly, floats beyond the kill tolerance."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return True
    if np.issubdtype(a.dtype, np.floating):
        return not np.allclose(
            a, b, rtol=KILL_TOLERANCE, atol=KILL_TOLERANCE, equal_nan=True
        )
    return not np.array_equal(a, b)


REFUSALS = Capacity.REFIT_ROWS | Capacity.STABLE_IDS | Capacity.LOG_ROWS


def _phase_after(phase, tick, bounds):
    """The phase a decision at `tick` leaves: the phase never goes back."""
    return max(phase, int((tick >= bounds).sum()))


def _check_ids(carry, first, count, ordinal):
    """The reserved internal ids of a checkpoint are fresh and unique.

    Each is above the id of every row written before the checkpoint (the id counter only
    advances) and held by no other row of the log. Consecutive order is checked with the
    rows themselves.
    """
    log = _numpy_log(carry)
    ids = log.acquisition_id
    mine = ids[first : first + count]
    earlier = ids[:first][log.valid[:first]]
    if count and earlier.size and mine.min() <= earlier.max():
        raise ValueError(
            f"checkpoint {ordinal}: a reserved id is not above earlier ids"
        )
    for slot in range(first, first + count):
        shared = (log.valid & (ids == ids[slot])).sum()
        if shared != 1:
            raise ValueError(
                f"checkpoint {ordinal}: product id at slot {slot} is shared"
            )


def _check_products(new, carry, first, count, ordinal):
    """The rows a replayed checkpoint wrote equal the run's rows at the same slots.

    Every column, the payload and the pedigree must match. The first internal id is
    the one the run reserved (the trace does not hold the id counter), so its freshness
    is checked separately (`_check_ids`).
    """
    for slot in range(first, first + count):
        mine = named_leaves(_row(new.log, slot))
        theirs = named_leaves(_row(carry.log, slot))
        mine["product"] = new.log.product[slot]
        theirs["product"] = carry.log.product[slot]
        for name, leaf in mine.items():
            if _row_unequal(leaf, theirs[name]):
                raise ValueError(
                    f"checkpoint {ordinal}: product row at slot {slot} differs in "
                    f"{name}"
                )


def replay_run(run, roots, carry, trace, *, refit=None, executor=None):
    """Replay one world with refresh from its log and trace, checking every product.

    `run` is a loaded run (its `study`, and its `refit` boundary or boundaries by
    executor; `executor` picks one, defaulting to the run's own) or a study with `refit`
    given. `roots` is the world's root key. Walks the handled `RELEASE`, `RESULT` and
    `REFRESH` steps in trace order and uses nothing else of the run's order: each
    audience's revision set is rebuilt from the releases walked so far, never from the
    final `superseded_by`. At a checkpoint's hold step the replay refits through the
    same boundary with the same keys (`refresh_key` over the set rebuilt here) and, at
    the completion step, installs and checks:

    - the rows written (payload and pedigree) against the log;
    - the written count and the log length against the trace;
    - the trace projection of the belief against the replayed one.

    Incremental observers fold the first accepted revision of each acquisition, as
    `replay_belief` does, and a derived row is never folded. The final `belief` and
    `pbelief` must equal the run's: ints exactly, floats within the kill tolerance.
    A hold with no completion (a run cut by its step budget) installs nothing, as in
    the run. Any mismatch raises `ValueError` naming it.
    """
    study = run.study if hasattr(run, "study") else run
    caps, observer = study.caps, study.observer
    boundary = _bound(run, refit, executor)
    if boundary is None and refresh_enabled(caps) and int(study.refresh.every_ticks):
        raise ValueError("a study that checkpoints needs the refit boundary of its run")
    incremental = observer.capabilities().incremental
    layout = observer.layout(caps)
    log = _numpy_log(carry)
    refit_kinds = {int(k) for k in caps.refit_kinds}
    names = ("kind", "tick", "subject", "acquisition_index", "channel", "log_slot")
    handled = np.asarray(trace.handled)
    cols = {n: np.asarray(v) for n, v in vars(trace).items() if v is not None}
    belief = observer.init(caps)
    pbelief = observer.init_summary(caps)
    dirty = np.zeros(np.shape(carry.refresh_dirty), dtype=bool)
    status = np.full(np.shape(carry.refresh_status), -1, dtype=np.int64)
    accepted, released = set(), set()  # slots an audience has received
    obs_acq, pub_acq = set(), {}  # acquisition ids received, with released revisions
    pending = None
    n_checkpoints = n_products = 0
    refused = []
    phase = 0
    bounds = np.asarray(study.agenda.phase_until)[:-1]
    wake = np.asarray(trace.wake)
    releasing = (EventKind.RELEASE, EventKind.RESULT)
    for step in np.flatnonzero(handled):
        at = {n: int(cols[n][step]) for n in (*names, "audience", "n_written", "n_log")}
        if at["kind"] == EventKind.REFRESH:
            ordinal = at["acquisition_index"]
            if pending is None:
                state = (belief, pbelief, dirty, status, accepted, released, phase)
                pending = _hold(study, carry, roots, boundary, at, state)
            else:
                if pending["ordinal"] != ordinal:
                    raise ValueError(
                        f"checkpoint {ordinal} completes checkpoint "
                        f"{pending['ordinal']}"
                    )
                new = pending["new"]
                written = int(new.n_log) - pending["n_log"]
                if written != at["n_written"] or int(new.n_log) != at["n_log"]:
                    raise ValueError(
                        f"checkpoint {ordinal}: wrote {written} rows, the trace says "
                        f"{at['n_written']}"
                    )
                _check_products(new, carry, pending["n_log"], written, ordinal)
                _check_ids(carry, pending["n_log"], written, ordinal)
                mean, aux = layout.project(new.belief)
                for got, name in ((mean, "belief_mean"), (aux, "belief_aux")):
                    if _row_unequal(got, vars(trace)[name][step]):
                        raise ValueError(f"checkpoint {ordinal}: trace {name} differs")
                bits = int(new.overflow_mask) & int(REFUSALS)
                if bits:
                    refused.append((ordinal, bits))
                belief, pbelief = new.belief, new.pbelief
                dirty = np.array(new.refresh_dirty)
                status = np.array(new.refresh_status)
                n_checkpoints += 1
                n_products += written
                pending = None
        elif at["kind"] in releasing:
            belief, pbelief = _release(
                study,
                carry,
                log,
                roots,
                at,
                (belief, pbelief, dirty),
                (accepted, released, obs_acq, pub_acq),
                incremental,
                refit_kinds,
            )
        elif at["kind"] == EventKind.WAIT and dirty.size:
            due = expired_subjects(
                observer, caps, pbelief, jnp.asarray(status[1]), at["tick"]
            )
            dirty |= np.asarray(due)[None, :]
        if wake[step]:
            phase = _phase_after(phase, at["tick"], bounds)
    ours = belief_leaves(_Pair(belief, pbelief), layout)
    theirs = belief_leaves(carry, layout)
    for name, leaf in ours.items():
        if _row_unequal(leaf, theirs[name]):
            raise ValueError(f"the replayed {name} differs from the run's")
    mask = int(np.asarray(carry.overflow_mask))
    for ordinal, bits in refused:
        if bits & ~mask:
            raise ValueError(f"checkpoint {ordinal} was refused in the replay only")
    return Replayed(
        belief,
        pbelief,
        n_checkpoints,
        n_products,
        tuple(refused),
        pending is not None,
        mask,
    )


def _release(study, carry, log, roots, at, state, seen, incremental, refit_kinds):
    """One handled release or result step: folds, received state and dirty flags.

    The folds see the row as the handler did: before this step stamped the ticks, with
    the supersession then in force (the observer's own is always clear, the policy's is
    the highest revision released to it before). Returns the new beliefs; `state` holds
    the beliefs and the dirty flags (updated in place), `seen` the received state
    (updated in place).
    """
    belief, pbelief, dirty = state
    accepted, released, obs_acq, pub_acq = seen
    caps = study.caps
    i = at["log_slot"]
    _check_release(
        log, i, at["tick"], at["subject"], at["acquisition_index"], at["channel"]
    )
    kind = int(log.kind[i])
    if log.status[i] == RecordStatus.UNRELEASED or kind in caps.derived_kinds:
        return belief, pbelief
    reach = at["audience"] & int(log.audience[i])
    acquisition = int(log.acquisition_id[i])
    full = _row(carry.log, i)
    key = fold_key(roots, full, study) if incremental else None
    eligible = kind in refit_kinds
    before = _fold_time_row(full, i in accepted, i in released)
    if reach & Audience.OBSERVER and i not in accepted:
        if log.accepted_tick[i] != at["tick"]:
            raise ValueError(
                f"release at tick {at['tick']} matches no accepted row at slot {i}"
            )
        first = acquisition not in obs_acq
        if incremental and first:
            row = masked_record(before, caps, Audience.OBSERVER)
            belief = _fold(study, belief, row, key)
        accepted_now = True
        if eligible and dirty.size:
            dirty[0, at["subject"]] = True
    else:
        accepted_now = False
    if reach & Audience.POLICY and i not in released:
        if log.released_tick[i] != at["tick"]:
            raise ValueError(
                f"release at tick {at['tick']} matches no released row at slot {i}"
            )
        earlier = pub_acq.setdefault(acquisition, [])
        revision = int(log.revision[i])
        replaced = max([r for r in earlier if r > revision], default=-1)
        if incremental and not earlier:
            seen_row = dataclasses.replace(
                before, superseded_by=jnp.asarray(replaced, I64)
            )
            row = masked_record(seen_row, caps, Audience.POLICY)
            pbelief = _fold_summary(study, pbelief, row, key)
        earlier.append(revision)
        released.add(i)
        if eligible and dirty.size:
            dirty[1, at["subject"]] = True
    if accepted_now:
        accepted.add(i)
        obs_acq.add(acquisition)
    return belief, pbelief


@dataclasses.dataclass(frozen=True)
class _Pair:
    belief: object
    pbelief: object


def _bound(run, refit, executor):
    """The refit boundary to replay with: given, or the run's own."""
    if refit is not None:
        return refit
    held = getattr(run, "refit", None)
    if isinstance(held, dict):
        name = executor or getattr(run, "executor", None)
        if name not in held:
            raise ValueError(f"the run binds no refit boundary for executor {name!r}")
        return held[name]
    return held


def _hold(study, carry, roots, boundary, at, state):
    """Refit a frozen replay world at its hold step: request, boundary, install."""
    belief, pbelief, dirty, status, accepted, released, phase = state
    ordinal, n_log = at["acquisition_index"], at["n_log"]
    log = carry.log
    # The id counter is not in the trace: a checkpoint that wrote rows reserved ids from
    # the first of them; one that wrote none (refused) is seeded with the run's final
    # id counter, which any refusal for lack of ids is consistent with.
    ahead = n_log < log.valid.shape[0] and bool(log.valid[n_log])
    next_sid = vars(log)["acquisition_id"][n_log] if ahead else carry.next_sid
    held = dataclasses.replace(
        blank_row(carry.request),
        tick=jnp.asarray(at["tick"], I64),
        acquisition_index=jnp.asarray(ordinal, I64),
        sid=jnp.asarray(-1, I64),
        valid=jnp.asarray(True),
    )
    view = _tree_set(
        carry,
        **_visible(carry, n_log, accepted, released),
        refresh=held,
        refresh_dirty=jnp.asarray(dirty),
        refresh_status=jnp.asarray(status),
        belief=belief,
        pbelief=pbelief,
        n_log=jnp.asarray(n_log, I64),
        next_sid=jnp.asarray(next_sid, I64),
        phase=jnp.asarray(phase, I64),
        # Reconstruct this install's refusals, not those of later checkpoints.
        overflow_mask=carry.overflow_mask & ~jnp.asarray(int(REFUSALS), I64),
    )
    request = _request(view, roots, study)
    batch = jax.tree_util.tree_map(lambda x: x[None], request)
    result = boundary.refit_batch(study, batch)
    one = jax.tree_util.tree_map(lambda x: x[0], result)
    new = refresh._commit(view, request, one, study)
    return {"ordinal": ordinal, "n_log": n_log, "new": new}
