"""Refresh protocol: checkpoints, the request and result of a boundary, installation.

A refit-capable observer's belief is recomputed from its accepted rows at
checkpoints, core events of kind `EventKind.REFRESH` on a tick grid
(`RefreshSchedule`). A checkpoint takes two handled transitions. The first holds the
world (`Carry.refresh`) and leaves the row queued; the executor then gathers a
`RefitRequest` from the frozen world, hands it to a boundary (`MaskedBatchRefit` for
array kernels, `HostLoopRefit` for host Python), and installs the `RefitResult` with
`commit_refresh`; the second transition consumes the row and the decision owed at
that tick runs after it.

Everything here has an audience axis of two, observer then policy, and a subject
axis. A refit returns a per-subject slice, not a whole belief: the leaves the
observer names in `subject_leaves`, taken at the refit subject. A boundary batched
over audiences and subjects therefore holds `2 * n_subjects` slices, not
`2 * n_subjects` full beliefs, and the core rebuilds the full-shaped belief
`install` reads from the slice.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from spaceodyssey.core.carry import _tree_set
from spaceodyssey.core.enums import (
    F64,
    I64,
    INSTALLING_STATUSES,
    PUBLIC,
    REFRESH_TIE,
    Audience,
    B,
    Capacity,
    EventKind,
    RecordStatus,
    RefitStatus,
    Stage,
)
from spaceodyssey.core.events import blank_row, enqueue, where_tree
from spaceodyssey.core.keys import fold_keys, refit_key
from spaceodyssey.core.lift import lifting_jit
from spaceodyssey.core.records import (
    PayloadColumn,
    Record,
    RecordKind,
    record_none,
    refit_rows,
    write_log,
)
from spaceodyssey.core.snapshot import _mix, named_leaves

# The derived record kind a refresh product row carries; a study registers it and
# names its index in `RefreshSchedule.kind`.
REFRESH_PRODUCT_KIND = RecordKind(
    "refresh_product",
    (PayloadColumn("status", PUBLIC), PayloadColumn("n_rows", PUBLIC)),
    derived=True,
)

AUDIENCES = (Audience.OBSERVER, Audience.POLICY)


def _as_i64(x):
    return jnp.asarray(x, I64)


class RefreshSchedule(eqx.Module):
    """The checkpoint grid and the identities of the product rows it writes.

    Checkpoints fall at `offset_ticks + k * every_ticks`, on an unclamped grid;
    `every_ticks = 0` disables refresh, so a study without it is untouched. The two
    channel ids name the observer's and the policy's product channels (they are not
    admission channels), and `kind` is the index of the derived record kind the
    product rows carry (`REFRESH_PRODUCT_KIND` in the study's registry).
    """

    every_ticks: jax.Array = eqx.field(default=0, converter=_as_i64)
    offset_ticks: jax.Array = eqx.field(default=0, converter=_as_i64)
    observer_channel: jax.Array = eqx.field(default=0, converter=_as_i64)
    policy_channel: jax.Array = eqx.field(default=0, converter=_as_i64)
    kind: jax.Array = eqx.field(default=0, converter=_as_i64)


def refresh_enabled(caps):
    """Whether the study's capacities carry the refresh protocol (structurally)."""
    return caps.rows_per_subject > 0


def _grid_position(schedule, tick, stage):
    """`(tick, ordinal, fits)` of the checkpoint, with overflow-safe arithmetic.

    `fits` is false when the grid tick is not representable in int64; the tick is then
    saturated at the largest int64 and the ordinal is meaningless.
    """
    big = jnp.asarray(np.iinfo(np.int64).max, I64)
    every = jnp.maximum(schedule.every_ticks, 1)
    offset = schedule.offset_ticks
    gap = jnp.maximum(tick - offset, 0)
    k = gap // every + (gap % every != 0).astype(I64)
    on_grid = (tick >= offset) & (gap % every == 0)
    bump = on_grid & (stage > Stage.ANALYZE)
    room = jnp.maximum(big - offset, 0) // every
    # k + bump <= room, tested before the increment so it cannot wrap
    fits = (k < room) | ((k == room) & ~bump)
    k = k + (bump & (k < big)).astype(I64)
    at = offset + jnp.where(fits, k, 0) * every
    return jnp.where(fits, at, big), k, fits


def next_grid_tick(schedule, tick, stage):
    """The checkpoint a row accepted at `(tick, stage)` falls in: `(tick, ordinal)`.

    The smallest grid tick at or after `tick`, or the next one when the position is
    already past `Stage.ANALYZE` of a grid tick (an `ADMIT`-stage result). The grid
    starts at the offset, so earlier positions map to ordinal 0. A grid tick beyond
    int64 saturates at the largest int64 instead of wrapping.
    """
    at, k, _ = _grid_position(schedule, tick, stage)
    return at, k


def schedule_refresh(c, mark, study):
    """Queue a checkpoint where `mark` and none is queued; a tick too far overflows.

    Returns the carry. The row sits at the grid tick in `Stage.ANALYZE`, which is never
    earlier than the current position, so the queue never relocates it.
    """
    caps = study.caps
    sched = study.refresh
    queued = jnp.any(c.ev.valid & (c.ev.kind == EventKind.REFRESH))
    want = mark & (sched.every_ticks > 0) & ~queued
    tick, k, fits = _grid_position(sched, c.tick, c.stage)
    beyond = want & (~fits | (tick > caps.max_tick))
    c1 = _tree_set(
        c,
        overflow=c.overflow | beyond,
        overflow_mask=c.overflow_mask | jnp.where(beyond, Capacity.GRID_TICK, 0),
    )
    return enqueue(
        c1,
        want & ~beyond,
        kind=EventKind.REFRESH,
        tick=tick,
        stage=Stage.ANALYZE,
        acquisition_index=k,
        audience=Audience.OBSERVER,
        tie=REFRESH_TIE,
        caps=caps,
    )


# ---- subject slices ---------------------------------------------------------------


def slice_spec(study):
    """`(name, tail shape, dtype)` of each slice leaf, in `subject_leaves` order."""
    leaves = named_leaves(study.observer.layout(study.caps).belief)
    return tuple(
        (name, tuple(leaves[name].shape[1:]), leaves[name].dtype)
        for name in study.observer.subject_leaves
    )


def take_subject(belief, names, subject):
    """The slice of `belief`: each named leaf at `subject`, as a tuple."""
    leaves = named_leaves(belief)
    return tuple(leaves[name][subject] for name in names)


def put_subject(belief, names, values, subject):
    """`belief` with each named leaf set to its slice value at `subject`."""
    paths, treedef = jax.tree_util.tree_flatten_with_path(belief)
    given = dict(zip(names, values, strict=True))
    out = []
    for path, leaf in paths:
        name = jax.tree_util.keystr(path, simple=True, separator=".")
        out.append(leaf.at[subject].set(given[name]) if name in given else leaf)
    return jax.tree_util.tree_unflatten(treedef, out)


# ---- request and result -----------------------------------------------------------


class RefitEntry(eqx.Module):
    """One audience and subject of a request: what a kernel or a host fit reads."""

    subject: jax.Array
    rows: Record  # [rows_per_subject], counting rows first, masked for the audience
    key: jax.Array  # the refit key of the row set
    row_keys: jax.Array  # [rows_per_subject], the fold key of each row
    n_rows: jax.Array  # counting rows before truncation


class RefitRequest(eqx.Module):
    """A frozen world's refresh request; every leaf has axes `[2, n_subjects, ...]`.

    `dirty` marks the entries to refit. `source` names, by flat index
    `audience * n_subjects + subject`, the entry whose computation serves each one: its
    own, or the observer's when both audiences' canonically masked rows, ints and float
    bits, and keys are equal. `tick` and `ordinal` identify the checkpoint.
    """

    tick: jax.Array
    ordinal: jax.Array
    dirty: jax.Array
    n_rows: jax.Array
    rows: Record
    row_keys: jax.Array
    keys: jax.Array
    source: jax.Array


class RefitResult(eqx.Module):
    """A boundary's answer, with axes `[2, n_subjects, ...]`.

    `slices` holds one array per `subject_leaves` name, the subject's part of the refit
    belief. `status` is a `RefitStatus`, `diagnostics` the boundary's own numbers, and
    `computed` is true where this entry was computed rather than copied from the entry
    `RefitRequest.source` names. A masked entry holds zeros, status `NO_SUPPORT` and is
    never read.
    """

    slices: tuple
    status: jax.Array
    diagnostics: jax.Array
    computed: jax.Array


def _expand(mask, like):
    return mask.reshape(mask.shape + (1,) * (like.ndim - mask.ndim))


def _canonical(rows):
    """Zero every leaf of a row that does not count, so inputs compare by content."""
    return jax.tree_util.tree_map(
        lambda x: jnp.where(_expand(rows.valid, x), x, jnp.zeros_like(x)), rows
    )


def _cut(x, n):
    """The first `n` entries along axis 0, zero padded when there are fewer."""
    if n <= x.shape[0]:
        return x[:n]
    return jnp.pad(x, [(0, n - x.shape[0])] + [(0, 0)] * (x.ndim - 1))


def _bits(x):
    if jnp.issubdtype(x.dtype, jnp.floating):
        return lax.bitcast_convert_type(x.astype(F64), I64)
    return x.astype(I64)


def revision_set_digest(rows):
    """64-bit digest of a subject's revision set from its refit rows.

    Hashes the `(channel, acquisition index, revision)` triples of the counting rows
    in the canonical order the rows come in, then their count. Rows that do not count
    leave the running hash untouched, so the digest depends on the set alone and not
    on the padded length of `rows`.
    """
    words = jnp.stack([rows.channel, rows.acquisition_index, rows.revision], axis=1)
    words = words.astype(I64).astype(jnp.uint64)

    seed = _mix(jnp.asarray(np.uint64(17), jnp.uint64))
    h, _ = lax.scan(lambda h, item: _fold_row(h, *item), seed, (words, rows.valid))
    count = rows.valid.sum().astype(jnp.uint64)
    return lax.bitcast_convert_type(_mix(h ^ count), I64)


def _fold_row(h, row, ok):
    new = h
    for k in range(3):
        new = _mix(new ^ row[k])
    return jnp.where(ok, new, h), None


def _entry_rows(log, caps, audience, subject, n):
    rows = refit_rows(log, caps, audience, subject)
    count = rows.valid.sum().astype(I64)
    return jax.tree_util.tree_map(lambda x: _cut(x, n), _canonical(rows)), count


def gather_refit_rows(carry, root, study):
    """The `RefitRequest` of one frozen world, read from its log.

    Exact because a frozen world's log does not change while the host computes. Rows
    come from `records.refit_rows` for each audience and subject, are zeroed where they
    do not count, and are cut to `rows_per_subject`; `n_rows` keeps the full count so
    that `commit_refresh` can refuse a truncation.
    """
    caps = study.caps
    n, count = caps.rows_per_subject, caps.n_subjects
    subjects = jnp.arange(count)
    per = [
        jax.vmap(lambda s, a=a: _entry_rows(carry.log, caps, a, s, n))(subjects)
        for a in AUDIENCES
    ]
    rows = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y]), per[0][0], per[1][0])
    n_rows = jnp.stack([per[0][1], per[1][1]])
    row_keys = jax.vmap(jax.vmap(lambda r: fold_keys(root, r, study)))(rows)
    digest = jax.vmap(jax.vmap(revision_set_digest))(rows)
    keys = jax.vmap(
        lambda d: jax.vmap(lambda dig, s: refit_key(root, s, dig))(d, subjects)
    )(digest)
    same = jnp.ones(count, B)
    for leaf in jax.tree_util.tree_leaves(rows):
        bits = _bits(leaf)
        same = same & (bits[0] == bits[1]).reshape(count, -1).all(axis=1)
    dirty = carry.refresh_dirty
    share = same & dirty[0] & dirty[1] & (keys[0] == keys[1]).all(axis=-1)
    source = jnp.stack([subjects, jnp.where(share, subjects, count + subjects)])
    return RefitRequest(
        tick=carry.refresh.tick,
        ordinal=carry.refresh.acquisition_index,
        dirty=dirty,
        n_rows=n_rows,
        rows=rows,
        row_keys=row_keys,
        keys=keys,
        source=source.astype(I64),
    )


# ---- installation -----------------------------------------------------------------


def _product_record(c, study, a, s, status, n_rows, valid, request):
    """The derived product row of one audience and subject at the checkpoint."""
    caps = study.caps
    sched = study.refresh
    tick = request.tick
    channel = sched.observer_channel if a == 0 else sched.policy_channel
    payload = jnp.zeros(caps.payload_width, F64)
    payload = payload.at[0].set(status.astype(F64)).at[1].set(n_rows.astype(F64))
    return dataclasses.replace(
        record_none(caps),
        kind=sched.kind,
        subject=s,
        channel=channel,
        acquisition_index=request.ordinal,
        start_tick=tick,
        end_tick=tick,
        available_tick=tick,
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
        phase=c.phase,
        acquisition_id=c.next_sid,
        revision=jnp.asarray(0, I64),
        # an observer row is accepted and a policy row released, as a release marks them
        released_tick=tick if a == 1 else jnp.asarray(-1, I64),
        accepted_tick=tick if a == 0 else jnp.asarray(-1, I64),
        superseded_by=jnp.asarray(-1, I64),
        audience=jnp.asarray(AUDIENCES[a], I64),
        source=channel,
        vintage=c.phase,
        payload=payload,
        valid=valid,
    )


def pedigree_words(request, a, s, status, caps):
    """The pedigree column of the product row of audience `a` and subject `s`.

    Audience flag, refit status and row count, then the `(channel, acquisition index,
    revision)` triples of the counting rows the entry was computed from, in canonical
    order, zero padded to `Capacities.product_width`. The rows are the request's, the
    revision set the boundary read.
    """
    rows = jax.tree_util.tree_map(lambda x: x[a, s], request.rows)
    ok = rows.valid
    triples = jnp.stack([rows.channel, rows.acquisition_index, rows.revision], axis=1)
    triples = jnp.where(ok[:, None], triples, 0).astype(I64).reshape(-1)
    head = jnp.stack(
        [
            jnp.asarray(int(AUDIENCES[a]), I64),
            jnp.asarray(status, I64),
            ok.sum().astype(I64),
        ]
    )
    words = jnp.concatenate([head, triples])
    return jnp.pad(words, (0, caps.product_width - words.shape[0]))


def commit_refresh(carry, request, result, study):
    """Install a boundary's answer into a frozen world, then release the hold.

    For each audience and subject whose entry was dirty, the slice the boundary returned
    replaces the subject's part of the belief when its status is `OK` or `DEGRADED`
    (the observer through `install`, the policy through `install_summary` of the refit
    belief's summary), the status is recorded in `Carry.refresh_status`, the dirty flag
    clears, and one derived product row per entry is logged and counted in its
    audience's release version. Nothing is folded. The observer's whole belief is
    analyzed once, after its installs. A subject with more counting rows than
    `rows_per_subject` refuses the whole answer and sets `Capacity.REFIT_ROWS`; so does
    too little room in the log or in the internal ids for every product row, which
    are reserved before anything changes (`Capacity.LOG_ROWS`, `Capacity.STABLE_IDS`).

    The queued checkpoint row is marked installed (`disposition`) with its product row
    count (`group`) and last slot, so its second handling consumes it, and
    `Carry.refresh` clears. A world that holds no checkpoint is returned unchanged.
    """
    caps = study.caps
    observer = study.observer
    names = observer.subject_leaves
    live = carry.refresh.valid
    too_many = (request.dirty & (request.n_rows > caps.rows_per_subject)).any()
    # Reserve every product row and internal id before anything changes: with too
    # little room the whole answer is refused, never an install without its rows.
    needed = request.dirty.sum().astype(I64)
    no_ids = carry.next_sid + needed > caps.max_sid
    no_rows = carry.n_log + needed > caps.n_log
    go = live & ~too_many & ~no_ids & ~no_rows
    base = observer.init(caps)

    def body(s, state):
        c, n_written, last_slot, any_obs = state
        for a in (0, 1):
            proc = go & request.dirty[a, s]
            status = result.status[a, s]
            installs = proc & jnp.isin(
                status, jnp.asarray([int(x) for x in INSTALLING_STATUSES], I64)
            )
            fitted = put_subject(
                base, names, tuple(piece[a, s] for piece in result.slices), s
            )
            if a == 0:
                new = observer.install(c.belief, fitted, s)
                c = _tree_set(c, belief=where_tree(installs, new, c.belief))
                any_obs = any_obs | installs
            else:
                new = observer.install_summary(c.pbelief, observer.summary(fitted), s)
                c = _tree_set(c, pbelief=where_tree(installs, new, c.pbelief))
            rec = _product_record(
                c, study, a, s, status, request.n_rows[a, s], proc, request
            )
            before = c.n_log
            c, slot = write_log(c, rec, caps)
            wrote = c.n_log > before
            pedigree = pedigree_words(request, a, s, status, caps)
            product = c.log.product.at[slot].set(
                jnp.where(wrote, pedigree, c.log.product[slot])
            )
            c = _tree_set(
                c,
                **{"log.product": product},
                next_sid=c.next_sid + wrote.astype(I64),
                release_version=c.release_version.at[a].add(wrote.astype(I64)),
                refresh_dirty=c.refresh_dirty.at[a, s].set(
                    jnp.where(proc, False, c.refresh_dirty[a, s])
                ),
                refresh_status=c.refresh_status.at[a, s].set(
                    jnp.where(proc, status, c.refresh_status[a, s])
                ),
            )
            n_written = n_written + wrote.astype(I64)
            last_slot = jnp.where(wrote, slot, last_slot)
        return c, n_written, last_slot, any_obs

    zero = jnp.asarray(0, I64)
    done, n_written, last_slot, any_obs = lax.fori_loop(
        0, caps.n_subjects, body, (carry, zero, zero, jnp.asarray(False))
    )
    analyzed = observer.analyze(done.belief)
    done = _tree_set(done, belief=where_tree(any_obs, analyzed, done.belief))
    refused = live & (too_many | no_ids | no_rows)
    refused_bits = (
        jnp.where(live & too_many, Capacity.REFIT_ROWS, 0)
        | jnp.where(live & no_ids, Capacity.STABLE_IDS, 0)
        | jnp.where(live & no_rows, Capacity.LOG_ROWS, 0)
    )
    held = done.ev.valid & (done.ev.kind == EventKind.REFRESH)
    held = held & (done.ev.sid == carry.refresh.sid)
    mark = held & live
    ev = done.ev
    ev = eqx.tree_at(
        lambda e: (e.disposition, e.group, e.slot),
        ev,
        (
            jnp.where(mark, 1, ev.disposition),
            jnp.where(mark, n_written, ev.group),
            jnp.where(mark, last_slot, ev.slot),
        ),
    )
    done = _tree_set(
        done,
        ev=ev,
        refresh=blank_row(done.refresh),
        overflow=done.overflow | refused,
        overflow_mask=done.overflow_mask | refused_bits,
    )
    return where_tree(live, done, carry)


_gather = lifting_jit(gather_refit_rows)
_commit = lifting_jit(commit_refresh)
_gather_worlds = lifting_jit(jax.vmap(gather_refit_rows, in_axes=(0, 0, None)))
_commit_worlds = lifting_jit(jax.vmap(commit_refresh, in_axes=(0, 0, 0, None)))


def drop_hold(carry):
    """Release a held checkpoint without installing: the belief stays stale.

    Used when the step budget cannot pay for the completion transition: installing
    then would leave a belief and product rows the trace never shows, so the
    checkpoint is left as it was queued and the run reports `TRUNCATED`.
    """
    return _tree_set(carry, refresh=blank_row(carry.refresh))


def answer_refresh(carry, root, study, boundary, *, can_finish=True):
    """Answer the held checkpoint of one world through `boundary`; returns the carry.

    Without budget for the completion transition (`can_finish` false) nothing is
    installed and the hold is dropped.
    """
    if not bool(carry.refresh.valid):
        return carry
    if not can_finish:
        return drop_hold(carry)
    request = _gather(carry, root, study)
    batch = jax.tree_util.tree_map(lambda x: x[None], request)
    result = boundary.refit_batch(study, batch)
    one = jax.tree_util.tree_map(lambda x: x[0], result)
    return _commit(carry, request, one, study)


def answer_refreshes(carries, roots, study, boundary, left=None):
    """Answer the held checkpoints of a batch of worlds in one boundary call.

    Worlds with no budget left (`left <= 0`) cannot pay for the completion
    transition: their holds are dropped without an install.

    The frozen worlds' requests are batched on a world axis (`MaskedBatchRefit` fills
    its lanes with them, with no Python loop over worlds); worlds that hold no
    checkpoint are untouched.
    """
    held = np.asarray(carries.refresh.valid)
    if left is not None:
        broke = held & (np.asarray(left) <= 0)
        if broke.any():
            carries = jax.vmap(lambda c, m: where_tree(m, drop_hold(c), c))(
                carries, jnp.asarray(broke)
            )
        held = held & ~broke
    frozen = np.flatnonzero(held)
    if frozen.size == 0:
        return carries
    requests = _gather_worlds(carries, roots, study)
    batch = jax.tree_util.tree_map(lambda x: x[frozen], requests)
    result = boundary.refit_batch(study, batch)
    n_worlds = roots.shape[0]
    padded = jax.tree_util.tree_map(
        lambda x: jnp.zeros((n_worlds, *x.shape[1:]), x.dtype).at[frozen].set(x),
        result,
    )
    return _commit_worlds(carries, requests, padded, study)


# ---- boundaries -------------------------------------------------------------------


class AbstractRefitBoundary:
    """Answers refresh requests: `refit_batch(study, requests) -> results`.

    `requests` is a `RefitRequest` with a leading world axis; the result has the same
    axis. A boundary is stateless, a pure function of the request rows and keys, and
    holds no study: the study is an argument, so one instance serves any study and
    cannot be bound to the wrong one. `kind`, `version` and `model_hash` are what a run
    document records of it; comparisons are valid only within one model hash.
    """

    kind: str = "refit"
    version: int = 1
    n_diag: int = 1
    model_hash: int = 0

    def refit_batch(self, study, requests):
        """Answer every dirty entry of every world in `requests`."""
        raise NotImplementedError

    def binding(self):
        """The fields a run document records about this boundary."""
        return {
            "kind": self.kind,
            "version": self.version,
            "model_hash": self.model_hash,
        }


def observer_refit_kernel(study, entry):
    """The observer's own `refit` over the entry's rows, as a subject slice.

    Starts from `observer.init`, refits the rows with the entry's keys and takes the
    refit subject's leaves, with status `OK` and the counting row count as diagnostic.
    """
    observer, caps = study.observer, study.caps
    fitted = observer.refit(
        observer.init(caps), entry.rows, caps, entry.key, entry.row_keys
    )
    pieces = take_subject(fitted, observer.subject_leaves, entry.subject)
    diag = jnp.zeros(1, F64).at[0].set(entry.rows.valid.sum().astype(F64))
    return pieces, jnp.asarray(RefitStatus.OK, I64), diag


def _flat(x):
    return x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))


def _unflat(x, count):
    return x.reshape((2, count, *x.shape[1:]))


def _masked_lane(kernel, study, n_diag, request):
    """One world lane: the unique dirty entries through the kernel, the rest copied."""
    count = study.caps.n_subjects
    n_entries = 2 * count
    flat = jax.tree_util.tree_map(_flat, (request.rows, request.row_keys, request.keys))
    dirty, source = _flat(request.dirty), _flat(request.source)
    own = jnp.arange(n_entries)
    computed = dirty & (source == own)
    entry = RefitEntry(
        subject=own % count,
        rows=flat[0],
        key=flat[2],
        row_keys=flat[1],
        n_rows=_flat(request.n_rows),
    )
    shapes = jax.eval_shape(
        lambda en: kernel(study, en), jax.tree_util.tree_map(lambda x: x[0], entry)
    )
    nothing = jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), shapes)

    def one(item):
        en, run = item
        return lax.cond(run, lambda: kernel(study, en), lambda: nothing)

    # a sequential map of real conditionals: only the unique active entries execute
    pieces, status, diag = lax.map(one, (entry, computed))
    diag = diag.reshape(n_entries, n_diag)

    def masked(x):
        return jnp.where(_expand(computed, x), x, jnp.zeros_like(x))

    pieces = jax.tree_util.tree_map(masked, pieces)
    status = jnp.where(computed, status, RefitStatus.NO_SUPPORT)
    diag = masked(diag)
    # a deduplicated entry copies the computing entry's answer
    take = jnp.where(dirty, source, own)
    copy = lambda x: x[take]  # noqa: E731
    return RefitResult(
        slices=tuple(_unflat(copy(p), count) for p in pieces),
        status=_unflat(copy(status), count).astype(I64),
        diagnostics=_unflat(copy(diag), count),
        computed=_unflat(computed, count),
    )


@lifting_jit
def _masked_call(kernel, study, n_diag, requests):
    return lax.map(lambda r: _masked_lane(kernel, study, n_diag, r), requests)


def _pad_lanes(requests, lanes):
    """Pad a batch of worlds to `lanes` with lanes that refit nothing."""
    have = requests.dirty.shape[0]
    pad = lanes - have
    if pad == 0:
        return requests
    grown = jax.tree_util.tree_map(
        lambda x: jnp.concatenate([x, jnp.repeat(x[:1], pad, axis=0)]), requests
    )
    keep = jnp.arange(lanes) < have
    return dataclasses.replace(grown, dirty=grown.dirty & keep[:, None, None])


class MaskedBatchRefit(AbstractRefitBoundary):
    """Array kernels: a fixed batch of `refit_lanes` world lanes, one jitted call.

    `kernel(study, entry)` is a pure array function of one `RefitEntry` returning
    `(slices, status, diagnostics)`; it defaults to the observer's own refit
    (`observer_refit_kernel`). Entries that are not dirty, or that a deduplicated entry
    shares, are masked and returned unchanged (zeros). A batch of more worlds than
    lanes runs as several calls of the one compiled program, so both executors run the
    same program: the host executor's single world fills lane 0.
    """

    kind = "masked_batch"

    def __init__(self, kernel=None, *, n_diag=1, model_hash=0, version=1):
        """Bind a kernel; `n_diag` is the width of its diagnostics."""
        self.kernel = observer_refit_kernel if kernel is None else kernel
        self.n_diag = n_diag
        self.model_hash = model_hash
        self.version = version

    def refit_batch(self, study, requests):
        """Run the batch through lanes of `Capacities.refit_lanes` worlds."""
        lanes = study.caps.refit_lanes
        if lanes <= 0:
            raise ValueError("a masked batch refit needs Capacities.refit_lanes > 0")
        n_worlds = requests.dirty.shape[0]
        parts = []
        for start in range(0, n_worlds, lanes):
            stop = start + lanes
            group = jax.tree_util.tree_map(lambda x, a=start, b=stop: x[a:b], requests)
            have = group.dirty.shape[0]
            out = _masked_call(
                self.kernel, study, self.n_diag, _pad_lanes(group, lanes)
            )
            parts.append(jax.tree_util.tree_map(lambda x, n=have: x[:n], out))
        return jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *parts)


class HostLoopRefit(AbstractRefitBoundary):
    """Host Python: a loop over frozen worlds and dirty entries.

    `fit(study, entry)` takes a `RefitEntry` of numpy arrays and returns `(slices,
    status, diagnostics)`, where `slices` is a tuple of arrays in `subject_leaves`
    order, or `None` for a status that installs nothing. A failure is a status, never an
    exception. An entry that shares another's computation is copied, not fitted.
    """

    kind = "host_loop"

    def __init__(self, fit, *, n_diag=1, model_hash=0, version=1):
        """Bind a fit function; `n_diag` is the width of its diagnostics."""
        self.fit = fit
        self.n_diag = n_diag
        self.model_hash = model_hash
        self.version = version

    def refit_batch(self, study, requests):
        """Fit every dirty entry that computes for itself, then copy the shared ones."""
        req = jax.tree_util.tree_map(np.asarray, requests)
        n_worlds, count = req.dirty.shape[0], study.caps.n_subjects
        spec = slice_spec(study)
        slices = [np.zeros((n_worlds, 2, count, *s), d) for _, s, d in spec]
        status = np.full((n_worlds, 2, count), int(RefitStatus.NO_SUPPORT), np.int64)
        diag = np.zeros((n_worlds, 2, count, self.n_diag))
        computed = np.zeros((n_worlds, 2, count), bool)
        for w in range(n_worlds):
            for a in range(2):
                for s in range(count):
                    if not req.dirty[w, a, s]:
                        continue
                    source = int(req.source[w, a, s])
                    if source != a * count + s:
                        continue
                    pick = lambda x, w=w, a=a, s=s: x[w, a, s]  # noqa: E731
                    entry = RefitEntry(
                        subject=np.asarray(s, np.int64),
                        rows=jax.tree_util.tree_map(pick, req.rows),
                        key=pick(req.keys),
                        row_keys=pick(req.row_keys),
                        n_rows=pick(req.n_rows),
                    )
                    given, code, numbers = self.fit(study, entry)
                    status[w, a, s] = int(code)
                    diag[w, a, s] = np.asarray(numbers, float).reshape(self.n_diag)
                    computed[w, a, s] = True
                    if given is not None:
                        for piece, value in zip(slices, given, strict=True):
                            piece[w, a, s] = np.asarray(value)
            for a in range(2):
                for s in range(count):
                    source = int(req.source[w, a, s])
                    if req.dirty[w, a, s] and source != a * count + s:
                        sa, ss = divmod(source, count)
                        for piece in slices:
                            piece[w, a, s] = piece[w, sa, ss]
                        status[w, a, s] = status[w, sa, ss]
                        diag[w, a, s] = diag[w, sa, ss]
        return RefitResult(
            slices=tuple(jnp.asarray(p) for p in slices),
            status=jnp.asarray(status),
            diagnostics=jnp.asarray(diag),
            computed=jnp.asarray(computed),
        )
