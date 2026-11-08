"""The observer kit: six checks an observer passes before its refit is trusted.

- `check_calibration`: the empirical distribution of the probability integral
  transforms (PITs) of the truth under the observer's posterior lies within an
  equivalence margin of the uniform one, over worlds batched in one run.
- `check_folded_once`: no world sets `dup_fold`, and a release delivered a second
  time leaves both beliefs unchanged and sets it.
- `check_promotion_parity`: after each of the first 1 to `n_log` rows the observer
  folded, installing the refit of that row's subject into the incremental belief
  leaves the belief unchanged, for the rows in release order and for the rows as
  `records.refit_rows` selects them; and `install` writes the refit subject's
  declared state and nothing else. `check_refit_calibration` applies the
  calibration check to the refit beliefs, for refits judged only in mean there.
- `check_declared_margins`: an observer judged within a margin declares the margin,
  its seeds, its Monte Carlo standard error and its stochastic leaves as class
  attributes (`AbstractObserver.statistical`, `AbstractObserver.order_sensitive`).
- `check_delayed_correction`: after a revision released later than the row it
  supersedes, the belief of the corrected subject equals a flat-prior refit of the
  corrected rows alone, so the superseded row is not kept.
- `check_derived_summary`: a derived summary of folded rows adds nothing to the
  belief, so the belief equals a refit that counts each datum once.
- `check_overlap_declared` (observers declaring `reads_priors`): per audience, no
  retained prior and an evidence row of its pedigree both carry weight, retained
  priors have disjoint pedigrees, and a weight of zero is justified only by evidence
  released to that audience. `check_zero_weight_rows` extends `check_folded_once` to
  such observers: a row of weight zero moves neither initial belief nor a fold, and
  `check_folded_once` runs it too.

Every refit the kit makes is keyed by `keys.refit_key(root, subject, digest)`, where
the digest names the revision set the refit reads, never a fold count.

Every check takes a batched `RunResult` (a leading world axis, as
`CompiledExecutor.run` and `HostExecutor.run_worlds` return it) and returns a
`CheckResult`. A check refuses, with `ValueError`, an input it cannot judge: too few
worlds for the margin, an observer that cannot refit, an observer without its
declarations, a run with nothing to compare. `replay_beliefs` gives the beliefs
another observer would hold over the same records, so one batch of worlds serves
several observers.

Equivalence is a confidence statement inside a declared margin, never a difference
that failed to reach significance. Worlds are the independent units: each check
draws its uncertainty from variation between worlds, never from several values of
one world, which may share randomness.

**Row order in promotion parity.** Two refits of each prefix are judged. The first
reads the prefix's rows in release order, the order in which the incremental folds
ran; every observer must reproduce its incremental belief from it within the parity
tolerances (or, for a statistical observer, within its margin), so the check
isolates whether `refit` and `fold` compute the same update. The second reads the
rows `records.refit_rows` selects from the prefix's log, in canonical order
(channel, acquisition index, revision) with the counting rows first, which is what
a refresh installs. An order-insensitive observer must reproduce its incremental
belief from that too, within the tolerances. An order-sensitive observer (a sampler
whose belief keeps the draw of its latest fold) declares `order_sensitive`: its
canonical refit is another sample of the same posterior, so only its declared
stochastic leaves are exempt from exact comparison there, and its trace projection
is judged within its margin. The detail entry `canonical_equal` reports whether the
canonical refit also matched exactly.
"""

import dataclasses
import math
import operator
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax

from spaceodyssey.core import handlers
from spaceodyssey.core.battery.result import make_result, tree_bits_equal
from spaceodyssey.core.enums import (
    I64,
    Audience,
    Capacity,
    EventKind,
    RecordStatus,
    RunStatus,
    Stage,
)
from spaceodyssey.core.events import empty_event_row, where_tree
from spaceodyssey.core.executors.base import GO_TOLERANCE, index_world
from spaceodyssey.core.keys import fold_keys, refit_key
from spaceodyssey.core.lift import lifting_jit
from spaceodyssey.core.records import (
    derived_rows,
    log_records,
    masked_record,
    record_none,
    refit_rows,
)
from spaceodyssey.core.refresh import revision_set_digest
from spaceodyssey.core.snapshot import named_leaves

# confidence level of every equivalence statement
ALPHA = 0.05
# two-sided 95 percent normal quantile
Z_975 = 1.959963984540054
# fewest worlds a margin is judged over: the t quantile below is then accurate
MIN_WORLDS = 10


def t_quantile(df):
    """The two-sided 95 percent Student t quantile with `df` degrees of freedom.

    The Cornish-Fisher expansion in 1 / df (Abramowitz and Stegun 26.7.5), within
    1e-3 of the exact value from 9 degrees of freedom, the fewest the kit uses.
    """
    z = Z_975
    terms = (
        (z**3 + z) / 4.0,
        (5 * z**5 + 16 * z**3 + 3 * z) / 96.0,
        (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384.0,
        (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160.0,
    )
    return z + sum(g / df ** (i + 1) for i, g in enumerate(terms))


def _positive(value):
    """Whether `value` is a positive, finite real number."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _integer(value):
    """`value` as an int if it is an integer (not a bool), else None."""
    if isinstance(value, bool):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


class ParityMargin(NamedTuple):
    """What a statistical observer declares: margin, refit seeds, standard error."""

    margin: float
    seeds: tuple[int, ...]
    standard_error: float


def declared_margin(observer):
    """The `ParityMargin` of an observer judged within a margin, or None.

    An observer is judged within a margin when it is `statistical` or
    `order_sensitive`. Refuses (`ValueError`) such an observer that lacks the
    margin, the seeds or the standard error, declares a nonpositive or non-finite
    one, repeats a seed, or sizes its margin below four standard errors; and an
    exact observer that declares any of them.
    """
    margin, seeds, se = observer.margin, observer.seeds, observer.standard_error
    stochastic = observer.stochastic_leaves
    if not (observer.statistical or observer.order_sensitive):
        if margin is not None or seeds or se is not None or stochastic:
            raise ValueError(
                "an exact observer declares no margin, seeds, standard error or "
                "stochastic leaves; set statistical or order_sensitive to compare "
                "within a margin"
            )
        return None
    missing = [
        name
        for name, value in (
            ("margin", margin),
            ("seeds", seeds),
            ("standard_error", se),
            ("stochastic_leaves", stochastic),
        )
        if value is None or (name in ("seeds", "stochastic_leaves") and not value)
    ]
    if missing:
        raise ValueError(
            f"an observer judged within a margin must declare {', '.join(missing)} "
            "as class attributes"
        )
    if not (_positive(margin) and _positive(se)):
        raise ValueError("margin and standard_error must be positive and finite")
    seeds = tuple(_integer(s) for s in seeds)
    if None in seeds:
        raise ValueError("seeds must be integers")
    if min(seeds) < 0 or max(seeds) >= 2**32 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct integers in [0, 2^32)")
    if margin < 4.0 * se:
        raise ValueError(
            f"a margin of {margin} is narrower than four standard errors "
            f"({4.0 * se}); size the run as N = (4 sigma / margin)^2"
        )
    return ParityMargin(float(margin), seeds, float(se))


def check_declared_margins(observer):
    """Check 4: an observer's margin declaration is complete and consistent."""
    try:
        declared = declared_margin(observer)
    except ValueError as err:
        return make_result("declared_margins", [str(err)], judged_in_margin=True)
    detail = {} if declared is None else declared._asdict()
    return make_result(
        "declared_margins", [], judged_in_margin=declared is not None, **detail
    )


def calibration_size(margin, n_slots):
    """Worlds per PIT slot that make the DKW slack at most half of `margin`."""
    if not _positive(margin):
        raise ValueError(f"a margin must be positive and finite, not {margin}")
    return math.ceil(2.0 * math.log(2.0 * n_slots / ALPHA) / margin**2)


def _ecdf_distance(values):
    """Kolmogorov distance between the empirical CDF of `values` and the uniform."""
    x = np.sort(values)
    i = np.arange(1, x.size + 1)
    return float(max(np.max(i / x.size - x), np.max(x - (i - 1) / x.size)))


def check_calibration(run, truths, pit, *, margin):
    """Check 1: the PITs of the truth are uniform within `margin`, over a batch.

    `pit(belief, truth)` returns `(F64[m], B[m])`: one posterior CDF value at a
    true value per PIT slot (normally a subject), and whether the slot counts. It is
    mapped over the world axis of `run.carry.belief` and `truths`.

    The check bounds the Kolmogorov distance between the true distribution of the
    pooled PITs and the uniform one: the empirical distance `D` plus a slack, which
    must stay within `margin`. Slots of one world may share randomness, so the slack
    is built from worlds, the independent units: by the Dvoretzky-Kiefer-Wolfowitz
    inequality the empirical CDF of slot j over its N_j worlds lies within
    sqrt(log(2 m / alpha) / (2 N_j)) of its true CDF for all m slots at once with
    probability 1 - alpha, and the pooled empirical CDF is the N_j-weighted mean of
    the slot CDFs, so the slack is the weighted mean of those widths. `D + slack <=
    margin` then says, at 95 percent confidence, that the PITs are within `margin` of
    uniform. Refuses a slack wider than half the margin (`calibration_size` gives the
    worlds needed).
    """
    if not _positive(margin):
        raise ValueError(f"a margin must be positive and finite, not {margin}")
    values, counted = jax.vmap(pit)(run.carry.belief, truths)
    return _calibration(values, counted, margin)


def _calibration(values, counted, margin):
    """Judge world-by-slot PITs with the world-clustered DKW slack."""
    values, counted = np.asarray(values), np.asarray(counted, dtype=bool)
    per_slot = counted.sum(axis=0)
    used = per_slot > 0
    n_slots = int(used.sum())
    if n_slots == 0:
        raise ValueError("no PIT counts; calibration is vacuous")
    n_pits = int(per_slot.sum())
    widths = np.sqrt(np.log(2.0 * n_slots / ALPHA) / (2.0 * per_slot[used]))
    slack = float(np.sum(per_slot[used] / n_pits * widths))
    if slack > margin / 2.0:
        raise ValueError(
            f"a DKW slack of {slack:.4f} cannot resolve a margin of {margin}; need "
            f"{calibration_size(margin, n_slots)} worlds per PIT slot"
        )
    pooled = values[counted]
    detail = {"n_pits": n_pits, "n_slots": n_slots, "slack": slack, "margin": margin}
    if not np.all(np.isfinite(pooled) & (pooled >= 0.0) & (pooled <= 1.0)):
        return make_result(
            "calibration", ["a PIT is outside [0, 1] or not finite"], **detail
        )
    distance = _ecdf_distance(pooled)
    detail["distance"] = distance
    failures = []
    if distance + slack > margin:
        failures.append(
            f"the PITs are {distance:.4f} from uniform; with the {slack:.4f} slack "
            f"that is not inside the margin {margin}"
        )
    return make_result("calibration", failures, **detail)


def double_fold_failures(carry):
    """`dup_fold` set in any world of a carry, batched or not, as failure strings."""
    flags = np.asarray(carry.dup_fold).reshape(-1)
    if flags.any():
        return [f"dup_fold is set (world {int(np.argmax(flags))})"]
    return []


@lifting_jit
def _deliver_again(carry, study, truth, root, slot):
    """Handle a second release of log row `slot` through the transaction's step."""
    ev = dataclasses.replace(
        empty_event_row(study.caps),
        kind=jnp.asarray(EventKind.RELEASE, I64),
        tick=carry.tick,
        stage=jnp.asarray(Stage.RELEASE, I64),
        slot=jnp.asarray(slot, I64),
        audience=carry.log.audience[slot],
        valid=jnp.asarray(True),
    )
    return handlers.step(carry, ev, study, truth, root)[0]


def check_folded_once(study, run, truths, roots):
    """Check 2: no record is folded twice.

    Fails if any world sets `dup_fold`, or if delivering the release of an accepted
    row a second time (the first row of world 0 that both audiences received, else
    the first the observer accepted) changes the belief or the policy summary, or
    leaves `dup_fold` clear. Refuses a run whose world 0 accepted no row.
    """
    failures = double_fold_failures(run.carry)
    carry = index_world(run.carry, 0)
    log = carry.log
    accepted = np.asarray(log.valid) & (np.asarray(log.accepted_tick) >= 0)
    both = accepted & (np.asarray(log.released_tick) >= 0)
    if not accepted.any():
        raise ValueError("world 0 accepted no row to deliver again")
    slot = int(np.argmax(both if both.any() else accepted))
    again = _deliver_again(
        carry, study, index_world(truths, 0), roots[0], jnp.asarray(slot, I64)
    )
    if not tree_bits_equal(again.belief, carry.belief):
        failures.append(f"a second release of log row {slot} changed the belief")
    if not tree_bits_equal(again.pbelief, carry.pbelief):
        failures.append(f"a second release of log row {slot} changed the summary")
    if not bool(again.dup_fold):
        failures.append(f"a second release of log row {slot} left dup_fold clear")
    if getattr(study.observer, "reads_priors", False):
        failures += _zero_weight_failures(study, study.observer)[0]
    n_worlds = int(np.asarray(run.carry.dup_fold).size)
    return make_result("folded_once", failures, n_worlds=n_worlds, slot=slot)


def _walk(carry, trace, audience):
    """Slots folded for `audience` in fold order, with the ticks each was folded with.

    Mirrors the release handler over the handled release and result steps of one
    world's trace: a row reaches the audience when both the event and the row
    address it and the row is valid and released, and is folded when it is the
    first row of its acquisition to reach the audience. The handler reads the row
    before it stamps `accepted_tick` and `released_tick`, so each fold sees the
    stamps of earlier steps only; those are returned beside the slots.
    """
    log = carry.log
    valid, status = np.asarray(log.valid), np.asarray(log.status)
    rec_audience, acquisition = np.asarray(log.audience), np.asarray(log.acquisition_id)
    kind = np.asarray(trace.kind)
    steps = np.asarray(trace.handled) & (
        (kind == EventKind.RELEASE) | (kind == EventKind.RESULT)
    )
    slots = np.asarray(trace.log_slot)[steps].tolist()
    audiences = np.asarray(trace.audience)[steps].tolist()
    ticks = np.asarray(trace.tick)[steps].tolist()
    accepted = np.full(valid.shape[0], -1, np.int64)
    released = np.full(valid.shape[0], -1, np.int64)
    reached = {Audience.OBSERVER: set(), Audience.POLICY: set()}
    order, stamps = [], []
    for slot, to, tick in zip(slots, audiences, ticks, strict=True):
        if not 0 <= slot < valid.shape[0]:
            raise ValueError(f"a release names log row {slot}, outside the log")
        readable = valid[slot] and status[slot] != RecordStatus.UNRELEASED
        to_observer = readable and bool(
            to & int(rec_audience[slot]) & Audience.OBSERVER
        )
        public = readable and bool(to & int(rec_audience[slot]) & Audience.POLICY)
        reaches = to_observer if audience == Audience.OBSERVER else public
        first = (accepted if audience == Audience.OBSERVER else released)[slot] < 0
        acq = int(acquisition[slot])
        if reaches and first and acq not in reached[audience]:
            order.append(slot)
            stamps.append((int(accepted[slot]), int(released[slot])))
        if reaches:
            reached[audience].add(acq)
        if to_observer and accepted[slot] < 0:
            accepted[slot] = tick
        if public:
            released[slot] = tick
    return order, stamps


def release_order(carry, trace, audience=Audience.OBSERVER):
    """Log slots folded for `audience`, in the order the run folded them."""
    return _walk(carry, trace, audience)[0]


def _orders(carry, trace, audience):
    """Fold orders and fold-time stamps of every world of a batch, padded.

    Returns `order I64[W, n_log]`, `count I64[W]` and the `accepted_tick` and
    `released_tick` each folded row carried when it was folded, `I64[W, n_log]`.
    """
    n_worlds, n_log = np.asarray(carry.log.valid).shape
    orders = np.zeros((n_worlds, n_log), np.int64)
    counts = np.zeros(n_worlds, np.int64)
    stamps = np.full((2, n_worlds, n_log), -1, np.int64)
    for w in range(n_worlds):
        slots, seen = _walk(index_world(carry, w), index_world(trace, w), audience)
        orders[w, : len(slots)] = slots
        counts[w] = len(slots)
        if seen:
            stamps[:, w, : len(seen)] = np.asarray(seen).T
    return orders, counts, stamps[0], stamps[1]


def _log_rows(log, order, valid, caps):
    """Log rows `order` as a `Record` batch with `valid` replaced."""
    rows = log_records(jax.tree_util.tree_map(lambda x: x[order], log))
    return dataclasses.replace(rows, valid=valid)


def _presented_rows(log, order, n, stamps, caps):
    """The rows `order[:n]` as the release handler presented them to a fold."""
    accepted, released = stamps
    rows = _log_rows(log, order, jnp.arange(order.shape[0]) < n, caps)
    if caps.derived_kinds:
        # a derived row is presented to no fold
        rows = dataclasses.replace(rows, valid=rows.valid & ~derived_rows(rows, caps))
    return dataclasses.replace(rows, accepted_tick=accepted, released_tick=released)


def _fold_path(rows, root, study, audience, step, start):
    """`start` with the valid `rows` folded by `step(belief, row, key)` in order."""
    keys = fold_keys(root, rows, study)
    seen = masked_record(rows, study.caps, audience)

    def body(belief, xs):
        row, key = xs
        belief = where_tree(row.valid, step(belief, row, key), belief)
        return belief, belief

    return lax.scan(body, start, (seen, keys))


@lifting_jit
def _replay(observer, study, logs, own, policy, roots):
    caps = study.caps

    def one(log, order, n, accepted, released, porder, pn, paccepted, preleased, root):
        rows = _presented_rows(log, order, n, (accepted, released), caps)
        belief = _fold_path(
            rows,
            root,
            study,
            Audience.OBSERVER,
            lambda b, r, k: observer.analyze(observer.fold(b, r, caps, key=k)),
            observer.init(caps),
        )[0]
        prows = _presented_rows(log, porder, pn, (paccepted, preleased), caps)
        summary = _fold_path(
            prows,
            root,
            study,
            Audience.POLICY,
            lambda b, r, k: observer.fold_summary(b, r, caps, key=k),
            observer.init_summary(caps),
        )[0]
        return belief, summary

    return jax.vmap(one)(logs, *own, *policy, roots)


def replay_beliefs(study, run, roots, observer=None):
    """`(belief, summary)` that `observer` would hold over a batched run's records.

    Each world's rows are folded in release order for each audience, as the release
    handler folds them: the full belief takes the rows the observer accepted
    (`fold`, then `analyze`), the summary the rows released to the policy
    (`fold_summary`), each with its fold key and with the `accepted_tick` and
    `released_tick` it carried when the handler folded it. This is the run of
    `observer` only when the records do not depend on the observer (no policy or
    process reads the belief); with the run's own observer it reproduces the run's
    beliefs. Refuses a log with a revised acquisition, whose `superseded_by` at
    each fold the final log does not hold.
    """
    observer = study.observer if observer is None else observer
    carry, trace = jax.device_get((run.carry, run.trace))
    revised = np.asarray(carry.log.valid) & (np.asarray(carry.log.revision) > 0)
    if revised.any():
        raise ValueError("replay_beliefs does not replay revised acquisitions")
    own = _orders(carry, trace, Audience.OBSERVER)
    policy = _orders(carry, trace, Audience.POLICY)
    return _replay(observer, study, run.carry.log, own, policy, roots)


def _shifted(tree):
    """Invert booleans and shift numeric values enough that each differs."""

    def shift(x):
        if x.dtype == jnp.bool_:
            return ~x
        one = jnp.ones((), x.dtype)
        if jnp.issubdtype(x.dtype, jnp.integer):
            return x + one
        return x + jnp.maximum(one, jnp.abs(x))

    return jax.tree_util.tree_map(shift, tree)


def _seeded(root, seed):
    """The root of a declared seed: the run's own root at seed 0, another one else."""
    return jnp.where(seed == 0, root, jr.fold_in(root, seed))


def _set_key(root, seed, subject, rows):
    """The refit key of a revision set, as the refresh protocol derives it.

    `rows` are the set's rows in canonical order (`revision_set_digest` hashes them in
    the order they come in), so one set gets one key whichever order it was gathered.
    """
    return refit_key(_seeded(root, seed), subject, revision_set_digest(rows))


def _refit_subject(observer, study, log, root, seed, subject):
    """`refit` of `subject` from `refit_rows` of `log`, keyed by `seed`."""
    caps = study.caps
    rows = refit_rows(log, caps, Audience.OBSERVER, subject)
    key = _set_key(root, seed, subject, rows)
    return observer.refit(
        observer.init(caps), rows, caps, key, fold_keys(root, rows, study)
    )


@lifting_jit
def _promotions(
    observer, study, logs, orders, counts, accepted, released, roots, seeds
):
    """Incremental, promoted and install-challenge beliefs of a batch of worlds.

    Per world: the belief after each of the `n_log` positions of `order` (unchanged
    past `n`); per position and seed, the refit of the position's subject installed
    into that belief and analyzed, once from the prefix's rows in release order and
    once from `refit_rows` of the prefix's log; and, before analysis, the release
    refit installed into a shifted copy of the belief, with that copy and the refit.
    """
    caps = study.caps
    layout = observer.layout(caps)

    def one(log, order, n, accepted, released, root):
        pos = jnp.arange(order.shape[0])
        rows = _presented_rows(log, order, n, (accepted, released), caps)
        keys = fold_keys(root, rows, study)
        seen = masked_record(rows, caps, Audience.OBSERVER)
        _, incremental = _fold_path(
            rows,
            root,
            study,
            Audience.OBSERVER,
            lambda b, r, k: observer.analyze(observer.fold(b, r, caps, key=k)),
            observer.init(caps),
        )
        counting = (
            rows.valid
            & jnp.isin(rows.kind, jnp.asarray(caps.refit_kinds, I64))
            & jnp.isin(rows.status, jnp.asarray(caps.refit_statuses, I64))
        )

        def promote(k, seed):
            subject = rows.subject[k]
            current = jax.tree_util.tree_map(lambda x: x[k], incremental)
            take = counting & (pos <= k) & (rows.subject == subject)
            first = jnp.argsort(~take, stable=True)
            picked = jax.tree_util.tree_map(
                lambda x: x[first], dataclasses.replace(seen, valid=take)
            )
            # the key names the set, not the release order the rows came in
            in_order = jnp.lexsort(
                (
                    picked.revision,
                    picked.acquisition_index,
                    picked.channel,
                    ~picked.valid,
                )
            )
            key = _set_key(
                root,
                seed,
                subject,
                jax.tree_util.tree_map(lambda x: x[in_order], picked),
            )
            fitted = observer.refit(observer.init(caps), picked, caps, key, keys[first])
            released_ = observer.analyze(observer.install(current, fitted, subject))
            in_prefix = (
                jnp.zeros(log.valid.shape, bool).at[order].max((pos <= k) & (pos < n))
            )
            prefix = dataclasses.replace(
                log, accepted_tick=jnp.where(in_prefix, log.accepted_tick, -1)
            )
            canon = _refit_subject(observer, study, prefix, root, seed, subject)
            canonical = observer.analyze(observer.install(current, canon, subject))
            shifted = _shifted(current)
            installed = observer.install(shifted, fitted, subject)
            return released_, canonical, (installed, shifted, fitted)

        each = jax.vmap(jax.vmap(promote, in_axes=(None, 0)), in_axes=(0, None))
        promoted, canonical, challenge = each(pos, seeds)
        projections = (
            jax.vmap(layout.project)(incremental)[0],
            jax.vmap(jax.vmap(layout.project))(promoted)[0],
            jax.vmap(jax.vmap(layout.project))(canonical)[0],
        )
        return incremental, promoted, canonical, challenge, projections

    return jax.vmap(one)(logs, orders, counts, accepted, released, roots)


def _mismatch(got, want, valid, loose=()):
    """`(index, leaf)` of the first disagreement of two batched beliefs, or None.

    `valid` masks the leading axes of `got`; `want` broadcasts against `got`.
    Integers must be equal. Floats must agree within `GO_TOLERANCE` (NaN only where
    both are NaN), except the leaves named in `loose`, which need only be finite
    wherever `want` is.
    """
    lead = valid.ndim
    named_want = named_leaves(want)
    for name, g in named_leaves(got).items():
        g, w = np.asarray(g), np.asarray(named_want[name])
        if g.dtype.kind in "biu":
            bad = g != w
        elif name in loose:
            bad = ~np.isfinite(g) & np.isfinite(w)
        else:
            bad = ~(np.abs(g - w) <= GO_TOLERANCE) & ~(np.isnan(g) & np.isnan(w))
        bad = np.broadcast_to(bad, np.broadcast_shapes(g.shape, w.shape))
        bad = bad.reshape(*bad.shape[:lead], -1).any(axis=-1) & valid
        if bad.any():
            return tuple(int(i) for i in np.argwhere(bad)[0]), name
    return None


def _expand(tree, axis):
    """Insert a unit axis in every leaf, so a belief broadcasts against its seeds."""
    return jax.tree_util.tree_map(lambda x: np.expand_dims(np.asarray(x), axis), tree)


def _row_label(where, subjects, seeds):
    w, k, j = where
    rows = f"{k + 1} row{'s' if k else ''}"
    return f"world {w}: after {rows}, subject {int(subjects[w, k])}, seed {seeds[j]}"


def _leaf_roles(observer, caps):
    """The subject-owned and stochastic leaf names, checked against the layout.

    Refuses a refit-capable observer that does not declare `subject_leaves`, a
    subject leaf whose leading axis is not `n_subjects` long, and a stochastic
    leaf that is not a floating-point leaf of the full belief.
    """
    leaves = named_leaves(observer.layout(caps).belief)
    owned = observer.subject_leaves
    if owned is None:
        raise ValueError("a refit-capable observer declares its subject_leaves")
    for name in owned:
        if name not in leaves or leaves[name].shape[:1] != (caps.n_subjects,):
            raise ValueError(
                f"subject leaf {name!r} is not a belief leaf with a leading axis "
                f"of {caps.n_subjects} subjects"
            )
    for name in observer.stochastic_leaves:
        if name not in leaves or not jnp.issubdtype(leaves[name].dtype, jnp.floating):
            raise ValueError(f"stochastic leaf {name!r} is not a float belief leaf")
    return set(owned), set(observer.stochastic_leaves)


def _same(a, b):
    """Elementwise equality, with NaN equal to NaN."""
    return (a == b) | (np.isnan(a) & np.isnan(b)) if a.dtype.kind == "f" else a == b


def _install_failures(challenge, owned, subjects, valid, seeds):
    """Install must write the refit subject's owned state and nothing else.

    Before analysis, every subject-owned leaf must hold the refit's values at the
    refit subject and the shifted belief's values at every other subject, and every
    other leaf must keep the shifted belief's values.
    """
    installed, shifted, fitted = (named_leaves(t) for t in challenge)
    every = valid[:, :, None]
    for name, got in installed.items():
        got, before, refit = (np.asarray(t[name]) for t in (installed, shifted, fitted))
        if name not in owned:
            bad = ~_same(got, before).reshape(*got.shape[:3], -1).all(-1) & every
            if bad.any():
                where = tuple(int(i) for i in np.argwhere(bad)[0])
                return [
                    f"{_row_label(where, subjects, seeds)}: install changed the "
                    f"shared leaf {name}"
                ]
            continue
        n_subjects = got.shape[3]
        mine = np.arange(n_subjects) == subjects[:, :, None, None]
        mine = mine.reshape(mine.shape + (1,) * (got.ndim - 4))
        kept = _same(got, before) | mine
        took = _same(got, refit) | ~mine
        for ok, what in (
            (took, "did not take the refit's"),
            (kept, "changed another subject's"),
        ):
            bad = ~ok.reshape(*got.shape[:3], -1).all(-1) & every
            if bad.any():
                where = tuple(int(i) for i in np.argwhere(bad)[0])
                return [f"{_row_label(where, subjects, seeds)}: install {what} {name}"]
    return []


def _stratum_failures(label, promoted, incremental, subjects, valid, declared):
    """Per (subject, depth) stratum, the projection difference judged in the margin.

    The difference at the refit subject is averaged over seeds within each world;
    each stratum's worlds give a t interval, with the standard error the larger of
    the observed and the declared one, that must lie inside the margin. A stratum
    with fewer than `MIN_WORLDS` worlds, or a non-finite difference, fails.
    """
    n_rows = subjects.shape[1]
    take = subjects[:, :, None, None]
    diff = np.take_along_axis(promoted, take, axis=-1)[..., 0]
    diff = (diff - np.take_along_axis(incremental, take[:, :, 0], axis=-1)).mean(-1)
    depth = np.stack(
        [
            (subjects[:, : k + 1] == subjects[:, k : k + 1]).sum(1)
            for k in range(n_rows)
        ],
        axis=1,
    )
    strata = {}
    for w, k in np.argwhere(valid):
        strata.setdefault((int(subjects[w, k]), int(depth[w, k])), []).append(
            diff[w, k]
        )
    failures, worst = [], 0.0
    for (subject, rows), values in sorted(strata.items()):
        values = np.asarray(values)
        name = f"{label}: subject {subject} after {rows} of its rows"
        if values.size < MIN_WORLDS:
            failures.append(f"{name} has {values.size} worlds; {MIN_WORLDS} needed")
            continue
        if not np.all(np.isfinite(values)):
            failures.append(f"{name}: a projection difference is not finite")
            continue
        estimate = float(values.mean())
        observed = float(values.std(ddof=1)) / math.sqrt(values.size)
        half = t_quantile(values.size - 1) * max(observed, declared.standard_error)
        worst = max(worst, abs(estimate) + half)
        if abs(estimate) + half > declared.margin:
            failures.append(
                f"{name}: projection difference {estimate:.4g} +/- {half:.3g} is "
                f"not inside +/- {declared.margin}"
            )
    return failures, worst


def check_promotion_parity(study, run, roots, *, observer=None):
    """Check 3: the incremental belief equals the refit installed into it.

    For each world and each prefix of the rows the observer folded, the subject of
    the prefix's last row is refit and installed into the incremental belief after
    that prefix, then analyzed, twice (see the module docstring on row order):

    - from the prefix's counting rows (`caps.refit_kinds`, `caps.refit_statuses`)
      in release order, counting rows first, with their fold keys;
    - from `refit_rows` of the prefix's log.

    Every leaf must match the incremental belief, integers exactly and floats within
    the parity tolerances, with one exception: the observer's declared
    `stochastic_leaves` need only be finite in the release-order path when it is
    `statistical`, and in the `refit_rows` path when it is `statistical` or
    `order_sensitive`. A path with loose leaves is judged in the observer's margin
    instead: per stratum (the refit subject and how many of its rows the prefix
    holds), the trace-projection difference at the refit subject, averaged over the
    declared seeds within each world, gives a t interval over worlds with the larger
    of the observed and declared standard errors, inside the margin, over
    `MIN_WORLDS` worlds at least.

    `install` is challenged too: the release-order refit, installed into a copy of
    the incremental belief with booleans inverted and numeric values shifted so
    each differs, must give the refit subject the refit's values in every leaf
    the observer declares in
    `subject_leaves`, keep every other subject's values there, and keep every other
    leaf, before analysis.

    `observer` defaults to the study's; another observer may be judged on the same
    run when its folds are the study observer's. The kit's incremental path is
    anchored to the run: after the last folded row it must equal the run's final
    belief within the parity tolerances.

    Refuses an observer that cannot refit, whose refit kinds the study's capacities
    do not hold, or whose leaf declarations or margin declarations are missing or
    inconsistent, and a run in which no world folded a row.
    """
    observer = study.observer if observer is None else observer
    caps = study.caps
    declared = declared_margin(observer)
    if not observer.capabilities().refit or not caps.refit_kinds:
        raise ValueError("promotion parity needs a refit-capable observer, lowered")
    owned, stochastic = _leaf_roles(observer, caps)
    seeds = (0,) if declared is None else declared.seeds
    carry, trace = jax.device_get((run.carry, run.trace))
    orders, counts, accepted, released = _orders(carry, trace, Audience.OBSERVER)
    if not counts.any():
        raise ValueError("no world folded a row; promotion parity is vacuous")
    incremental, promoted, canonical, challenge, projections = jax.device_get(
        _promotions(
            observer,
            study,
            run.carry.log,
            jnp.asarray(orders),
            jnp.asarray(counts),
            jnp.asarray(accepted),
            jnp.asarray(released),
            roots,
            jnp.asarray(seeds, I64),
        )
    )
    n_rows = orders.shape[1]
    valid = np.arange(n_rows)[None, :] < counts[:, None]
    subjects = np.take_along_axis(np.asarray(carry.log.subject), orders, axis=1)
    worlds = np.flatnonzero(counts)
    last = jax.tree_util.tree_map(lambda x: x[worlds, counts[worlds] - 1], incremental)
    ran = jax.tree_util.tree_map(lambda x: x[worlds], carry.belief)
    anchor = _mismatch(last, ran, np.ones(worlds.size, bool))
    if anchor:
        (i,), leaf = anchor
        return make_result(
            "promotion_parity",
            [f"world {int(worlds[i])}: the folds do not reproduce the run: {leaf}"],
        )
    statistical, ordered = observer.statistical, observer.order_sensitive
    by_seed = valid[:, :, None] & np.ones(len(seeds), bool)
    current = _expand(incremental, 2)
    failures = []
    paths = (
        ("release order", promoted, statistical),
        ("refit_rows", canonical, statistical or ordered),
    )
    for label, beliefs, loose in paths:
        found = _mismatch(beliefs, current, by_seed, stochastic if loose else ())
        if found:
            where, leaf = found
            failures.append(
                f"{label}: {_row_label(where, subjects, seeds)}: the installed "
                f"refit's {leaf} differs from the incremental belief"
            )
    failures += _install_failures(challenge, owned, subjects, valid, seeds)
    detail = {
        "n_worlds": int(counts.size),
        "n_checked": int(by_seed.sum()),
        "canonical_equal": _mismatch(canonical, current, by_seed) is None,
        "statistical": statistical,
        "order_sensitive": ordered,
    }
    p_inc, p_rel, p_can = projections
    for (label, _, loose), promoted_p in zip(paths, (p_rel, p_can), strict=True):
        if loose:
            found, worst = _stratum_failures(
                label, promoted_p, p_inc, subjects, valid, declared
            )
            failures += found
            detail[f"{label} worst"] = worst
    return make_result("promotion_parity", failures, **detail)


@lifting_jit
def _refit_all(observer, study, beliefs, logs, roots, seed):
    """Every subject of each world refit from `refit_rows`, installed, analyzed."""

    def one(belief, log, root):
        for subject in range(study.caps.n_subjects):
            fitted = _refit_subject(observer, study, log, root, seed, subject)
            belief = observer.install(belief, fitted, jnp.asarray(subject, I64))
        return observer.analyze(belief)

    return jax.vmap(one)(beliefs, logs, roots)


def refit_beliefs(study, run, roots, *, observer=None, seed=0):
    """The beliefs a refresh of every subject would install at the end of a run.

    Each subject of each world is refit from `refit_rows` of the final log, with
    the refit key of `seed`, and installed into the run's belief, which is then
    analyzed: what a refresh at the last tick produces.
    """
    observer = study.observer if observer is None else observer
    return _refit_all(
        observer, study, run.carry.belief, run.carry.log, roots, jnp.asarray(seed)
    )


def check_refit_calibration(study, run, truths, roots, pit, *, margin, observer=None):
    """Check 1 applied to refits at each subject row depth, per declared seed.

    Promotion parity holds a statistical refit to the incremental belief in mean
    only; this holds its draws to the truth at every canonical prefix parity
    exercises. `pit` returns one PIT slot per subject. Only the refitted subject's
    PIT counts at each prefix, pooled over subjects separately at each row depth
    under each declared seed (or seed 0). Every depth must pass the calibration
    margin, with worlds as the independent units for the DKW slack.
    """
    if not _positive(margin):
        raise ValueError(f"a margin must be positive and finite, not {margin}")
    observer = study.observer if observer is None else observer
    declared = declared_margin(observer)
    seeds = (0,) if declared is None else declared.seeds
    carry, trace = jax.device_get((run.carry, run.trace))
    orders, counts, accepted, released = _orders(carry, trace, Audience.OBSERVER)
    if not counts.any():
        raise ValueError("no world folded a row; refit calibration is vacuous")
    canonical = _promotions(
        observer,
        study,
        run.carry.log,
        jnp.asarray(orders),
        jnp.asarray(counts),
        jnp.asarray(accepted),
        jnp.asarray(released),
        roots,
        jnp.asarray(seeds, I64),
    )[2]

    def one(beliefs, truth):
        return jax.vmap(jax.vmap(lambda belief: pit(belief, truth)))(beliefs)

    values, counted = (np.asarray(x) for x in jax.vmap(one)(canonical, truths))
    n_subjects = study.caps.n_subjects
    if values.shape[3:] != (n_subjects,):
        raise ValueError("refit calibration needs one PIT slot per subject")
    n_rows = orders.shape[1]
    valid = np.arange(n_rows)[None, :] < counts[:, None]
    subjects = np.take_along_axis(np.asarray(carry.log.subject), orders, axis=1)
    depth = np.stack(
        [
            (subjects[:, : k + 1] == subjects[:, k : k + 1]).sum(1)
            for k in range(n_rows)
        ],
        axis=1,
    )
    failures, detail = [], {}
    for j, seed in enumerate(seeds):
        per_depth = {}
        for rows in sorted(set(depth[valid].tolist())):
            take = (
                valid[:, :, None]
                & (depth[:, :, None] == rows)
                & (subjects[:, :, None] == np.arange(n_subjects))
            )
            # Each world contributes at most once per subject at this depth.
            pits = np.where(take, values[:, :, j], 0.0).sum(axis=1)
            used = (take & counted[:, :, j]).any(axis=1)
            result = _calibration(pits, used, margin)
            label = f"after {rows} of its rows"
            per_depth[label] = result.detail
            if not result:
                failures.append(f"seed {seed}, {label}: {result.diagnostic}")
        detail[f"seed {seed}"] = per_depth
    return make_result("refit_calibration", failures, **detail)


def _corrected_slots(log, caps, subject, exclude=()):
    """Log slots of a subject's corrected rows in canonical order.

    Written from the rule and not from `records.revision_set`, so the check does not
    trust the code it audits: among the rows the observer accepted, other than
    derived ones and the kinds in `exclude`, each acquisition keeps its highest
    revision; that row counts when its kind and status are listed for refit; the
    counting rows sort by channel, acquisition index and revision.
    """
    col = {
        "valid": np.asarray(log.valid),
        "accepted_tick": np.asarray(log.accepted_tick),
        "subject": np.asarray(log.subject),
        "kind": np.asarray(log.kind),
        "status": np.asarray(log.status),
        "acquisition_id": np.asarray(log.acquisition_id),
        "revision": np.asarray(log.revision),
        "channel": np.asarray(log.channel),
        "acquisition_index": np.asarray(log.acquisition_index),
    }
    out = tuple(caps.derived_kinds) + tuple(exclude)
    mine = (
        col["valid"]
        & (col["accepted_tick"] >= 0)
        & (col["subject"] == subject)
        & ~np.isin(col["kind"], out)
    )
    latest = {}
    for slot in np.flatnonzero(mine):
        acquisition = int(col["acquisition_id"][slot])
        held = latest.get(acquisition)
        if held is None or col["revision"][slot] > col["revision"][held]:
            latest[acquisition] = slot
    counting = [
        slot
        for slot in latest.values()
        if int(col["kind"][slot]) in caps.refit_kinds
        and int(col["status"][slot]) in caps.refit_statuses
    ]
    return sorted(
        counting,
        key=lambda s: (
            int(col["channel"][s]),
            int(col["acquisition_index"][s]),
            int(col["revision"][s]),
        ),
    )


@lifting_jit
def _flat_prior_refit(observer, study, log, root, order, count, subject):
    """The subject's leaves of a flat-prior refit of the log rows `order[:count]`.

    The rows are masked for the observer and zeroed where they do not count, the
    refit key is the refresh protocol's (`refit_key` of the set's digest), and the
    refit is installed into a fresh belief and analyzed, as a refresh installs it.
    """
    caps = study.caps
    valid = jnp.arange(order.shape[0]) < count
    rows = log_records(jax.tree_util.tree_map(lambda x: x[order], log), valid=valid)
    rows = masked_record(rows, caps, Audience.OBSERVER)
    rows = jax.tree_util.tree_map(
        lambda x: jnp.where(
            valid.reshape((-1,) + (1,) * (x.ndim - 1)), x, jnp.zeros_like(x)
        ),
        rows,
    )
    key = refit_key(root, subject, revision_set_digest(rows))
    fitted = observer.refit(
        observer.init(caps), rows, caps, key, fold_keys(root, rows, study)
    )
    belief = observer.analyze(observer.install(observer.init(caps), fitted, subject))
    return named_leaves(belief)


def _refit_capable(study):
    """The observer and the leaf names an installed refit owns, or a refusal."""
    caps = study.caps
    if caps.rows_per_subject <= 0:
        raise ValueError("the study has no refresh protocol: rows_per_subject is 0")
    observer = study.observer
    if not observer.capabilities().refit:
        raise ValueError("the observer is not refit-capable")
    _leaf_roles(observer, caps)
    return observer, observer.subject_leaves


def _unfinished(run):
    """A refusal text if a world stopped before its queue drained, else None.

    `COMPLETE` and `IDLE` both mean nothing was left to run, so no refresh
    checkpoint is pending; every other status leaves the last refresh possibly unread.
    """
    status = np.asarray(run.status)
    done = (status == RunStatus.COMPLETE) | (status == RunStatus.IDLE)
    if done.all():
        return None
    bad = sorted({int(x) for x in status[~done]})
    return f"a run stopped early (status {bad}): its last refresh may be unread"


def _refit_comparison(name, study, run, roots, picks, exclude=()):
    """Compare each world's final belief at `picks[w]` with a flat-prior refit.

    The reference is `_corrected_slots` of the final log, refit from a flat prior
    by the observer; the run's belief holds what the last refresh installed, so the
    two agree when the protocol replaced superseded rows and counted no summary.
    """
    study_caps = study.caps
    observer, owned = _refit_capable(study)
    failures, subjects_seen = [], set()
    carry = jax.device_get(run.carry)
    width = max(study_caps.rows_per_subject, study_caps.n_log)
    for w, subjects in enumerate(picks):
        log = index_world(carry.log, w)
        for subject in sorted(subjects):
            slots = _corrected_slots(log, study_caps, subject, exclude)
            if len(slots) > study_caps.rows_per_subject:
                raise ValueError(
                    f"world {w}, subject {subject}: {len(slots)} rows exceed "
                    f"rows_per_subject = {study_caps.rows_per_subject}"
                )
            order = np.zeros(width, np.int64)
            order[: len(slots)] = slots
            want = _flat_prior_refit(
                observer,
                study,
                log,
                roots[w],
                jnp.asarray(order),
                jnp.asarray(len(slots)),
                jnp.asarray(subject, I64),
            )
            have = named_leaves(index_world(carry.belief, w))
            subjects_seen.add(subject)
            for leaf in owned:
                got, ref = (
                    np.asarray(have[leaf][subject]),
                    np.asarray(want[leaf][subject]),
                )
                ok = (
                    np.array_equal(got, ref)
                    if got.dtype.kind != "f"
                    else bool(np.all(np.abs(got - ref) <= GO_TOLERANCE))
                )
                if not ok:
                    failures.append(
                        f"world {w}, subject {subject}: {leaf} is {got.tolist()} but "
                        f"a flat-prior refit of its {len(slots)} corrected rows gives "
                        f"{ref.tolist()}"
                    )
    return make_result(
        name, failures, n_worlds=len(picks), subjects=sorted(subjects_seen)
    )


def check_delayed_correction(study, run, roots):
    """Check 5: a revision released after acceptance replaces the row it corrects.

    `run` is a batched run of a refit-capable observer under the refresh protocol
    (a boundary bound, every world complete) whose log holds a correction: a row of
    a higher revision accepted at a later tick than the revision it supersedes. At
    the end of the run the belief of every corrected subject must equal a
    flat-prior refit of the corrected rows, each acquisition at its latest revision
    and listed status. An observer that keeps the superseded row (an `install`
    that merges into what the subject holds, a refit that reads rows the protocol
    dropped) differs from that refit and fails.
    """
    _refit_capable(study)
    if (why := _unfinished(run)) is not None:
        raise ValueError(why)
    carry = jax.device_get(run.carry)
    picks = []
    for w in range(np.asarray(carry.log.valid).shape[0]):
        log = index_world(carry.log, w)
        valid = np.asarray(log.valid) & (np.asarray(log.accepted_tick) >= 0)
        acquisition, revision = np.asarray(log.acquisition_id), np.asarray(log.revision)
        accepted, subject = np.asarray(log.accepted_tick), np.asarray(log.subject)
        late = set()
        for i in np.flatnonzero(valid & (revision > 0)):
            earlier = (
                valid
                & (acquisition == acquisition[i])
                & (revision < revision[i])
                & (accepted < accepted[i])
            )
            if earlier.any():
                late.add(int(subject[i]))
        picks.append(late)
    if not any(picks):
        raise ValueError(
            "no world holds a correction accepted after the row it corrects"
        )
    return _refit_comparison("delayed_correction", study, run, roots, picks)


def check_derived_summary(study, run, roots, *, summary_kind):
    """Check 6: a derived summary of folded rows adds nothing to the belief.

    `summary_kind` names the record kind of a summary computed from rows the
    observer also folded (a pedigree overlapping the revision set). Each world must
    hold at least one accepted row of that kind. The belief of every subject with a
    summary must equal a flat-prior refit that counts each datum once: the summary
    kind is left out of the reference whether or not the study registers it as
    derived, so a summary registered as an ordinary kind and counted by the observer
    fails. A summary is a new acquisition: the core refuses a correction that reuses
    a derived acquisition id (`Capacity.INVALID_CORRECTION`), and a run that hit
    that refusal is refused here, because its log is not the one the study sent.
    """
    _refit_capable(study)
    names = study.caps.kind_names
    if summary_kind not in names:
        raise ValueError(
            f"no record kind named {summary_kind!r}; the study has {names}"
        )
    kind = names.index(summary_kind)
    carry = jax.device_get(run.carry)
    if np.any(np.asarray(carry.overflow_mask) & int(Capacity.INVALID_CORRECTION)):
        raise ValueError(
            "a summary was refused as a correction: "
            "give each summary its own acquisition"
        )
    if (why := _unfinished(run)) is not None:
        raise ValueError(why)
    picks = []
    for w in range(np.asarray(carry.log.valid).shape[0]):
        log = index_world(carry.log, w)
        held = (
            np.asarray(log.valid)
            & (np.asarray(log.accepted_tick) >= 0)
            & (np.asarray(log.kind) == kind)
        )
        picks.append({int(x) for x in np.asarray(log.subject)[held]})
    if not all(picks):
        raise ValueError(f"a world holds no accepted {summary_kind!r} row to judge")
    return _refit_comparison(
        "derived_summary", study, run, roots, picks, exclude=(kind,)
    )


# ---- observers that read imported priors --------------------------------------------

AUDIENCES = (int(Audience.OBSERVER), int(Audience.POLICY))


def _prior_reader(study, observer):
    """The observer to judge, or a refusal unless it follows the prior protocol.

    An observer that declares `reads_priors` holds the imported table as `table`
    (columns `source`, `pedigree`, `n_pedigree`, `kind`, `audience`, `channel`,
    `local_index`, `payload`; a namespace of -1 pads) and reports the weight in
    {0, 1} of each row for an audience as `weights(caps, audience)`; `init` and
    `init_summary` are its two initial beliefs.
    """
    observer = study.observer if observer is None else observer
    if not getattr(observer, "reads_priors", False):
        raise ValueError("the observer does not declare reads_priors")
    if getattr(observer, "table", None) is None or not callable(
        getattr(observer, "weights", None)
    ):
        raise ValueError("an observer that reads priors holds a table and weights")
    return observer


class _Roles(NamedTuple):
    """One audience's view of an imported table, computed here from its columns."""

    prior: np.ndarray  # [N] a row of a derived kind released to the audience
    evidence: np.ndarray  # [N] a raw row released to the audience
    ids: list  # [N] the overlap identity of each row's source
    named: list  # [N] the set of identities each row's pedigree names


def _roles(table, caps, audience):
    """The table's priors, evidence and identities for `audience`, independently."""
    source, pedigree = table.source, table.pedigree
    cols = (source.namespace, source.producer, source.subject, source.source_index)
    ids = [tuple(int(v) for v in row) for row in np.stack(cols, axis=-1)]
    cols = (pedigree.namespace, pedigree.producer, pedigree.subject)
    grid = np.stack((*cols, pedigree.source_index), axis=-1)
    count = np.asarray(table.n_pedigree)
    named = [
        {tuple(int(v) for v in grid[i, j]) for j in range(int(count[i]))}
        for i in range(len(ids))
    ]
    present = np.asarray(source.namespace) != -1
    derived = np.isin(np.asarray(table.kind), np.asarray(caps.derived_kinds, int))
    released = present & ((np.asarray(table.audience) & audience) != 0)
    return _Roles(released & derived, released & ~derived, ids, named)


def _weights(observer, caps, audience, n_rows):
    """The observer's weights for `audience` as booleans, or a failure text.

    The weights must be a vector of one finite value in {0, 1} per table row; a NaN,
    a fraction or another shape is refused before any comparison reads it.
    """
    w = np.asarray(observer.weights(caps, audience))
    ok = w.shape == (n_rows,) and bool(np.all(np.isin(w, (0, 1))))
    if not ok:
        return None, (
            f"audience {audience}: the weights are not a {{0, 1}} vector of one entry "
            f"per row (shape {w.shape}, values {np.unique(w).tolist()[:4]})"
        )
    return w == 1, None


def check_overlap_declared(study, observer=None):
    """Check 7: the weights an observer resolves for each audience are sound.

    For each audience (observer, then policy), with the observer's own weights
    (`observer.weights(caps, audience)`) and the table it holds (`observer.table`):
    no retained prior (weight one) and an evidence row of its pedigree both carry
    weight; no two retained priors name a common source (external entries count);
    and a weight of zero is justified by that audience's own evidence: a dropped
    prior names one of its evidence rows, a dropped evidence row is named by one of
    its priors. Weights that are not a finite {0, 1} vector of one entry per row
    fail. Refuses an observer that does not declare `reads_priors`, and a table with
    no prior for either audience (the check would pass without judging anything).
    """
    observer = _prior_reader(study, observer)
    caps, table = study.caps, observer.table
    failures, judged = [], 0
    n_rows = len(np.asarray(table.kind))
    roles_of = {a: _roles(table, caps, a) for a in AUDIENCES}
    if not any(roles.prior.any() for roles in roles_of.values()):
        raise ValueError("the table holds no prior for either audience to judge")
    for audience in AUDIENCES:
        roles = roles_of[audience]
        w, bad = _weights(observer, caps, audience, n_rows)
        if bad:
            failures.append(bad)
            continue
        priors = np.flatnonzero(roles.prior)
        evidence = np.flatnonzero(roles.evidence)
        seen = {roles.ids[r] for r in evidence}
        judged += len(priors)
        for p in priors:
            if not w[p]:
                if not roles.named[p] & seen:
                    failures.append(
                        f"audience {audience}: prior {p} is dropped, but its "
                        "pedigree names no evidence that audience receives"
                    )
                continue
            for r in evidence:
                if w[r] and roles.ids[r] in roles.named[p]:
                    failures.append(
                        f"audience {audience}: retained prior {p} and evidence row "
                        f"{r} of its pedigree both carry weight"
                    )
        for r in evidence:
            named = any(roles.ids[r] in roles.named[p] for p in priors)
            if not w[r] and not named:
                failures.append(
                    f"audience {audience}: evidence row {r} is dropped, but no prior "
                    "of that audience names it"
                )
        kept = [int(p) for p in priors if w[p]]
        for i, p in enumerate(kept):
            for q in kept[i + 1 :]:
                if roles.named[p] & roles.named[q]:
                    failures.append(
                        f"audience {audience}: retained priors {p} and {q} have "
                        "intersecting pedigrees"
                    )
    return make_result("overlap_declared", failures, n_priors=judged)


def _imported_row(table, i, caps, audience):
    """Raw row `i` of the table as the import writes it and `audience` folds it.

    The metadata are the import's: the row's own audience, its span from the offsets
    relative to its release at tick 0 and availability at tick 0. The release folds
    a row before it stamps it, so `accepted_tick` and `released_tick` stay -1, as
    the observer sees them in `fold` and `fold_summary`. The payload is masked for
    `audience`.
    """
    base = record_none(caps)
    width = np.shape(table.payload)[1]
    addressed = int(table.audience[i])
    zero, unset = jnp.asarray(0, I64), jnp.asarray(-1, I64)
    rec = dataclasses.replace(
        base,
        kind=jnp.asarray(table.kind[i], I64),
        subject=jnp.asarray(table.source.subject[i], I64),
        channel=jnp.asarray(table.channel[i], I64),
        acquisition_index=jnp.asarray(table.local_index[i], I64),
        revision=jnp.asarray(table.source.revision[i], I64),
        audience=jnp.asarray(addressed, I64),
        start_tick=jnp.asarray(table.start_offset[i], I64),
        end_tick=jnp.asarray(table.end_offset[i], I64),
        available_tick=zero,
        status=jnp.asarray(RecordStatus.COMPLETED, I64),
        accepted_tick=unset,
        released_tick=unset,
        superseded_by=unset,
        payload=base.payload.at[:width].set(table.payload[i]),
        valid=jnp.asarray(True),
    )
    return masked_record(rec, caps, audience)


def _zero_weight_failures(study, observer):
    """`(failures, probes)`: zero-weight rows that move a belief; tries made."""
    caps, table = study.caps, observer.table
    failures, probes = [], 0
    n_rows = len(np.asarray(table.kind))
    for audience in AUDIENCES:
        kept, bad = _weights(observer, caps, audience, n_rows)
        if bad:
            failures.append(bad)
            continue
        w = kept.astype(int)
        roles = _roles(table, caps, audience)
        policy = audience == int(Audience.POLICY)
        start = observer.init_summary(caps) if policy else observer.init(caps)
        gone = np.flatnonzero(roles.prior & (w == 0))
        probes += int(gone.size > 0)
        if gone.size:
            namespace = jnp.asarray(table.source.namespace).at[gone].set(-1)
            without = eqx.tree_at(
                lambda o: o.table.source.namespace, observer, namespace
            )
            other = without.init_summary(caps) if policy else without.init(caps)
            if not tree_bits_equal(other, start):
                failures.append(
                    f"audience {audience}: priors {gone.tolist()} of weight zero move "
                    "the initial belief"
                )
        for r in np.flatnonzero(roles.evidence & (w == 0)):
            probes += 1
            rec = _imported_row(table, int(r), caps, audience)
            if policy:
                folded = observer.fold_summary(start, rec, caps)
            else:
                folded = observer.analyze(observer.fold(start, rec, caps))
            if not tree_bits_equal(folded, start):
                failures.append(
                    f"audience {audience}: evidence row {r} of weight zero moves the "
                    "belief when folded"
                )
    return failures, probes


def check_zero_weight_rows(study, observer=None):
    """Check 2 for observers that read priors: a row of weight zero moves nothing.

    For each audience, removing the priors of weight zero leaves the initial belief
    (`init` for the observer, `init_summary` for the policy) unchanged bit for bit,
    and folding an imported evidence row of weight zero, as that audience receives
    it, leaves the belief unchanged. Weights that are not a finite {0, 1} vector
    fail. Refuses a table with no row of weight zero for either audience: there is
    nothing to remove or fold, so the check would pass without trying. Needs no run;
    `check_folded_once` applies it to such observers as well (where a table with
    nothing to probe adds no failure).
    """
    observer = _prior_reader(study, observer)
    failures, probes = _zero_weight_failures(study, observer)
    if probes == 0 and not failures:
        raise ValueError("the table holds no row of weight zero to probe")
    return make_result("zero_weight_rows", failures, n_probes=probes)
