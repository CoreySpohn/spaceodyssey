"""The observer kit: four checks an observer passes before its refit is trusted.

- `check_calibration`: the probability integral transforms (PITs) of the truth under
  the observer's posterior are uniform within an equivalence margin, over worlds
  batched in one run.
- `check_folded_once`: no world sets `dup_fold`, and a release delivered a second
  time leaves both beliefs unchanged and sets it.
- `check_promotion_parity`: after each of the first 1 to `n_log` rows the observer
  folded, installing the refit of that row's subject into the incremental belief
  leaves the belief unchanged: within the parity tolerances for an exact observer,
  within the declared margin for a statistical one.
- `check_declared_margins`: a statistical observer declares its margin, seeds and
  Monte Carlo standard error as class attributes (`AbstractObserver.statistical`).

Every check takes a batched `RunResult` (a leading world axis, as
`CompiledExecutor.run` and `HostExecutor.run_worlds` return it) and returns a
`CheckResult`. A check refuses, with `ValueError`, an input it cannot judge: too few
draws for the margin, an observer that cannot refit, a statistical observer without
its declarations, a run with nothing to compare.

Equivalence is a confidence interval inside a declared margin, never a difference
that failed to reach significance. A statistical check uses N = (4 sigma / margin)^2
draws at least, so the margin is four standard errors wide; its 95 percent interval
then lies inside the margin unless the estimate is more than about two standard
errors from the reference.

**Row order in promotion parity.** The refit of a prefix reads its rows in release
order, the order in which the incremental folds ran. An observer whose belief
depends on the order of its rows (a sampler whose belief keeps the draw of its
latest fold) then matches its incremental belief exactly, so the check isolates
whether `refit` and `fold` compute the same update. `records.refit_rows` hands a
refresh the rows in canonical order (channel, acquisition index, revision); for an
order-sensitive observer that refit is another sample of the same posterior, not the
incremental belief. The check therefore also refits every prefix in canonical order
and reports, without failing, whether that agrees too (`canonical_equal` in the
result's detail).
"""

import dataclasses
import itertools
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import lax

from spaceodyssey.core import handlers
from spaceodyssey.core.battery.result import make_result, tree_bits_equal
from spaceodyssey.core.enums import I64, Audience, EventKind, RecordStatus, Stage
from spaceodyssey.core.events import empty_event_row, where_tree
from spaceodyssey.core.executors.base import GO_TOLERANCE, index_world
from spaceodyssey.core.keys import fold_keys
from spaceodyssey.core.records import masked_record, record_none
from spaceodyssey.core.snapshot import named_leaves

# two-sided 95 percent normal quantile
Z_95 = 1.959963984540054
# standard deviations of the two PIT statistics under uniformity: the PIT itself,
# and the indicator of the central half [1/4, 3/4]
PIT_SD = {"mean": 1.0 / math.sqrt(12.0), "central half": 0.5}


class ParityMargin(NamedTuple):
    """What a statistical observer declares: margin, refit seeds, standard error."""

    margin: float
    seeds: tuple[int, ...]
    standard_error: float


def declared_margin(observer):
    """The `ParityMargin` of a statistical observer, or None for an exact one.

    Refuses (`ValueError`) a statistical observer that lacks the margin, the seeds
    or the standard error, declares a nonpositive or non-finite one, repeats a
    seed, or sizes its margin below four standard errors; and an exact observer
    that declares any of them.
    """
    margin, seeds, se = observer.margin, observer.seeds, observer.standard_error
    if not observer.statistical:
        if margin is not None or seeds or se is not None:
            raise ValueError(
                "an exact observer declares no margin, seeds or standard error; "
                "set statistical = True to compare within a margin"
            )
        return None
    missing = [
        name
        for name, value in (
            ("margin", margin),
            ("seeds", seeds),
            ("standard_error", se),
        )
        if value is None or (name == "seeds" and not value)
    ]
    if missing:
        raise ValueError(
            f"a statistical observer must declare {', '.join(missing)} as class "
            "attributes"
        )
    if not all(math.isfinite(v) and v > 0 for v in (margin, se)):
        raise ValueError("margin and standard_error must be positive and finite")
    seeds = tuple(int(s) for s in seeds)
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct nonnegative integers")
    if margin < 4.0 * se:
        raise ValueError(
            f"a margin of {margin} is narrower than four standard errors "
            f"({4.0 * se}); size the run as N = (4 sigma / margin)^2"
        )
    return ParityMargin(float(margin), seeds, float(se))


def check_declared_margins(observer):
    """Check 4: an observer's parity declaration is complete and consistent."""
    try:
        declared = declared_margin(observer)
    except ValueError as err:
        return make_result("declared_margins", [str(err)], statistical=True)
    detail = {} if declared is None else declared._asdict()
    return make_result(
        "declared_margins", [], statistical=declared is not None, **detail
    )


def calibration_size(margin):
    """PITs needed to resolve `margin`: N = (4 sigma / margin)^2, sigma = 1/2."""
    return math.ceil((4.0 * max(PIT_SD.values()) / margin) ** 2)


def check_calibration(run, truths, pit, *, margin):
    """Check 1: PITs of the truth are uniform within `margin`, over a batch of worlds.

    `pit(belief, truth)` returns `(F64[n_subjects], B[n_subjects])`: each subject's
    posterior CDF at its true value, and whether the subject's PIT counts (it has
    data). It is mapped over the world axis of `run.carry.belief` and `truths`. Two
    statistics are compared with their value under uniformity, 1/2: the mean PIT
    (a biased posterior moves it) and the fraction of PITs in the central half
    [1/4, 3/4] (a posterior too narrow or too wide moves it). Each passes when its
    95 percent interval lies inside 1/2 +/- `margin`. Refuses fewer PITs than
    `calibration_size(margin)`.
    """
    values, counted = jax.vmap(pit)(run.carry.belief, truths)
    values = np.asarray(values)[np.asarray(counted)]
    n = int(values.size)
    need = calibration_size(margin)
    if n < need:
        raise ValueError(f"{n} PITs cannot resolve a margin of {margin}; need {need}")
    failures = []
    if not np.all(np.isfinite(values) & (values >= 0.0) & (values <= 1.0)):
        failures.append("a PIT is outside [0, 1] or not finite")
        return make_result("calibration", failures, n_pits=n)
    detail = {"n_pits": n, "margin": margin}
    inside = ((values >= 0.25) & (values <= 0.75)).astype(np.float64)
    for name, x in (("mean", values), ("central half", inside)):
        estimate = float(x.mean())
        half = Z_95 * float(x.std(ddof=1)) / math.sqrt(n)
        detail[name] = estimate
        if abs(estimate - 0.5) + half > margin:
            failures.append(
                f"{name} PIT statistic {estimate:.4f}: its 95 percent interval "
                f"[{estimate - half:.4f}, {estimate + half:.4f}] is not inside "
                f"0.5 +/- {margin}"
            )
    return make_result("calibration", failures, **detail)


def double_fold_failures(carry):
    """`dup_fold` set in any world of a carry, batched or not, as failure strings."""
    flags = np.asarray(carry.dup_fold).reshape(-1)
    if flags.any():
        return [f"dup_fold is set (world {int(np.argmax(flags))})"]
    return []


@eqx.filter_jit
def _deliver_again(carry, study, truth, root, slot):
    """Handle a second release of log row `slot` through the transaction's step."""
    ev = dataclasses.replace(
        empty_event_row(),
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
    n_worlds = int(np.asarray(run.carry.dup_fold).size)
    return make_result("folded_once", failures, n_worlds=n_worlds, slot=slot)


def release_order(carry, trace):
    """Log slots the observer folded, in the order the run folded them.

    Walks the handled release and result steps of one world's trace with the release
    handler's rule: a row is accepted when both the event and the row address the
    observer and the row is valid and released; it is folded when it is the first
    accepted row of its acquisition.
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
    accepted, seen, order = set(), set(), []
    for slot, audience in zip(slots, audiences, strict=True):
        if not 0 <= slot < valid.shape[0]:
            raise ValueError(f"a release names log row {slot}, outside the log")
        readable = valid[slot] and status[slot] != RecordStatus.UNRELEASED
        to_observer = audience & int(rec_audience[slot]) & Audience.OBSERVER
        if not (readable and to_observer) or slot in accepted:
            continue
        accepted.add(slot)
        if int(acquisition[slot]) not in seen:
            order.append(slot)
        seen.add(int(acquisition[slot]))
    return order


def _log_rows(log, order, valid, caps):
    """Log rows `order` as a `Record` batch with `valid` replaced."""
    # `RecordLog` holds the columns of `Record` in the same order (see `refit_rows`)
    leaves = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda x: x[order], log))
    rows = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(record_none(caps)), leaves
    )
    return dataclasses.replace(rows, valid=valid)


@eqx.filter_jit
def _promotions(observer, study, log, order, n_folded, root, seeds):
    """Incremental beliefs and promoted beliefs of one world, per folded row.

    Returns the belief after each of the `n_log` positions of `order` (unchanged
    past `n_folded`), the promoted beliefs `[n_log, n_seeds]` in release order and in
    canonical order, and the subject-wise trace projections of all three.
    """
    caps = study.caps
    pos = jnp.arange(order.shape[0])
    rows = _log_rows(log, order, pos < n_folded, caps)
    keys = fold_keys(root, rows, study)
    seen = masked_record(rows, caps, Audience.OBSERVER)

    def fold(belief, xs):
        row, key = xs
        new = observer.analyze(observer.fold(belief, row, caps, key=key))
        belief = where_tree(row.valid, new, belief)
        return belief, belief

    _, incremental = lax.scan(fold, observer.init(caps), (seen, keys))
    counts = (
        rows.valid
        & jnp.isin(rows.kind, jnp.asarray(caps.refit_kinds, I64))
        & jnp.isin(rows.status, jnp.asarray(caps.refit_statuses, I64))
    )
    canonical = jnp.lexsort((rows.revision, rows.acquisition_index, rows.channel))

    def promote(k, perm, seed):
        subject = rows.subject[k]
        take = counts & (pos <= k) & (rows.subject == subject)
        picked = jax.tree_util.tree_map(
            lambda x: x[perm], dataclasses.replace(seen, valid=take)
        )
        key = jr.fold_in(jr.fold_in(root, seed), subject)
        fitted = observer.refit(observer.init(caps), picked, caps, key, keys[perm])
        current = jax.tree_util.tree_map(lambda x: x[k], incremental)
        return observer.analyze(observer.install(current, fitted, subject))

    def per_row(perm):
        each_seed = jax.vmap(promote, in_axes=(None, None, 0))
        return jax.vmap(each_seed, in_axes=(0, None, None))(pos, perm, seeds)

    release, reordered = per_row(pos), per_row(canonical)
    project = observer.layout(caps).project
    projections = (
        jax.vmap(project)(incremental)[0],
        jax.vmap(jax.vmap(project))(release)[0],
    )
    return incremental, release, reordered, projections, rows.subject


def _leaf_failure(name, a, b, statistical):
    """The largest disagreement of two leaves, or None within the parity tolerance.

    Integers must be equal. Floats must agree within `GO_TOLERANCE` (NaN only where
    both are NaN), unless the observer is statistical, whose floats are judged by
    the projection margin instead.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "biu":
        bad = a != b
        if bad.any():
            return f"{name} differs by {int(np.abs(a.astype(np.int64) - b).max())}"
        return None
    if statistical:
        return None
    diff = np.where(np.isnan(a) & np.isnan(b), 0.0, np.abs(a - b))
    worst = float(np.nanmax(np.where(np.isnan(diff), np.inf, diff), initial=0.0))
    return None if worst <= GO_TOLERANCE else f"{name} differs by {worst:.3g}"


def _tree_failure(a, b, statistical=False):
    """The first leaf of two beliefs that disagrees, or None."""
    la, lb = named_leaves(a), named_leaves(b)
    for name in la:
        failure = _leaf_failure(name, la[name], lb[name], statistical)
        if failure:
            return failure
    return None


def check_promotion_parity(study, run, roots, *, observer=None):
    """Check 3: the incremental belief equals the refit installed into it.

    For each world and each prefix of the rows the observer folded (release order,
    see the module docstring), the subject of the prefix's last row is refit from the
    prefix's rows that count (`caps.refit_kinds`, `caps.refit_statuses`) with their
    fold keys, installed into the incremental belief after that prefix, and
    analyzed. An exact observer must reproduce the incremental belief within the
    parity tolerances. A statistical observer must reproduce its integer leaves
    exactly; the difference of its trace projection at the installed subject,
    averaged per world over the prefixes and its declared refit seeds, must have a
    95 percent interval inside its margin across worlds, with the standard error
    taken as the larger of the observed one and the declared one.

    `observer` defaults to the study's; another observer may be judged on the same
    run when its folds are the study observer's. The kit's incremental path is
    anchored to the run: after the last folded row it must equal the run's final
    belief within the parity tolerances, or the check fails.

    Refuses an observer that cannot refit or whose refit kinds the study's
    capacities do not hold, a statistical observer without its declarations or with
    fewer than two worlds, and a run in which no world folded a row.
    """
    observer = study.observer if observer is None else observer
    caps = study.caps
    declared = declared_margin(observer)
    if not observer.capabilities().refit or not caps.refit_kinds:
        raise ValueError("promotion parity needs a refit-capable observer, lowered")
    seeds = (0,) if declared is None else declared.seeds
    statistical = declared is not None
    n_worlds = int(np.asarray(run.carry.tick).shape[0])
    if statistical and n_worlds < 2:
        raise ValueError("a statistical observer is judged over two worlds at least")
    failures, per_world, n_checked, canonical_equal = [], [], 0, True
    batch = jax.device_get((run.carry, run.trace))
    for w in range(n_worlds):
        carry, trace = index_world(batch, w)
        folded = release_order(carry, trace)
        n = len(folded)
        if n == 0:
            continue
        order = np.zeros(caps.n_log, np.int64)
        order[:n] = folded
        incremental, release, reordered, (p_inc, p_rel), subjects = jax.device_get(
            _promotions(
                observer,
                study,
                carry.log,
                jnp.asarray(order),
                jnp.asarray(n, I64),
                roots[w],
                jnp.asarray(seeds, I64),
            )
        )
        anchor = _tree_failure(index_world(incremental, n - 1), carry.belief)
        if anchor:
            failures.append(f"world {w}: the folds do not reproduce the run: {anchor}")
            break
        for k, j in itertools.product(range(n), range(len(seeds))):
            current = index_world(incremental, k)
            got = index_world(index_world(release, k), j)
            failure = _tree_failure(got, current, statistical)
            if failure:
                failures.append(
                    f"world {w}: after {k + 1} row{'s' if k else ''}, the refit of "
                    f"subject {int(subjects[k])} installed: {failure}"
                )
                break
            canonical_equal &= (
                _tree_failure(index_world(index_world(reordered, k), j), current)
                is None
            )
            n_checked += 1
        if failures:
            break
        s = subjects[:n]
        diffs = p_rel[np.arange(n), :, s] - p_inc[np.arange(n), s][:, None]
        per_world.append(float(diffs.mean()))
    if n_checked == 0 and not failures:
        raise ValueError("no world folded a row; promotion parity is vacuous")
    detail = {
        "n_worlds": n_worlds,
        "n_checked": n_checked,
        "canonical_equal": canonical_equal,
        "statistical": statistical,
    }
    if statistical and not failures:
        failures += _margin_failures(np.asarray(per_world), declared, detail)
    return make_result("promotion_parity", failures, **detail)


def _margin_failures(per_world, declared, detail):
    """The projection-difference interval of a statistical observer, judged."""
    estimate = float(per_world.mean())
    observed = float(per_world.std(ddof=1)) / math.sqrt(per_world.size)
    se = max(observed, declared.standard_error)
    detail.update(estimate=estimate, observed_se=observed, margin=declared.margin)
    if abs(estimate) + Z_95 * se > declared.margin:
        return [
            f"projection difference {estimate:.4g} (standard error {se:.3g}) has a "
            f"95 percent interval outside +/- {declared.margin}"
        ]
    return []
