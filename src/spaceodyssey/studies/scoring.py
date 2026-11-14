"""Host measurement folds and scorer-only joins of tracks to catalog identities."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.adapters.measurement import (
    MEASUREMENT_KIND_NAME,
    MeasurementRow,
    participating_entries,
    usable_measurement,
)
from spaceodyssey.adapters.parameters import (
    CATALOG_KIND_NAME,
    catalog_template,
    scale_au2,
)
from spaceodyssey.core.enums import F64, I64, RecordStatus
from spaceodyssey.core.kinds import AbstractObjective
from spaceodyssey.core.records import unpack
from spaceodyssey.criteria.transition import TransitionTable

MISASSOCIATED = "misassociated"
DUPLICATE = "duplicate"
MISSED = "missed"
RowLog = dict[str, np.ndarray]


def _completed(log, kind):
    return (
        log.valid
        & (log.kind == kind)
        & (
            (log.status == RecordStatus.COMPLETED)
            | (log.status == RecordStatus.PARTIAL)
        )
    )


def row_log(result, study) -> RowLog:
    """Unpack completed and partial measurements into NumPy [rows, ...] columns.

    Rows are ordered by world and completion tick, stably for coincident rows.
    Channel is the admission channel index; mode and duration_index are the
    measurement's declared indices. Empty logs preserve every trailing shape.
    Status is retained so science folds can reject unusable partials while
    activity accounting retains their occupied time.
    """
    log = result.carry.log
    template = study.channel_model.measurement_template
    fields = dataclasses.fields(MeasurementRow)
    columns = {name: [] for name in ("world", "star", "end_tick", "channel", "status")}
    columns.update({f.name: [] for f in fields})
    valid = np.asarray(
        _completed(log, study.caps.kind_names.index(MEASUREMENT_KIND_NAME))
    )
    for world in range(valid.shape[0]):
        indices = np.flatnonzero(valid[world])
        indices = indices[
            np.argsort(np.asarray(log.end_tick)[world, indices], kind="stable")
        ]
        for r in indices:
            row = unpack(template, log.payload[world, r])
            for name, value in (
                ("world", world),
                ("star", log.subject[world, r]),
                ("end_tick", log.end_tick[world, r]),
                ("channel", log.channel_index[world, r]),
                ("status", log.status[world, r]),
            ):
                columns[name].append(value)
            for f in fields:
                columns[f.name].append(np.asarray(getattr(row, f.name)))
    out = {
        name: np.asarray(columns[name], np.int64)
        for name in ("world", "star", "end_tick", "channel", "status")
    }
    for f in fields:
        leaf = getattr(template, f.name)
        out[f.name] = np.asarray(columns[f.name], dtype=leaf.dtype).reshape(
            (-1, *leaf.shape)
        )
    return out


_update = eqx.filter_jit(
    lambda criterion, state, row, modes: criterion.update(state, row, modes)
)
_value = eqx.filter_jit(
    lambda criterion, state, claims, tick: criterion.value(state, claims, tick)
)


def refold(criterion, rows, world, star, modes, columns="measured"):
    """Pure host reference fold of one star in stable completion order.

    `columns="true"` substitutes true counts and variance, leaving assignments
    intact. The detection width in the row schema is the fixed track capacity.
    """
    if columns not in ("measured", "true"):
        raise ValueError("columns must be 'measured' or 'true'")
    state = criterion.init_state(rows["assignment"].shape[1], rows["counts"].shape[2])
    indices = np.flatnonzero((rows["world"] == world) & (rows["star"] == star))
    indices = indices[np.argsort(rows["end_tick"][indices], kind="stable")]
    for i in indices:
        values = {
            f.name: jnp.asarray(rows[f.name][i])
            for f in dataclasses.fields(MeasurementRow)
        }
        if not bool(
            usable_measurement(
                rows["status"][i], values["valid_science"], values["variance"]
            )
        ):
            continue
        if columns == "true":
            values.update(
                counts=values["true_counts"], variance=values["true_variance"]
            )
        state = _update(criterion, state, MeasurementRow(**values), modes)
    return state


def _catalogs(carry, study):
    """Last completed catalog per star, plus a presence-of-row mask."""
    channel = study.channel_model
    valid = _completed(carry.log, study.caps.kind_names.index(CATALOG_KIND_NAME))
    stars = jnp.arange(study.caps.n_subjects)
    mine = valid[None, :] & (carry.log.subject[None, :] == stars[:, None])
    indices = jnp.argmax(jnp.where(mine, carry.log.end_tick[None, :], -1), axis=1)
    catalog = jax.vmap(
        lambda payload: unpack(
            catalog_template(channel.n_slots, channel.n_bands), payload
        )
    )(carry.log.payload[indices])
    return catalog, mine.any(-1)


def _truth_claims(criterion, params, luminosity_lsun):
    if not criterion.claims:
        return jnp.zeros((*params.ln_period_d.shape, 0), F64)
    return jnp.stack(
        [c.evaluate(params, luminosity_lsun) for c in criterion.claims], -1
    ).astype(F64)


def _truth_value(criterion, state, claims, tick):
    """Score the declared answer against truth independently of its utility.

    A transition table's target predicate must hold to certify its achieved
    answer. Other criteria express their answers through their value method.
    """
    values, labels = _value(criterion, state, claims, tick)
    if isinstance(criterion, TransitionTable) and criterion.target_claim is not None:
        labels = jnp.where(criterion._target_probability(claims) > 0, labels, -1)
    return values, labels


def _track_join(seen, present):
    """A track must name exactly one present identity across all its rows."""
    n_slots = present.size
    identities = jnp.argmax(seen[:, :n_slots], axis=-1)
    joined = (seen.sum(-1) == 1) & seen[:, :n_slots].any(-1) & present[identities]
    return identities, joined


def _credit(values, labels, identities, joined, n_slots):
    """One highest-value track per true identity; ties go to the first track."""
    matches = joined[:, None] & (identities[:, None] == jnp.arange(n_slots))
    scores = jnp.where(matches, values[:, None], -1.0)
    winners = jnp.argmax(scores, axis=0)
    credited = matches.any(0)
    unique = jnp.any(
        (jnp.arange(n_slots)[:, None] == winners[None, :]) & credited[None, :], axis=1
    )
    return (
        jnp.where(credited, values[winners], 0.0),
        jnp.where(credited, labels[winners], -1),
        unique,
    )


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Counts and credit [W], with mission and scorer labels [W, n, S].

    Labels address tracks. Counts and yield credit each catalog identity once,
    using the highest-value consistent track. A mixed or absent identity is
    misassociated; extra consistent tracks are duplicates; an unjoined present
    catalog identity is missed. `wrong` counts mission-labeled tracks with a
    wrong label, false target predicate, misassociation or duplication.
    Mission labels index the observer criterion's labels; scorer labels index
    the scoring criterion's labels. Wrong-label comparisons use label names.

    `reason` uses the policy's last fresh post-kickoff selection snapshot:
    `no_package` for no feasible positive-gain variant, `below_price` for a
    package omitted by selection, and empty for selected tracks. A truth join
    outside belief support overrides either open reason with `unsupported`.
    Before any selection refresh, only the truth-side override is available.
    """

    counts: dict[str, np.ndarray]
    mission_label: np.ndarray
    scorer_label: np.ndarray
    yield_: np.ndarray
    wrong: np.ndarray
    reason: np.ndarray


def verdict(result, study, criterion=None) -> Verdict:
    """Score true rows with criterion, preserving the observer's mission labels.

    The optional criterion overrides truth scoring only. Retained mission state
    and claim columns are always evaluated with study.observer.criterion.
    """
    observer_criterion = study.observer.criterion
    criterion = observer_criterion if criterion is None else criterion
    mission_names = np.asarray((*observer_criterion.labels, ""))
    scorer_names = np.asarray((*criterion.labels, ""))
    rows = row_log(result, study)
    n_worlds, n_stars = result.carry.belief.p_exist.shape[:2]
    channel, n_slots = study.channel_model, study.channel_model.n_slots
    shape = (n_worlds, n_stars, n_slots)
    mission, scorer = np.full(shape, -1, np.int64), np.full(shape, -1, np.int64)
    reason = np.full(shape, "", dtype="U11")
    counts = {
        name: np.zeros(n_worlds, np.float64)
        for name in (*criterion.labels, MISASSOCIATED, DUPLICATE, MISSED)
    }
    yield_, wrong = np.zeros(n_worlds), np.zeros(n_worlds)
    modes = channel.mode_table
    population = study.decision_view().belief_population
    for w in range(n_worlds):
        carry = jax.tree.map(lambda x, w=w: x[w], result.carry)
        catalogs, has_catalog = _catalogs(carry, study)
        policy = study.agenda.policies[0]
        selection = unpack(policy.state_layout(study.caps), carry.pstate)
        for star in range(n_stars):
            present = np.asarray(catalogs.present[star]) & bool(has_catalog[star])
            seen = np.zeros((n_slots, n_slots + 1), bool)
            mine = (
                (rows["world"] == w)
                & (rows["star"] == star)
                & (rows["valid_science"] > 0)
                & np.asarray(
                    usable_measurement(
                        rows["status"], rows["valid_science"], rows["variance"]
                    )
                )
            )
            participates = np.asarray(
                participating_entries(
                    rows["valid_science"][mine],
                    rows["detected"][mine],
                    rows["counts"][mine],
                    rows["variance"][mine],
                    rows["mode"][mine],
                    modes,
                )
            )
            for assignment, identity, entries in zip(
                rows["assignment"][mine],
                rows["identity"][mine],
                participates,
                strict=True,
            ):
                for track, planet, entry in zip(
                    assignment, identity, entries, strict=True
                ):
                    if entry and 0 <= track < n_slots:
                        seen[track, planet if 0 <= planet < n_slots else n_slots] = True
            identities, joined = _track_join(jnp.asarray(seen), jnp.asarray(present))
            params = jax.tree.map(
                lambda x, star=star, identities=identities: x[star][identities],
                catalogs.params,
            )
            claims = _truth_claims(
                criterion, params, channel.scenes.luminosity_lsun[star]
            )
            state = refold(criterion, rows, w, star, modes, columns="true")
            if criterion.labels:
                values, labels = _truth_value(criterion, state, claims, carry.tick)
            else:
                values, labels = jnp.zeros(n_slots, F64), jnp.full(n_slots, -1, I64)
            if observer_criterion.labels:
                observed = unpack(
                    observer_criterion.state_template(n_slots, channel.n_bands),
                    carry.belief.state[star],
                )
                _, mission[w, star] = _value(
                    observer_criterion, observed, carry.belief.p_claim[star], carry.tick
                )
            labels = jnp.where(joined, labels, -1)
            scorer[w, star] = np.asarray(labels)
            credit, credited_labels, unique = _credit(
                values, labels, identities, joined, n_slots
            )
            yield_[w] += float(credit.sum())
            for i, name in enumerate(criterion.labels):
                counts[name][w] += np.count_nonzero(np.asarray(credited_labels) == i)
            misassociated = seen.any(-1) & ~np.asarray(joined)
            duplicates = np.asarray(joined & ~unique)
            counts[MISASSOCIATED][w] += misassociated.sum()
            counts[DUPLICATE][w] += duplicates.sum()
            reached = np.zeros(n_slots, bool)
            reached[np.asarray(identities)[np.asarray(joined)]] = True
            counts[MISSED][w] += (present & ~reached).sum()
            wrong[w] += np.count_nonzero(
                (mission[w, star] >= 0)
                & (
                    misassociated
                    | duplicates
                    | (mission_names[mission[w, star]] != scorer_names[scorer[w, star]])
                )
            )
            if hasattr(selection, "selection_tick") and int(selection.n_trace) > 0:
                open_tracks = mission[w, star] < 0
                has_package = np.asarray(selection.has_package[star])
                selected = np.asarray(selection.selected[star])
                reason[w, star, open_tracks & ~has_package] = "no_package"
                reason[w, star, open_tracks & has_package & ~selected] = "below_price"
            support = population.support
            unsupported = np.asarray(joined) & np.asarray(
                [
                    not support.contains(float(p), float(e))
                    for p, e in zip(
                        jnp.exp(params.ln_period_d), params.ecc, strict=True
                    )
                ]
            )
            low, high = population.scale_range_au2(channel.detection_band)
            ln_scale_au2 = np.asarray(
                jnp.log(scale_au2(params, channel.imaging_band_index))
            )
            unsupported |= np.asarray(joined) & (
                ~np.isfinite(ln_scale_au2)
                | (ln_scale_au2 < low)
                | (ln_scale_au2 > high)
            )
            reason[w, star, unsupported & (mission[w, star] < 0)] = "unsupported"
    return Verdict(counts, mission, scorer, yield_, wrong, reason)


def cross_yield(result, study, criterion) -> np.ndarray:
    """Yield F64[W] of a flight's true rows counted under criterion.

    Native criteria retain the verdict even without measurements. Foreign
    transition tables require, in each world, a row in at least one allowed
    mode of every step. Other criterion families require every named mode.
    Uncovered worlds receive NaN. Coverage uses completed measurement rows,
    independently of whether their counts pass a requirement.
    """
    if eqx.tree_equal(criterion, study.observer.criterion):
        return verdict(result, study).yield_
    rows = row_log(result, study)
    n_worlds = result.carry.belief.p_exist.shape[0]
    names = tuple(m.name for m in study.channel_model.mode_table.modes)
    observed = np.zeros((n_worlds, len(criterion.mode_names)), bool)
    usable = np.asarray(
        usable_measurement(rows["status"], rows["valid_science"], rows["variance"])
    )
    known = []
    for i, name in enumerate(criterion.mode_names):
        if name in names:
            known.append(i)
            observed[rows["world"][(rows["mode"] == names.index(name)) & usable], i] = (
                True
            )
    if isinstance(criterion, TransitionTable):
        covered = (
            (np.asarray(criterion.step_modes)[None, :, :] & observed[:, None, :])
            .any(-1)
            .all(-1)
        )
    else:
        covered = observed.all(-1)
    if not covered.any():
        return np.full(n_worlds, np.nan, np.float64)
    if isinstance(criterion, TransitionTable) and len(known) < len(
        criterion.mode_names
    ):
        criterion = dataclasses.replace(
            criterion,
            mode_names=tuple(criterion.mode_names[i] for i in known),
            step_modes=criterion.step_modes[:, jnp.asarray(known)],
        )
    return np.where(covered, verdict(result, study, criterion).yield_, np.nan)


@dataclasses.dataclass(frozen=True)
class Attribution:
    """Paired yield changes F64[W] from the definition and planning response.

    Missing cross coverage leaves both parts NaN; total remains the contrast
    between the two native verdicts.
    """

    total: np.ndarray
    definition_alone: np.ndarray
    planning_response: np.ndarray


def attribution(flown, counted) -> Attribution:
    """Decompose two (result, study) flights using their declared criteria."""
    result, study = flown
    counted_result, counted_study = counted
    native = verdict(result, study).yield_
    foreign = cross_yield(result, study, counted_study.observer.criterion)
    replanned = verdict(counted_result, counted_study).yield_
    return Attribution(replanned - native, foreign - native, replanned - foreign)


class CriterionObjective(AbstractObjective):
    """JAX scorer credit from true measurement columns and catalog claim points."""

    def evaluate(self, carry, study):
        """Return scalar yield, crediting each consistently joined identity once."""
        criterion, channel = study.observer.criterion, study.channel_model
        if not criterion.labels:
            return jnp.asarray(0.0, F64)
        n_slots, n_bands = channel.n_slots, channel.n_bands
        catalogs, has_catalog = _catalogs(carry, study)
        log = carry.log
        valid = _completed(log, study.caps.kind_names.index(MEASUREMENT_KIND_NAME))
        order = jnp.argsort(
            jnp.where(valid, log.end_tick, jnp.iinfo(I64).max), stable=True
        )

        def score_star(star):
            def fold(acc, index):
                state, seen = acc
                row = unpack(channel.measurement_template, log.payload[index])
                usable = usable_measurement(
                    log.status[index], row.valid_science, row.variance
                )
                participates = participating_entries(
                    row.valid_science,
                    row.detected,
                    row.counts,
                    row.variance,
                    row.mode,
                    channel.mode_table,
                )
                row = dataclasses.replace(
                    row, counts=row.true_counts, variance=row.true_variance
                )
                take = valid[index] & (log.subject[index] == star) & usable
                state = jax.lax.cond(
                    take,
                    lambda s: criterion.update(s, row, channel.mode_table),
                    lambda s: s,
                    state,
                )
                assigned = (
                    (jnp.arange(n_slots)[:, None] == row.assignment[None, :])
                    & take
                    & participates[None, :]
                )
                identity = jnp.where(
                    (row.identity >= 0) & (row.identity < n_slots),
                    row.identity,
                    n_slots,
                )
                hits = jnp.any(
                    assigned[:, :, None]
                    & (identity[None, :, None] == jnp.arange(n_slots + 1)),
                    axis=1,
                )
                return (state, seen | hits), None

            (state, seen), _ = jax.lax.scan(
                fold,
                (
                    criterion.init_state(n_slots, n_bands),
                    jnp.zeros((n_slots, n_slots + 1), bool),
                ),
                order,
            )
            identities, joined = _track_join(
                seen, catalogs.present[star] & has_catalog[star]
            )
            params = jax.tree.map(
                lambda x, star=star, identities=identities: x[star][identities],
                catalogs.params,
            )
            claims = _truth_claims(
                criterion, params, channel.scenes.luminosity_lsun[star]
            )
            values, labels = criterion.value(state, claims, carry.tick)
            credit, _, _ = _credit(values, labels, identities, joined, n_slots)
            return credit.sum()

        return jax.lax.map(score_star, jnp.arange(study.caps.n_subjects)).sum()
