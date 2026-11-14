"""Shared pieces of the posterior-kernel tests: visit rows realized from the fixture.

A request holds the realized `measurement` rows of chosen stars, entered for both
audiences, so a test runs the posterior boundary on rows a study would hold.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.adapters.measurement import MEASUREMENT_KIND_NAME
from spaceodyssey.core.enums import Audience
from spaceodyssey.core.kinds import RealizeContext
from spaceodyssey.core.records import record_none
from spaceodyssey.core.refresh import RefitRequest


def visit_context(channel, start_d, mode):
    """The realization context of a visit of `mode` that starts on day `start_d`."""
    built = channel.assets()
    per_day = int(built.context.clock.ticks_per_day)
    settle = int(built.context.settle_ticks)
    start = round(start_d * per_day)
    end = start + settle + round(channel.mode_table.duration_s(mode) * 1000)
    zero = jnp.asarray(0)
    return RealizeContext(
        jnp.asarray(start), jnp.asarray(end), zero, jnp.asarray(1.0), zero, zero
    )


def realize_visits(channel, truth, star, mode, days, seed=0):
    """`[(row, end_tick)]` of one star's visits on `days`, one noise key per visit."""
    out = []
    for i, day in enumerate(days):
        ctx = visit_context(channel, day, mode)
        key = jax.random.fold_in(jax.random.key(seed), i)
        row = channel.realize(truth, star, mode, 1.0, key, None, None, context=ctx)
        out.append((np.asarray(row), int(ctx.end_tick)))
    return out


def visit_request(study, visits, *, tick, seed=0, mask_policy=False):
    """A one-world request from `visits[star] = [(row, end_tick)]`, every star dirty.

    Entry `[0, 0, s]` is the observer's and `[0, 1, s]` the policy's. With
    `mask_policy` the policy's payload reads zero, as a column hidden from it does.
    """
    caps = study.caps
    n_subjects = caps.n_subjects
    kind = caps.kind_names.index(MEASUREMENT_KIND_NAME)
    base = jax.tree_util.tree_map(np.asarray, record_none(caps))
    shape = (1, 2, n_subjects, caps.rows_per_subject)
    rows = jax.tree_util.tree_map(
        lambda x: np.broadcast_to(x, shape + x.shape).copy(), base
    )
    for star, seen in visits.items():
        for j, (row, end) in enumerate(seen):
            for a in range(2):
                audience = (Audience.OBSERVER, Audience.POLICY)[a]
                visible = np.asarray(caps.payload_audience[kind][: len(row)]) & int(
                    audience
                )
                rows.payload[0, a, star, j, : len(row)] = (
                    0.0 if (mask_policy and a == 1) else np.where(visible, row, 0.0)
                )
                rows.valid[0, a, star, j] = True
                rows.kind[0, a, star, j] = kind
                rows.end_tick[0, a, star, j] = end
    rng = np.random.default_rng(seed)
    own = np.arange(2 * n_subjects).reshape(2, n_subjects)
    dirty = np.zeros((1, 2, n_subjects), bool)
    for star in visits:
        dirty[0, :, star] = True
    return RefitRequest(
        tick=np.full(1, tick, np.int64),
        ordinal=np.zeros((1, 2, n_subjects), np.int64),
        dirty=dirty,
        n_rows=np.full((1, 2, n_subjects), caps.rows_per_subject, np.int64),
        rows=rows,
        row_keys=np.zeros((1, 2, n_subjects, caps.rows_per_subject, 2), np.uint32),
        keys=rng.integers(0, 2**31, (1, 2, n_subjects, 2)).astype(np.uint32),
        source=own[None].copy(),
    )


def _previous_fit_lane(channel, population, lane, st, engine, scale):
    """One slot with detections: refit, then its summary leaves and the status."""
    refit = ph._photomancy_refit()
    _, target = ph._star_target(refit, channel, lane.subject, st)
    report = ph._report_of(refit, channel, lane, lane.detected)
    astrometry = lane.detected & lane.valid & (lane.ra_err > 0) & (lane.dec_err > 0)
    records = ph.RelativeAstromData(
        times=lane.epochs,
        ra=lane.ra,
        dec=lane.dec,
        ra_err=jnp.where(astrometry, lane.ra_err, 1.0),
        dec_err=jnp.where(astrometry, lane.dec_err, 1.0),
        corr=jnp.where(astrometry, lane.corr, 0.0),
        planet_id=jnp.zeros(lane.valid.shape, int),
        is_valid=astrometry,
    )
    n_round = st.n_round * scale
    key = ph._slot_key(lane, scale)
    if engine == "single":
        cloud = refit.scale_rotate_refit(
            target,
            records,
            key,
            n_particles=st.n_particles,
            n_proposals=n_round,
            ess_floor=st.ess_floor,
            report=report,
        )
    else:
        cloud = refit.collapsed_refit(
            target,
            records,
            key,
            n_particles=st.n_particles,
            n_round=n_round,
            n_rounds=st.n_rounds,
            n_centers=st.n_centers,
            defensive=st.defensive,
            n_constants=st.n_constants,
            ess_floor=st.ess_floor,
            report=report,
            max_rounds=st.max_rounds,
        )
    albedo = population.albedo_prior(
        jax.random.fold_in(key, ph._ALBEDO_TAG), st.n_particles, channel.bands
    )
    elements = ph.from_photomancy(
        cloud.elements, albedo, ph._detection_band_index(channel)
    )
    last = jnp.max(jnp.where(lane.detected & lane.valid, lane.end_tick, -1))
    leaves = ph._slot_leaves(channel, lane.subject, lane.tick, last, elements, st)
    finite = ph._parameters_finite(elements)
    return (
        leaves.p_det,
        leaves.scatter,
        leaves.pos_mean,
        leaves.p_claim,
        leaves.ln_period_q,
        cloud.ess,
        cloud.status.astype(ph.I64),
        finite,
        elements,
    )


def _previous_first_lane(channel, population, lane, st, scale):
    """Per-slot existence and conditional leaves from the belief population."""
    cloud, elements = ph._first_tier_cloud(channel, population, lane, st, scale)
    expected = population.expected_planets(
        ph._population_star(channel, lane.subject),
        jax.random.fold_in(lane.key, ph._EXPECTED_TAG),
        st.n_estimate_systems,
    )
    leaves = ph._first_leaves(
        channel,
        lane.subject,
        lane.tick,
        lane.n_undetected,
        elements,
        cloud.log_evidence,
        expected,
        st,
    )
    return (
        *leaves,
        cloud.ess,
        cloud.status.astype(ph.I64),
        ph._parameters_finite(elements),
        elements,
        cloud.log_evidence,
        expected,
    )


@eqx.filter_jit
def _previous_fit_lanes(channel, population, lanes, st, engine, scale):
    """A batch of slot fits, one output tuple per leaf."""
    fit = ph._shrink_lane if engine == "shrink" else _previous_fit_lane
    return ph._over_lanes(
        ph._if_live(lambda lane: fit(channel, population, lane, st, engine, scale)),
        lanes,
        st.lane_batch,
    )


@eqx.filter_jit
def _previous_first_lanes(channel, population, lanes, st, scale):
    """A batch of first-tier entries."""
    return ph._over_lanes(
        ph._if_live(
            lambda lane: _previous_first_lane(channel, population, lane, st, scale)
        ),
        lanes,
        st.lane_batch,
    )


def golden_slices(boundary, study, requests):
    """The previous positional boundary and kernels, evaluated on identical inputs.

    This reference retains the pre-module result ordering independently of the
    named-module path. It never reads or writes a serialized golden.
    """
    assert boundary.kernel == "posterior" and not boundary.resummary
    return _previous_refit(boundary, study, requests)


def _previous_refit(self, study, requests):
    """Evaluate the positional gather and scatter kept as an independent reference."""
    ph.require_x64()
    # Import outside any trace: the module builds JAX constants as it loads.
    ph._photomancy_refit()
    req = jax.tree_util.tree_map(np.asarray, requests)
    n_worlds, _, count = req.dirty.shape
    spec = ph.slice_spec(study)
    names = tuple(name for name, _, _ in spec)
    channel = study.channel_model
    st = self._post_settings(study)
    n_slots = st.n_slots
    readable = np.asarray(self._readable(study, channel))
    flat = np.arange(2 * count).reshape(2, count)
    own = req.dirty & (req.source == flat[None])
    mine = own & readable[None, :, None]
    slices = {n: np.zeros((n_worlds, 2, count, *s), d) for n, s, d in spec}
    status = np.full((n_worlds, 2, count), int(ph.RefitStatus.NO_SUPPORT), np.int64)
    diag = np.zeros((n_worlds, 2, count, self.n_diag))
    tick = np.asarray(req.tick)
    tick = tick.reshape(tick.shape + (1,) * (3 - tick.ndim))
    tick = np.broadcast_to(tick, (n_worlds, 2, count))
    kind_index = study.caps.kind_names.index(self.report_kind)
    shrink = self.kernel == "shrinkage"
    cat_index = study.caps.kind_names.index(self.catalog_kind) if shrink else -1
    width = channel.widths()[0]
    cap = req.rows.payload.shape[3]
    buckets = sorted({b for b in self.row_buckets if b < cap} | {cap})

    entries = []
    for w, a, s in np.argwhere(mine):
        use = req.rows.valid[w, a, s] & (req.rows.kind[w, a, s] == kind_index)
        measured = ph.unpack(channel.measurement_template, req.rows.payload[w, a, s])
        imaging = np.asarray(
            tuple(m.kind == "imaging" for m in channel.mode_table.modes)
        )
        use &= imaging[np.where(use, np.asarray(measured.mode), 0)]
        idx = np.flatnonzero(use)
        if len(idx) == 0:
            continue
        rows = next(b for b in buckets if b >= len(idx))
        payload = np.zeros((rows, width))
        payload[: len(idx)] = req.rows.payload[w, a, s][idx, :width]
        end = np.zeros(rows, np.int64)
        end[: len(idx)] = req.rows.end_tick[w, a, s][idx]
        used = np.arange(rows) < len(idx)
        truth, has_cat = None, False
        if shrink:
            is_cat = req.rows.valid[w, a, s] & (req.rows.kind[w, a, s] == cat_index)
            has_cat = bool(is_cat.any())
            if has_cat:
                row = req.rows.payload[w, a, s][int(np.argmax(is_cat))]
                truth = ph.unpack(
                    ph.catalog_template(n_slots, channel.n_bands), row
                ).params
        entries.append(
            {
                "at": (w, a, s),
                "truth": truth,
                "has_cat": has_cat,
                "n_rep": len(idx),
                "rows": rows,
                "payload": payload,
                "use": used,
                "end": end,
                "key": np.asarray(req.keys[w, a, s], np.uint32),
                "tick": int(tick[w, a, s]),
            }
        )

    by_rows = {}
    for e in entries:
        by_rows.setdefault(e["rows"], []).append(e)
    for group in by_rows.values():
        for start in range(0, len(group), self.lanes):
            part = group[start : start + self.lanes]
            padded = part + [part[0]] * (self.lanes - len(part))
            lanes = ph._RowLane(
                payload=jnp.asarray(np.stack([e["payload"] for e in padded])),
                use=jnp.asarray(np.stack([e["use"] for e in padded])),
                subject=jnp.asarray([e["at"][2] for e in padded]),
            )
            result = {
                k: np.asarray(v) for k, v in ph._grid_lanes(channel, lanes, st).items()
            }
            for i, e in enumerate(part):
                e["report"] = {k: v[i] for k, v in result.items()}

    fits, firsts = [], []
    for e in entries:
        rep = e["report"]
        astrometry = (
            rep["detected"] & rep["valid"] & (rep["ra_err"] > 0) & (rep["dec_err"] > 0)
        )
        e["n_det"] = (rep["detected"] & rep["valid"]).sum(axis=1)
        e["live"] = np.flatnonzero(e["n_det"] > 0)
        e["dark"] = np.flatnonzero(e["n_det"] == 0)
        e["skipped"] = []
        base = {
            "subject": np.asarray(e["at"][2]),
            "tick": np.asarray(e["tick"]),
            "key": e["key"],
            "epochs": rep["epochs"],
            "sep": rep["sep"],
            "dmag0": rep["dmag0"],
            "limit": rep["limit"],
            "end_tick": e["end"],
            "n_undetected": np.asarray(len(e["dark"])),
        }
        for slot in e["live"]:
            if shrink and not e["has_cat"]:
                e["skipped"].append(slot)
                continue
            item = dict(base)
            item["slot"] = np.asarray(slot)
            if shrink:
                item["truth"] = jax.tree.map(lambda x, slot=slot: x[slot], e["truth"])
            for f in (
                "detected",
                "valid",
                "counts",
                "ra",
                "dec",
                "ra_err",
                "dec_err",
                "corr",
            ):
                item[f] = rep[f][slot]
            n_records = int(astrometry[slot].sum())
            engine = "shrink" if shrink else "single" if n_records <= 1 else "multi"
            fits.append((e, slot, engine, item))
        if len(e["dark"]):
            first = dict(base)
            first["slot"] = np.asarray(n_slots)
            # a row counts for the first tier when every undetected slot read it
            first["valid"] = rep["valid"][e["dark"]].all(axis=0)
            for f in (
                "detected",
                "counts",
                "ra",
                "dec",
                "ra_err",
                "dec_err",
                "corr",
            ):
                first[f] = np.zeros_like(rep[f][0])
            firsts.append((e, first))

    lane_cls = ph._ShrinkLane if shrink else ph._FitLane
    fit_fields = [f.name for f in dataclasses.fields(lane_cls) if f.name != "live"]

    def run_fits(selected, scale):
        results = {}
        for engine in ("shrink",) if shrink else ("single", "multi"):
            group = [(k, it) for k, (_, _, eng, it) in selected if eng == engine]
            for rows in {it["epochs"].shape[0] for _, it in group}:
                part = [(k, it) for k, it in group if it["epochs"].shape[0] == rows]
                out = self._run_groups(
                    lambda lanes, engine=engine: _previous_fit_lanes(
                        channel,
                        study.decision_view().belief_population,
                        lanes,
                        st,
                        engine,
                        scale,
                    ),
                    [it for _, it in part],
                    fit_fields,
                    lane_cls,
                )
                results.update({k: o for (k, _), o in zip(part, out, strict=True)})
        return results

    def run_firsts(selected, scale):
        results = {}
        for rows in {it["epochs"].shape[0] for _, (_, it) in selected}:
            part = [(k, it) for k, (_, it) in selected if it["epochs"].shape[0] == rows]
            out = self._run_groups(
                lambda lanes: _previous_first_lanes(
                    channel,
                    study.decision_view().belief_population,
                    lanes,
                    st,
                    scale,
                ),
                [it for _, it in part],
                first_fields,
            )
            results.update({k: o for (k, _), o in zip(part, out, strict=True)})
        return results

    first_fields = [f.name for f in dataclasses.fields(ph._FitLane) if f.name != "live"]
    low = int(ph.UpdaterStatus.LOW_ESS)
    indexed_fits = list(enumerate(fits))
    fit_out, stored_retries = {}, set()
    digests = {}
    if self.resummary:
        digests = {
            k: ph._fit_digest(
                it, st, study.decision_view().belief_population, channel.bands
            )
            for k, (_, _, eng, it) in indexed_fits
        }
        fit_out, stored_retries = self._answer_from_store(
            channel, st, indexed_fits, digests
        )
    to_fit = [(k, fit) for k, fit in indexed_fits if k not in fit_out]
    fit_out.update(run_fits(to_fit, 1))
    retried_fits = [(k, fit) for k, fit in to_fit if fit_out[k][6] == low]
    fit_out.update(run_fits(retried_fits, ph.RETRY_FACTOR))
    if self.resummary:
        self.resummary_stats["misses"] += len(to_fit)
        retried_now = {k for k, _ in retried_fits}
        for k, _ in to_fit:
            self._store(digests[k], fit_out[k], k in retried_now)
    indexed_firsts = list(enumerate(firsts))
    first_out, stored_first_retries, first_digests = {}, set(), {}
    if self.resummary:
        first_digests = {
            k: ph._fit_digest(
                it, st, study.decision_view().belief_population, channel.bands
            )
            for k, (_, it) in indexed_firsts
        }
        first_out, stored_first_retries = self._answer_firsts_from_store(
            channel, st, indexed_firsts, first_digests
        )
    first_fit = [(k, f) for k, f in indexed_firsts if k not in first_out]
    first_out.update(run_firsts(first_fit, 1))
    retried_firsts = [(k, f) for k, f in first_fit if first_out[k][5] == low]
    first_out.update(run_firsts(retried_firsts, ph.RETRY_FACTOR))
    if self.resummary:
        self.resummary_stats["misses"] += len(first_fit)
        retried_now = {k for k, _ in retried_firsts}
        for k, _ in first_fit:
            self._store_first(first_digests[k], first_out[k], k in retried_now)
    retried_ids = (
        {id(fits[k][0]) for k, _ in retried_fits}
        | {id(fits[k][0]) for k in stored_retries}
        | {id(firsts[k][0]) for k, _ in retried_firsts}
        | {id(firsts[k][0]) for k in stored_first_retries}
    )

    per_entry = {}
    for k, (e, slot, _, _) in enumerate(fits):
        per_entry.setdefault(id(e), {"e": e, "fits": {}, "first": None})["fits"][
            slot
        ] = fit_out[k]
    for k, (e, _) in enumerate(firsts):
        per_entry.setdefault(id(e), {"e": e, "fits": {}, "first": None})["first"] = (
            first_out[k]
        )
    for item in per_entry.values():
        e = item["e"]
        w, a, s = e["at"]
        codes = np.zeros(n_slots, np.int64)
        units = []
        for slot, out in item["fits"].items():
            p_det, scatter, pos, p_claim, q, ess, raw, finite, _ = out
            valid = bool(finite) and all(np.isfinite(x).all() for x in out[:6])
            codes[slot] = ph.slot_status(raw, valid)
            units.append(codes[slot])
            if codes[slot] in ph.SLOT_KEPT:
                continue
            slices["p_exist"][w, a, s, slot] = 1.0
            slices["p_det"][w, a, s, slot] = p_det
            slices["scatter"][w, a, s, slot] = scatter
            slices["pos_mean"][w, a, s, slot] = pos
            slices["p_claim"][w, a, s, slot] = p_claim
            slices["ln_period_q"][w, a, s, slot] = q
            slices["ess"][w, a, s, slot] = ess
        if item["first"] is not None:
            p_exist, p_det, p_claim, q, ess, raw, finite, _, _, _ = item["first"]
            valid = bool(finite) and all(
                np.isfinite(x).all() for x in item["first"][:5]
            )
            code = ph.slot_status(raw, valid)
            codes[e["dark"]] = code
            for slot in e["dark"] if code not in ph.SLOT_KEPT else ():
                slices["p_exist"][w, a, s, slot] = p_exist
                slices["p_det"][w, a, s, slot] = p_det
                slices["p_claim"][w, a, s, slot] = p_claim
                slices["ln_period_q"][w, a, s, slot] = q
                slices["ess"][w, a, s, slot] = ess
                codes[slot] = code
            units.append(code)
        for slot in e["skipped"]:
            codes[slot] = ph.SLOT_NO_SUPPORT
            units.append(ph.SLOT_NO_SUPPORT)
        n_low = sum(c == ph.SLOT_DEGRADED for c in units)
        n_fail = sum(c in ph.SLOT_KEPT for c in units)
        slices["ref_tick"][w, a, s] = e["tick"]
        slices["n_obs"][w, a, s] = e["n_rep"]
        slices["n_det"][w, a, s] = e["n_det"]
        slices["p_exist"][w, a, s, e["n_det"] > 0] = 1.0
        slices["status"][w, a, s] = codes
        status[w, a, s] = (
            int(ph.RefitStatus.FAILED)
            if n_fail == len(units)
            else int(ph.RefitStatus.DEGRADED)
            if n_fail or n_low
            else int(ph.RefitStatus.OK)
        )
        diag[w, a, s] = [e["n_rep"], float(id(e) in retried_ids), n_low, n_fail]
    for w, a, s in np.argwhere(req.dirty & ~own):
        sa, ss = divmod(int(req.source[w, a, s]), count)
        for piece in slices.values():
            piece[w, a, s] = piece[w, sa, ss]
        status[w, a, s] = status[w, sa, ss]
        diag[w, a, s] = diag[w, sa, ss]
    return ph.RefitResult(
        slices=tuple(jnp.asarray(slices[n]) for n in names),
        status=jnp.asarray(status),
        diagnostics=jnp.asarray(diag),
        computed=jnp.asarray(mine),
    )
