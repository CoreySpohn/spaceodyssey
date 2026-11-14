"""Shared pieces of the posterior-kernel tests: visit rows realized from the fixture.

A request holds the realized `measurement` rows of chosen stars, entered for both
audiences, so a test runs the posterior boundary on rows a study would hold.
"""

import dataclasses

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


def golden_slices(boundary, study, requests):
    """Positional gather/scatter reference for uncached imaging-only requests.

    Share the unchanged numerical kernels and lane runner, but bypass StarLane,
    tree-based track gathering and scatter_results. Keep only the old positional
    summary contract, evaluated from the same rows without a serialized golden.
    Forwarded entries, catalog rows and cached clouds are outside this fixture.
    """
    assert boundary.kernel == "posterior" and not boundary.resummary
    ph.require_x64()
    ph._photomancy_refit()
    req = jax.tree.map(np.asarray, requests)
    channel = study.channel_model
    population = study.decision_view().belief_population
    settings = boundary._post_settings(study)
    shape = req.dirty.shape
    own = req.source == np.arange(shape[1] * shape[2]).reshape(shape[1:])
    assert np.all(own[req.dirty])
    computed = req.dirty & np.asarray(boundary._readable(study, channel))[None, :, None]
    spec = ph.slice_spec(study)
    slices = {name: np.zeros((*shape, *tail), dtype) for name, tail, dtype in spec}
    slices["available"].fill(True)
    status = np.full(shape, int(ph.RefitStatus.NO_SUPPORT), np.int64)
    diagnostics = np.zeros((*shape, boundary.n_diag))
    fields = [f.name for f in dataclasses.fields(ph._FitLane) if f.name != "live"]
    summary_fields = (
        "p_det",
        "fc_counts",
        "fc_variance",
        "scatter",
        "pos_mean",
        "p_claim",
        "ln_period_q",
        "ess",
        "p_exist",
    )

    def fit_track(item, engine):
        def run(scale):
            def kernel(lanes):
                if engine == "first":
                    return ph._first_tier_lanes(
                        channel, population, lanes, settings, scale
                    )
                return ph._fit_lanes(
                    channel, population, lanes, settings, engine, scale
                )

            out = boundary._run_groups(kernel, [item], fields)[0]
            return out.result if engine == "first" else out

        result = run(1)
        retried = result.updater_status == int(ph.UpdaterStatus.LOW_ESS)
        if retried:
            result = run(ph.RETRY_FACTOR)
        positional = (
            result.p_det,
            result.fc_counts,
            result.fc_variance,
            result.scatter,
            result.pos_mean,
            result.p_claim,
            result.ln_period_q,
            result.ess,
            result.p_exist,
        )
        finite = bool(result.finite) and all(np.isfinite(x).all() for x in positional)
        return positional, ph.slot_status(result.updater_status, finite), retried

    for w, a, subject in np.argwhere(computed):
        at = (w, a, subject)
        use = req.rows.valid[at]
        assert np.all(
            req.rows.kind[at][use] == study.caps.kind_names.index(boundary.report_kind)
        )
        measured = ph.unpack(channel.measurement_template, req.rows.payload[at][use])
        assert all(
            channel.mode_table.modes[int(mode)].kind == "imaging"
            for mode in measured.mode
        )
        n_rows = int(use.sum())
        assert n_rows > 0
        cap = use.size
        rows = min(b for b in (*boundary.row_buckets, cap) if n_rows <= b <= cap)
        payload = np.zeros((rows, channel.widths()[0]))
        payload[:n_rows] = req.rows.payload[at][use, : payload.shape[1]]
        end_tick = np.zeros(rows, np.int64)
        end_tick[:n_rows] = req.rows.end_tick[at][use]
        grid = ph._grid_lanes(
            channel,
            ph._RowLane(
                payload=jnp.asarray(np.repeat(payload[None], boundary.lanes, axis=0)),
                use=jnp.asarray(
                    np.repeat((np.arange(rows) < n_rows)[None], boundary.lanes, axis=0)
                ),
                subject=jnp.full(boundary.lanes, subject),
            ),
            settings,
        )
        report = jax.tree.map(lambda x: np.asarray(x[0]), grid)
        n_det = (report["detected"] & report["valid"]).sum(axis=1)
        dark = np.flatnonzero(n_det == 0)
        base = dict(
            subject=np.asarray(subject),
            tick=np.asarray(req.tick[w]),
            key=req.keys[at],
            n_undetected=np.asarray(dark.size),
            end_tick=end_tick,
            **{name: report[name] for name in ("epochs", "sep", "dmag0", "limit")},
        )
        track_fields = (
            "detected",
            "counts",
            "valid",
            "ra",
            "dec",
            "ra_err",
            "dec_err",
            "corr",
        )
        codes, retried = [], False
        for slot in np.flatnonzero(n_det):
            item = dict(
                base,
                slot=np.asarray(slot),
                **{name: report[name][slot] for name in track_fields},
            )
            astrometry = (
                item["detected"]
                & item["valid"]
                & (item["ra_err"] > 0)
                & (item["dec_err"] > 0)
            )
            values, code, retry = fit_track(
                item, "single" if astrometry.sum() <= 1 else "multi"
            )
            codes.append(code)
            retried |= retry
            slices["status"][(*at, slot)] = code
            if code not in ph.SLOT_KEPT:
                for name, value in zip(summary_fields, values, strict=True):
                    slices[name][(*at, slot)] = value
        if dark.size:
            item = dict(
                base,
                slot=np.asarray(settings.n_slots),
                **{name: np.zeros_like(report[name][0]) for name in track_fields},
            )
            item["valid"] = report["valid"][dark].all(axis=0)
            values, code, retry = fit_track(item, "first")
            codes.append(code)
            retried |= retry
            slices["status"][(*at, dark)] = code
            if code not in ph.SLOT_KEPT:
                for name, value in zip(summary_fields, values, strict=True):
                    slices[name][(*at, dark)] = value
        n_low = codes.count(ph.SLOT_DEGRADED)
        n_fail = sum(code in ph.SLOT_KEPT for code in codes)
        slices["ref_tick"][at] = req.tick[w]
        slices["n_obs"][at] = n_rows
        slices["n_det"][at] = n_det
        slices["p_exist"][(*at, n_det > 0)] = 1.0
        status[at] = int(
            ph.RefitStatus.FAILED
            if n_fail == len(codes)
            else ph.RefitStatus.DEGRADED
            if n_fail or n_low
            else ph.RefitStatus.OK
        )
        diagnostics[at] = [n_rows, float(retried), n_low, n_fail]
    return ph.RefitResult(
        slices=tuple(jnp.asarray(slices[name]) for name, _, _ in spec),
        status=jnp.asarray(status),
        diagnostics=jnp.asarray(diagnostics),
        computed=jnp.asarray(computed),
    )
