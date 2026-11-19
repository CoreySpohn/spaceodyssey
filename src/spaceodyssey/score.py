"""Scoring a run: the log read by the verdict, set against the unsealed truth.

``score`` is the only reader of the truth. Each star's log is replayed through the
study's roles stage by stage, a stage ending at each release time of the star's rows,
and every planet of the truth is scored at each stage. It returns the tables milestone 1
needs, as generic columns: the information table, the certificate table, the resolution
numbers and the predicted and realized counts. ``format_report`` prints them.
``found_count`` reads the planets a run's searches found against the truth.

The claim is read two ways. Its information is that of the whole conjunction, the
planet's claim bit: a claim-true planet scores ``ln q`` and a look-alike ``ln(1 - q)``,
each ``q`` clipped to the cloud's resolution first, and a claim-true planet's
information is split over the conjuncts by ``chain_gain``. Its certificate is that of
the claim as stated, conditional when ``claim.given`` is set; a planet whose truth fails
the condition is void, recorded in its own column beside the certificate.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from spaceodyssey._standin.diagnostics import khat_threshold
from spaceodyssey.belief import (
    Belief,
    absorb,
    detected_slots,
    draw_claim_mass,
    due_rows,
    new_planets,
    replay,
    replay_belief,
)
from spaceodyssey.claim import Claim
from spaceodyssey.information import (
    INAPPLICABLE,
    NEGATIVE,
    POSITIVE,
    UNRESOLVED,
    certify,
    chain_gain,
    clip_q,
    diagnose,
    refit_trigger,
    widen_by_mass_outside,
)
from spaceodyssey.study import Study
from spaceodyssey.types import (
    ACCEPTED,
    Atoms,
    ObservationLog,
    RefitTerm,
    clock_d,
    planet_tracks,
    real_rows,
)
from spaceodyssey.world import SealedWorld


class Report(eqx.Module):
    """The tables of one scored run, each a dict of named columns of equal length.

    The information and resolution tables are read by the verdict members, roles 1 to
    ``R - 1`` (column ``member``); the forecast does not judge.

    Every table holds the planets the star's cloud tracks (``score``): a planet the
    run found enters from the stage that refits it, and an undetected pool planet
    enters none.

    Attributes:
        information: one entry per star, planet, stage and member: the stage
            (``stage``, from 1), the time it ends (``time_d``), whether that is the
            run's end (``end``; a later stage holds rows in flight then), whether
            the planet is claim-true (``truth``), the whole claim's clipped ``q``
            before and after the stage, the signed ``information`` (the change of
            ``ln q`` for a claim-true planet, of ``ln(1 - q)`` for a look-alike), and
            the ``chain_gain`` of each conjunct, shape ``(n, C)``, NaN for a
            look-alike.
        certificates: one entry per star and planet, at the last stage: the
            certificate's ``status``, ``q_lo`` and ``q_hi``; ``void`` (the truth fails
            the claim's condition); ``truth``; the refit trigger ``refit``; and the
            candidacy range over the members, ``psi_lo`` and ``psi_hi``.
        resolution: one entry per star, planet and member, at the last stage: the
            claim's ``q``, its ``mc_se``, ``khat`` against ``khat_max`` and whether
            the member's tail is refused there (``tail_refused``: ``khat`` above it
            or NaN, the gate of the certificates and the refit trigger; a planner's
            gate, Kish size and Pareto shape at its own thresholds, is
            ``belief.refused_roles``), ``n_eff``, the
            Kish size of the condition-weighted draws ``n_cond`` (the ``n`` of the
            certificate's Wilson bound), the clip ``resolution`` ``1 / (2 n_eff)``
            and the number of stages whose whole claim ``q`` it ``clipped``, the
            condition mass ``psi``, and the member's
            ``prior_mass_outside`` ``m`` with the hard bound it puts on ``q``,
            ``[(1 - m) q, (1 - m) q + m]`` (``q_bound_lo``, ``q_bound_hi``).
        counts: one entry per member, over the tracked planets: the predicted
            applicable count ``sum psi`` beside the realized one, the predicted
            applicable errors ``sum (psi - q)``, with ``q`` the whole claim's, beside
            the realized ones (applicable planets that are not claim-true), and the
            stars at which the member's Pareto shape is above the threshold
            (``tail_refused_stars``).
    """

    information: dict[str, Array]
    certificates: dict[str, Array]
    resolution: dict[str, Array]
    counts: dict[str, Array]


_INFORMATION = (
    "star", "planet", "stage", "time_d", "end", "member", "truth", "q_before",
    "q_after",
    "information", "chain_gain",
)  # fmt: skip
_CERTIFICATES = (
    "star", "planet", "status", "q_lo", "q_hi", "void", "truth", "refit", "psi_lo",
    "psi_hi",
)  # fmt: skip
_RESOLUTION = (
    "star", "planet", "member", "q", "mc_se", "khat", "khat_max", "tail_refused",
    "n_eff",
    "n_cond", "resolution", "clipped", "psi", "prior_mass_outside", "q_bound_lo",
    "q_bound_hi",
)  # fmt: skip
_COUNTS = (
    "member", "predicted_applicable", "realized_applicable", "predicted_errors",
    "realized_errors", "tail_refused_stars",
)  # fmt: skip

_replay = eqx.filter_jit(replay)
_absorb = eqx.filter_jit(absorb)
_replay_belief = eqx.filter_jit(replay_belief)
_new_planets = eqx.filter_jit(new_planets)

_REFIT_KEY_TAG = 3
"""Tag folded into the default score key, apart from the run keys' tags
(``world.run_keys``)."""


@eqx.filter_jit
def _refit(refitter, r, log, models, star, time_d, key):
    """A star's refit at a stage: its cloud updated by every row due then, ungated.

    ``score`` gates in ``certify``, so the refit's own gate is off (a refit is never
    refused here, and its diagnostics are read like any cloud's).
    """
    term = RefitTerm(log, due_rows(log, r.refit, time_d), models, star)
    fresh = refitter.refit(r.refit, term, key, min_n_eff=0.0, max_khat=math.inf)
    return replay(fresh, log, models, star, time_d)


@eqx.filter_jit
def _reading(b, truth, claim, star, planet, truth_planet, outside, alpha, khat_max):
    """What ``score`` reads off one stage's belief about one planet, every role.

    ``planet`` is the planet's slot in the draws and ``truth_planet`` its slot in the
    truth (``associate_planets``).
    """
    whole = Claim(claim.conjuncts)
    q_whole = diagnose(b, whole, star, planet, outside)
    d = diagnose(b, claim, star, planet, outside)
    true_mass = draw_claim_mass(truth, claim, star, truth_planet)[:, 0]
    return {
        "q": q_whole.q,
        "q_clipped": clip_q(q_whole.q, 0.5 / q_whole.n_eff),
        "diagnostics": d,
        "certificate": certify(d, alpha, khat_max=khat_max),
        "refit": refit_trigger(d, alpha, khat_max=khat_max),
        "truth": true_mass[-1] > 0.5,
        "applicable": true_mass[claim.given] > 0.5,
    }


@eqx.filter_jit
def _chain_gains(before, after, claim, star, planet):
    """``chain_gain`` of the whole claim for every role, shape ``(R, C)``."""
    whole = Claim(claim.conjuncts)
    roles = range(before.log_prior_ratio.shape[0])
    return jnp.stack([chain_gain(before, after, whole, star, planet, r) for r in roles])


def _star(tree, s: int):
    """Star ``s`` of a pytree with a leading star axis."""
    return jax.tree.map(lambda leaf: leaf[s], tree)


def _stage_beliefs(study: Study, log: ObservationLog, s: int, end_d: float, key):
    """The times that end star ``s``'s stages, the run's end stage, and the beliefs.

    Stage 0 ends at time zero, with the grant; each later stage ends at a release time
    of one of the star's accepted rows, or at the run's end ``end_d``, a stage of its
    own. A release that agrees with the end to rounding is the end stage. The replay
    is brought from one stage to the next by ``absorb``, so each row goes through it
    once. At a stage whose rows detected a planet the star's cloud holds no track for
    (``new_planets``), the star is first refit by the study's refitter from every row
    due then, with a key folded from ``key``, the star and the number of refits before
    it at the star: independent of the policy, so two arms that reach the same rows
    get the same refit.

    Returns:
        The stage times, the time of the end stage, the belief after each stage, and
        the slots each stage's cloud holds a track in (``planet_tracks``), ``(K,)``
        each.

    Raises:
        ValueError: if a stage needs a refit and the study has no refitter.
    """
    star_col, refusal, release_d = (
        np.asarray(x) for x in (log.rows.star, log.rows.refusal, log.rows.release_d)
    )
    mine = np.asarray(real_rows(log)) & (star_col == s) & (refusal == ACCEPTED)
    releases = sorted({float(t) for t in release_d[mine] if t > 0.0})
    end = next((t for t in releases if math.isclose(t, end_d, rel_tol=1e-12)), end_d)
    times = [0.0, *sorted({*releases, end} if end > 0.0 else releases)]

    models, roles = study.design.models, study.roles
    refit, pool = _star(study.refits, s), _star(study.pool, s)
    star = _star(study.stars, s)
    star_key = jax.random.fold_in(key, s)
    # Raw beliefs: score reads each role's diagnostics and gates in ``certify``.
    raw = {"min_n_eff": 0.0, "max_khat": math.inf}
    beliefs, tracks, n_refit = [], [], 0
    r = None
    for time_d in times:
        t = jnp.asarray(time_d)
        r = (
            _replay(refit, log, models, star, t)
            if r is None
            else _absorb(r, log, models, star, t)
        )
        if bool(jnp.any(_new_planets(r.refit, log, models, t))):
            if study.refitter is None:
                raise ValueError(
                    f"star {s}'s rows found a planet its cloud holds no track for, and "
                    "the study has no refitter"
                )
            k = jax.random.fold_in(star_key, n_refit)
            r = _refit(study.refitter, r, log, models, star, t, k)
            n_refit += 1
        beliefs.append(_replay_belief(r, roles, models, star, pool, **raw))
        tracks.append(np.asarray(planet_tracks(r.refit.atoms)))
    return times, end, beliefs, tracks


def associate_planets(truth: Atoms, draws: Atoms) -> Int[Array, " K"]:
    """The slot in the draws that stands for each planet of the truth, at one star.

    Planets are associated by truth label: planet ``k`` of the truth is planet ``k`` of
    every draw, so the association is the identity. That is a declared simplification.
    It holds when the draws are built on the truth's slots, as in a constructed study;
    the draws of a real refit label their planets by the fit, and a study of them
    replaces this function with a rule that matches each planet of the truth to a slot
    of the draws (by orbit, say). ``score`` reads the association only from here.

    Args:
        truth: the truth at one star, one draw of ``K`` planet slots.
        draws: the draws at the star.

    Returns:
        The draws' slot of each planet of the truth, shape ``(K,)``.

    Raises:
        ValueError: if the truth and the draws hold different numbers of planet
            slots.
    """
    n_slot = draws.exists.shape[-1]
    if truth.exists.shape[-1] != n_slot:
        raise ValueError(
            f"the truth holds {truth.exists.shape[-1]} planet slots, the draws {n_slot}"
        )
    return jnp.arange(n_slot)


def _truth_belief(truth, pool, n_role: int) -> Belief:
    """The truth as a belief of every role, its pool shaped like the run's and empty.

    The truth's one draw has unit weight under every role; the pool has the run's
    marks and no expected planets, so it never counts.
    """
    flat = jnp.zeros((n_role, 1))
    empty = eqx.tree_at(lambda p: p.log_mass, pool, jnp.full(n_role, -jnp.inf))
    return Belief(truth, flat, flat, empty)


def _log_score(q, true: bool):
    """The log score of a claim bit: ``ln q`` if it is true, ``ln(1 - q)`` if not."""
    return np.log(q) if true else np.log1p(-q)


def score(log: ObservationLog, world: SealedWorld, study: Study, *, key=None) -> Report:
    """Score a run's log against the truth of its world.

    Each star's rows are replayed through the study's roles, stage by stage (stage 0
    the grant, then one stage per release time of the star's accepted rows). At a
    stage whose rows detected a planet the star's cloud holds no track for, the star
    is refit first by the study's refitter, with a key independent of the policy
    (``_stage_beliefs``), so a planet the run found is judged against a cloud that can
    explain it. Each planet the truth holds is scored by every verdict member, read
    from the slot of the draws that ``associate_planets`` gives it, at every stage at
    which the cloud holds a track there: a planet first tracked at a stage (one the
    run found) has information rows from the stage after it, and a planet no stage
    tracks (an undetected pool planet) has no per-planet row at all. It has no
    per-planet belief; it is counted in the pool (``found_count`` against the truth,
    ``expected_found`` under the forecast). The certificates and the refit trigger
    gate the Pareto shape at ``khat_threshold`` of the refit's draw count.

    Every row the run took is scored at its own release time, those still in flight
    when the run ended included: their data were taken, and the science arrives after
    the last exposure. The run's end, the calendar time its last row ends
    (``clock_d``), is a stage of its own at every star, marked in the information
    table's ``end`` column, so the table shows what was known at the end and what the
    rows in flight added after it. A refused row that asked to wait moves the clock to
    its requested start (``step``), so it moves the end stage with it.
    The certificates, the resolution and the counts are read after the last release.

    Args:
        log: the run's log.
        world: the run's sealed world, one world (no world axis).
        study: the study the run belongs to.
        key: the key of the refits; by default one fixed key, folded apart from the
            run keys' tags, so every arm's refits share it.

    Returns:
        The report.

    Raises:
        ValueError: if the planets of the truth cannot be associated with the draws'
            slots (``associate_planets``), or a stage needs a refit and the study has
            no refitter.
    """
    if key is None:
        key = jax.random.fold_in(jax.random.key(0), _REFIT_KEY_TAG)
    truth = world.unseal()
    claim = study.claim
    n_star, n_role = study.prior_mass_outside.shape
    khat_max = khat_threshold(study.refits.atoms.cloud.log_weights.shape[-1])
    members = range(1, n_role)
    end_d = float(clock_d(log))

    information, certificates, resolution, planets = [], [], [], []
    for s in range(n_star):
        times, end, beliefs, tracks = _stage_beliefs(study, log, s, end_d, key)
        star, outside = _star(study.stars, s), study.prior_mass_outside[s]
        truth_s = _star(truth, s)
        slots = associate_planets(truth_s, _star(study.refits, s).atoms)
        truth_b = _truth_belief(truth_s, _star(study.pool, s), n_role)
        for k in np.flatnonzero(np.asarray(truth.exists[s, 0])):
            planet, truth_planet = slots[k], jnp.asarray(k)
            held = [bool(t[int(planet)]) for t in tracks]
            if not held[-1]:
                continue  # no stage tracks it: a pool planet, counted in the pool
            first = held.index(True)  # tracks only grow
            stages = [
                _reading(
                    b,
                    truth_b,
                    claim,
                    star,
                    planet,
                    truth_planet,
                    outside,
                    study.alpha,
                    khat_max,
                )
                for b in beliefs[first:]
            ]
            last, true = stages[-1], bool(stages[-1]["truth"])
            q_whole = np.asarray([x["q"] for x in stages])  # (stage, R)
            q_clip = np.asarray([x["q_clipped"] for x in stages])
            for j in range(1, len(stages)):
                before_b, after_b = beliefs[first + j - 1], beliefs[first + j]
                gain = np.asarray(_chain_gains(before_b, after_b, claim, star, planet))
                if not true:
                    gain = np.full_like(gain, np.nan)  # split only for a true claim
                for m in members:
                    before, after = q_clip[j - 1, m], q_clip[j, m]
                    change = _log_score(after, true) - _log_score(before, true)
                    information.append(
                        {
                            "star": s, "planet": k, "stage": first + j,
                            "time_d": times[first + j], "end": times[first + j] == end,
                            "member": m, "truth": true, "q_before": before,
                            "q_after": after, "information": change,
                            "chain_gain": gain[m],
                        }
                    )  # fmt: skip

            d, cert = last["diagnostics"], last["certificate"]
            applicable = bool(last["applicable"])
            psi = np.asarray(d.cond_mass)
            certificates.append(
                {
                    "star": s, "planet": k, "status": int(cert.status),
                    "q_lo": float(cert.q_lo), "q_hi": float(cert.q_hi),
                    "void": not applicable, "truth": true, "refit": bool(last["refit"]),
                    "psi_lo": psi[1:].min(), "psi_hi": psi[1:].max(),
                }
            )  # fmt: skip
            for m in members:
                q_m, mass = d.q[m], d.prior_mass_outside[m]
                bound_lo, bound_hi = widen_by_mass_outside(q_m, q_m, mass)
                n_eff = float(d.n_eff[m])
                refused = not float(d.khat[m]) <= khat_max
                resolution.append(
                    {
                        "star": s, "planet": k, "member": m, "q": float(q_m),
                        "mc_se": float(d.mc_se[m]), "khat": float(d.khat[m]),
                        "khat_max": khat_max, "tail_refused": refused,
                        "n_eff": n_eff,
                        "n_cond": float(d.n_cond[m]), "resolution": 0.5 / n_eff,
                        "clipped": int(np.sum(q_whole[:, m] != q_clip[:, m])),
                        "psi": psi[m], "prior_mass_outside": float(mass),
                        "q_bound_lo": float(bound_lo), "q_bound_hi": float(bound_hi),
                    }
                )  # fmt: skip
                planets.append(
                    (m, psi[m], q_whole[-1, m], applicable, true, refused, s)
                )

    counts = [
        {
            "member": m,
            "predicted_applicable": sum(p[1] for p in planets if p[0] == m),
            "realized_applicable": sum(p[3] for p in planets if p[0] == m),
            "predicted_errors": sum(p[1] - p[2] for p in planets if p[0] == m),
            "realized_errors": sum(p[3] and not p[4] for p in planets if p[0] == m),
            "tail_refused_stars": len({p[6] for p in planets if p[0] == m and p[5]}),
        }
        for m in members
    ]
    n_conjunct = len(claim.conjuncts)
    return Report(
        information=_columns(information, _INFORMATION, chain_gain=n_conjunct),
        certificates=_columns(certificates, _CERTIFICATES),
        resolution=_columns(resolution, _RESOLUTION),
        counts=_columns(counts, _COUNTS),
    )


@eqx.filter_jit
def _found(log, truth, refits, stars, models, claim, slots):
    """Per star, the claim-true planets of the truth its accepted searches found.

    Only planets the star's time-zero refit held no track for are counted; truth
    planet ``k`` is slot ``slots[k]`` of the draws (``associate_planets``).
    """
    rows = log.rows
    accepted = real_rows(log) & (rows.refusal == ACCEPTED)

    def one(refit, truth_s, star):
        mine = accepted & (rows.star == refit.star_index)
        found = detected_slots(log, mine, models, truth_s.exists.shape[-1])
        held = jnp.ones(truth_s.exists.shape, float)
        for conjunct in claim.conjuncts:
            held = held * conjunct.prob(truth_s, star)
        claim_true = (held[0] > 0.5) & truth_s.exists[0]
        untracked = ~planet_tracks(refit.atoms)[slots]
        return jnp.sum(found & untracked & claim_true)

    return jax.vmap(one)(refits, truth, stars)


def found_count(log: ObservationLog, world: SealedWorld, study: Study) -> Array:
    """The planets a run found: claim-true planets of the truth its searches detected.

    Read against the truth, per star: a planet counts when it is claim-true in the
    truth (every conjunct of the claim, the condition included, holds), the star's
    time-zero refit held no track for it (it was undetected: ``Count``'s planets), and
    an accepted row of a searching model detected it (``detected_slots``), released or
    still in flight at the run's end. This is the realized ``Count``;
    ``expected_found`` is the forecast's expectation of it. A planet's slot in a
    payload is the truth's (declared deviation 2), read into the draws' slots by
    ``associate_planets``.

    Args:
        log: the run's log.
        world: the run's sealed world, one world (no world axis).
        study: the study the run belongs to.

    Returns:
        The number of planets found at each star, shape ``(S,)``.

    Raises:
        ValueError: if the truth and the refits hold different numbers of planet
            slots (``associate_planets``).
    """
    truth = world.unseal()
    slots = associate_planets(_star(truth, 0), _star(study.refits, 0).atoms)
    models, claim = study.design.models, study.claim
    return _found(log, truth, study.refits, study.stars, models, claim, slots)


def _columns(entries: list[dict], names: tuple[str, ...], **widths: int) -> dict:
    """The entries of one table as named columns, ``widths`` naming row columns.

    A column in ``widths`` holds a row of that width per entry, so it keeps its shape
    when the table is empty.
    """
    return {
        name: jnp.asarray(
            np.asarray(
                [e[name] for e in entries], dtype=float if name in widths else None
            )
        ).reshape(len(entries), *((widths[name],) if name in widths else ()))
        for name in names
    }


_STATUS = {
    POSITIVE: "positive",
    NEGATIVE: "negative",
    INAPPLICABLE: "inapplicable",
    UNRESOLVED: "unresolved",
}


def _cell(name: str, value) -> str:
    """One value of a table column as text."""
    if name == "status":
        return _STATUS[int(value)]
    if value.dtype == bool:
        return "yes" if value else "no"
    if np.issubdtype(value.dtype, np.integer):
        return str(int(value))
    text = f"{float(value):.4f}"
    return "0.0000" if text == "-0.0000" else text  # a value that rounds to zero


def _table(title: str, columns: dict) -> str:
    """A table of named columns as aligned text; a column of rows is spread out."""
    text = {}
    for name, column in columns.items():
        column = np.asarray(column)
        if column.ndim == 2:
            for c in range(column.shape[1]):
                text[f"{name}_{c}"] = [_cell(name, v) for v in column[:, c]]
        else:
            text[name] = [_cell(name, v) for v in column]
    width = {n: max([len(n), *(len(v) for v in vals)]) for n, vals in text.items()}
    n_row = len(next(iter(text.values()), []))
    lines = [title, "  ".join(n.rjust(width[n]) for n in text)]
    lines += ["  ".join(text[n][i].rjust(width[n]) for n in text) for i in range(n_row)]
    return "\n".join(lines)


def _totals(information: dict) -> dict:
    """Each planet's information summed over its stages, per member.

    The sum runs up to the run's end (``information_at_end``) and over every stage
    (``information``).
    """
    info = {name: np.asarray(column) for name, column in information.items()}
    keys = sorted({*zip(info["star"], info["planet"], info["member"], strict=True)})
    picks = [
        (info["star"] == s) & (info["planet"] == k) & (info["member"] == m)
        for s, k, m in keys
    ]
    end_d = info["time_d"][info["end"]].max(initial=0.0)
    at_end = info["time_d"] <= end_d
    return {
        "star": np.asarray([key[0] for key in keys], int),
        "planet": np.asarray([key[1] for key in keys], int),
        "member": np.asarray([key[2] for key in keys], int),
        "truth": np.asarray([info["truth"][pick].any() for pick in picks], bool),
        "information_at_end": np.asarray(
            [info["information"][pick & at_end].sum() for pick in picks]
        ),
        "information": np.asarray([info["information"][pick].sum() for pick in picks]),
        "chain_gain": np.asarray(
            [info["chain_gain"][pick].sum(axis=0) for pick in picks]
        ).reshape(len(keys), info["chain_gain"].shape[1]),
    }


def format_report(report: Report) -> str:
    """The report's tables as text, with each planet's information in total.

    Args:
        report: the report.

    Returns:
        The tables, one after another.
    """
    return "\n\n".join(
        [
            _table("Information by stage (nats)", report.information),
            _table("Information in total (nats)", _totals(report.information)),
            _table("Certificates", report.certificates),
            _table("Resolution", report.resolution),
            _table("Counts", report.counts),
        ]
    )
