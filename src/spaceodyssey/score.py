"""Scoring a run: the log read by the verdict, set against the unsealed truth.

``score`` is the only reader of the truth. Each star's log is replayed through the
study's roles stage by stage, a stage ending at each release time of the star's rows,
and every planet of the truth is scored at each stage. It returns the tables milestone 1
needs, as generic columns: the information table, the certificate table, the resolution
numbers and the predicted and realized counts. ``format_report`` prints them.

The claim is read two ways. Its information is that of the whole conjunction, the
planet's claim bit: a claim-true planet scores ``ln q`` and a look-alike ``ln(1 - q)``,
each ``q`` clipped to the cloud's resolution first, and a claim-true planet's
information is split over the conjuncts by ``chain_gain``. Its certificate is that of
the claim as stated, conditional when ``claim.given`` is set; a planet whose truth fails
the condition is void, recorded in its own column beside the certificate.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Int

from spaceodyssey._standin.diagnostics import khat_threshold
from spaceodyssey.belief import Belief, absorb, draw_claim_mass, replay, replay_belief
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
from spaceodyssey.types import ACCEPTED, Atoms, ObservationLog
from spaceodyssey.world import SealedWorld


class Report(eqx.Module):
    """The tables of one scored run, each a dict of named columns of equal length.

    The information and resolution tables are read by the verdict members, roles 1 to
    ``R - 1`` (column ``member``); the forecast does not judge.

    Attributes:
        information: one entry per star, planet, stage and member: the stage
            (``stage``, from 1) and the time it ends (``time_d``), whether the planet
            is claim-true (``truth``), the whole claim's clipped ``q`` before and after
            the stage, the signed ``information`` (the change of ``ln q`` for a
            claim-true planet, of ``ln(1 - q)`` for a look-alike), and the
            ``chain_gain`` of each conjunct, shape ``(n, C)``, NaN for a look-alike.
        certificates: one entry per star and planet, at the last stage: the
            certificate's ``status``, ``q_lo`` and ``q_hi``; ``void`` (the truth fails
            the claim's condition); ``truth``; the refit trigger ``refit``; and the
            candidacy range over the members, ``psi_lo`` and ``psi_hi``.
        resolution: one entry per star, planet and member, at the last stage: the
            claim's ``q``, its ``mc_se``, ``khat`` against ``khat_max``, ``n_eff``, the
            clip ``resolution`` ``1 / (2 n_eff)`` and the number of stages whose whole
            claim ``q`` it ``clipped``, the condition mass ``psi``, and the member's
            ``prior_mass_outside`` ``m`` with the hard bound it puts on ``q``,
            ``[(1 - m) q, (1 - m) q + m]`` (``q_bound_lo``, ``q_bound_hi``).
        counts: one entry per member: the predicted applicable count ``sum psi``
            beside the realized one, and the predicted applicable errors
            ``sum (psi - q)``, with ``q`` the whole claim's, beside the realized ones
            (applicable planets that are not claim-true).
    """

    information: dict[str, Array]
    certificates: dict[str, Array]
    resolution: dict[str, Array]
    counts: dict[str, Array]


_INFORMATION = (
    "star", "planet", "stage", "time_d", "member", "truth", "q_before", "q_after",
    "information", "chain_gain",
)  # fmt: skip
_CERTIFICATES = (
    "star", "planet", "status", "q_lo", "q_hi", "void", "truth", "refit", "psi_lo",
    "psi_hi",
)  # fmt: skip
_RESOLUTION = (
    "star", "planet", "member", "q", "mc_se", "khat", "khat_max", "n_eff",
    "resolution", "clipped", "psi", "prior_mass_outside", "q_bound_lo", "q_bound_hi",
)  # fmt: skip
_COUNTS = (
    "member", "predicted_applicable", "realized_applicable", "predicted_errors",
    "realized_errors",
)  # fmt: skip

_replay = eqx.filter_jit(replay)
_absorb = eqx.filter_jit(absorb)
_replay_belief = eqx.filter_jit(replay_belief)


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
    roles = range(before.log_prior.shape[0])
    return jnp.stack([chain_gain(before, after, whole, star, planet, r) for r in roles])


def _star(tree, s: int):
    """Star ``s`` of a pytree with a leading star axis."""
    return jax.tree.map(lambda leaf: leaf[s], tree)


def _stage_beliefs(study: Study, log: ObservationLog, s: int):
    """The times that end star ``s``'s stages and its belief after each.

    Stage 0 ends at time zero, with the grant; each later stage ends at a release time
    of one of the star's accepted rows. The replay is brought from one stage to the
    next by ``absorb``, so each row goes through it once.
    """
    star_col, refusal, release_d = (
        np.asarray(x) for x in (log.rows.star, log.rows.refusal, log.rows.release_d)
    )
    real = np.arange(star_col.shape[0]) < int(log.count)
    mine = real & (star_col == s) & (refusal == ACCEPTED)
    times = [0.0, *sorted({float(t) for t in release_d[mine] if t > 0.0})]

    models, roles = study.design.models, study.roles
    refit, pool = _star(study.refits, s), _star(study.pool, s)
    star = _star(study.stars, s)
    r = _replay(refit, log, models, star, jnp.asarray(0.0))
    beliefs = [_replay_belief(r, roles, models, star, pool)]
    for time_d in times[1:]:
        r = _absorb(r, log, models, star, jnp.asarray(time_d))
        beliefs.append(_replay_belief(r, roles, models, star, pool))
    return times, beliefs


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


def score(log: ObservationLog, world: SealedWorld, study: Study) -> Report:
    """Score a run's log against the truth of its world.

    Each star's rows are replayed through the study's roles, stage by stage (stage 0
    the grant, then one stage per release time of the star's accepted rows), and each
    planet the truth holds is scored at every stage by every verdict member, each
    planet read from the slot of the draws that ``associate_planets`` gives it. The
    certificates and the refit trigger gate the Pareto shape
    at ``khat_threshold`` of the refit's draw count. All rows are read, those still in
    flight when the run ended included.

    Args:
        log: the run's log.
        world: the run's sealed world, one world (no world axis).
        study: the study the run belongs to.

    Returns:
        The report.

    Raises:
        ValueError: if the planets of the truth cannot be associated with the draws'
            slots (``associate_planets``).
    """
    truth = world.unseal()
    claim = study.claim
    n_star, n_role = study.prior_mass_outside.shape
    khat_max = khat_threshold(study.refits.atoms.cloud.log_weights.shape[-1])
    members = range(1, n_role)

    information, certificates, resolution, planets = [], [], [], []
    for s in range(n_star):
        times, beliefs = _stage_beliefs(study, log, s)
        star, outside = _star(study.stars, s), study.prior_mass_outside[s]
        truth_s = _star(truth, s)
        slots = associate_planets(truth_s, _star(study.refits, s).atoms)
        truth_b = _truth_belief(truth_s, _star(study.pool, s), n_role)
        for k in np.flatnonzero(np.asarray(truth.exists[s, 0])):
            planet, truth_planet = slots[k], jnp.asarray(k)
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
                for b in beliefs
            ]
            last, true = stages[-1], bool(stages[-1]["truth"])
            q_whole = np.asarray([x["q"] for x in stages])  # (stage, R)
            q_clip = np.asarray([x["q_clipped"] for x in stages])
            for j in range(1, len(stages)):
                gain = np.asarray(
                    _chain_gains(beliefs[j - 1], beliefs[j], claim, star, planet)
                )
                if not true:
                    gain = np.full_like(gain, np.nan)  # split only for a true claim
                for m in members:
                    before, after = q_clip[j - 1, m], q_clip[j, m]
                    change = _log_score(after, true) - _log_score(before, true)
                    information.append(
                        {
                            "star": s, "planet": k, "stage": j, "time_d": times[j],
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
                resolution.append(
                    {
                        "star": s, "planet": k, "member": m, "q": float(q_m),
                        "mc_se": float(d.mc_se[m]), "khat": float(d.khat[m]),
                        "khat_max": khat_max, "n_eff": n_eff, "resolution": 0.5 / n_eff,
                        "clipped": int(np.sum(q_whole[:, m] != q_clip[:, m])),
                        "psi": psi[m], "prior_mass_outside": float(mass),
                        "q_bound_lo": float(bound_lo), "q_bound_hi": float(bound_hi),
                    }
                )  # fmt: skip
                planets.append((m, psi[m], q_whole[-1, m], applicable, true))

    counts = [
        {
            "member": m,
            "predicted_applicable": sum(p[1] for p in planets if p[0] == m),
            "realized_applicable": sum(p[3] for p in planets if p[0] == m),
            "predicted_errors": sum(p[1] - p[2] for p in planets if p[0] == m),
            "realized_errors": sum(p[3] and not p[4] for p in planets if p[0] == m),
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
    return f"{float(value):.4f}"


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
    """Each planet's information summed over its stages, per member."""
    info = {name: np.asarray(column) for name, column in information.items()}
    keys = sorted({*zip(info["star"], info["planet"], info["member"], strict=True)})
    picks = [
        (info["star"] == s) & (info["planet"] == k) & (info["member"] == m)
        for s, k, m in keys
    ]
    return {
        "star": np.asarray([key[0] for key in keys], int),
        "planet": np.asarray([key[1] for key in keys], int),
        "member": np.asarray([key[2] for key in keys], int),
        "truth": np.asarray([info["truth"][pick].any() for pick in picks], bool),
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
