"""A scripted policy that names programs, and the two-program toy study."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import F64, I64, Disposition, EffectKind, Scope
from spaceodyssey.core.snapshot import EffectTable
from spaceodyssey.studies import toy

SCRIPT_ROWS = 8
NONE = -1  # a script row that matches no decision


def row(n, kind, *, subject=0, channel=10, program=0, scope=0, disp=0, tick=0):
    """One scripted effect: the decision index it answers, then its fields."""
    return (n, int(kind), subject, channel, program, int(scope), int(disp), tick)


def observe(n, subject, channel, program):
    """An Observe of `subject` on channel id `channel`, charged to `program`."""
    return row(n, EffectKind.OBSERVE, subject=subject, channel=channel, program=program)


def stop(n, scope, disp, program=0):
    """A Stop of `scope` with `disp`; `program` names the program of a program Stop."""
    return row(n, EffectKind.STOP, scope=scope, disp=disp, program=program)


def wait(n, ticks):
    """A Wait for `ticks` after the decision."""
    return row(n, EffectKind.WAIT, tick=ticks)


def cancel(n, lane=0):
    """A live Cancel of the lane's current occupancy."""
    return row(n, EffectKind.CANCEL, tick=0, subject=lane)


def end(n):
    """Drain the campaign once the running occupancies settle."""
    return stop(n, Scope.CAMPAIGN, Disposition.DRAIN)


def _as_script(rows):
    out = np.full((SCRIPT_ROWS, 8), NONE, np.int64)
    for i, r in enumerate(rows):
        out[i] = r
    return jnp.asarray(out, I64)


class ScriptedPrograms(toy.NullPolicy):
    """Emits the scripted effects of decision `view.n_dec`, in script order.

    Each row of `script` is (decision, kind, subject, channel id, program, scope,
    disposition, tick); the e-th row of a decision becomes effect row e, in a group of
    its own. A cancel names its lane in the subject column.
    """

    script: object = eqx.field(converter=_as_script)

    def declare(self):
        """The programs and stops the script names, for the pre-flight."""
        rows = np.asarray(self.script).tolist()
        names = {
            r[4]
            for r in rows
            if r[1] == EffectKind.OBSERVE
            or (r[1] == EffectKind.STOP and r[5] == Scope.PROGRAM)
        }
        stops = tuple(
            (Scope(r[5]), Disposition(r[6])) for r in rows if r[1] == EffectKind.STOP
        )
        return Declaration(programs=tuple(sorted(names)), stops=stops)

    def decide(self, pstate, view, study, key):
        """Emit the script rows of this decision."""
        caps = study.caps
        n = view.n_dec
        match = self.script[:, 0] == n
        rank = jnp.cumsum(match) - 1
        cols = {
            k: []
            for k in (
                "kind",
                "subject",
                "channel_index",
                "group",
                "tick",
                "lane",
                "scope",
                "disposition",
                "program",
            )
        }
        for e in range(caps.n_effects):
            pick = match & (rank == e)
            has = pick.any()
            r = self.script[jnp.argmax(pick)]
            is_cancel = r[1] == EffectKind.CANCEL
            ids = study.channels.id
            cols["kind"].append(jnp.where(has, r[1], EffectKind.NONE))
            cols["subject"].append(jnp.where(is_cancel, 0, jnp.maximum(r[2], 0)))
            cols["channel_index"].append(jnp.argmax(ids == r[3]).astype(I64))
            cols["group"].append(n * caps.n_effects + e)
            cols["tick"].append(jnp.maximum(r[7], 0) * (r[1] == EffectKind.WAIT))
            cols["lane"].append(jnp.where(is_cancel, jnp.maximum(r[2], 0), 0))
            cols["scope"].append(jnp.maximum(r[5], 0))
            cols["disposition"].append(jnp.maximum(r[6], 0))
            cols["program"].append(jnp.maximum(r[4], 0))
        eff = EffectTable(
            **{k: jnp.asarray(v, I64) for k, v in cols.items() if k != "program"},
            disruptive=jnp.zeros(caps.n_effects, jnp.bool_),
            program=jnp.asarray(cols["program"], I64) if caps.n_programs > 1 else None,
        )
        # a Wait is relative to the decision's tick
        eff = eqx.tree_at(
            lambda t: t.tick,
            eff,
            jnp.where(eff.kind == EffectKind.WAIT, view.tick + eff.tick, eff.tick),
        )
        return eff, pstate, jnp.asarray(0.0, F64)


def two_program_study(*rows, n_programs=2, **overrides):
    """The toy study with `n_programs` programs and a scripted policy."""
    caps = Capacities(n_programs=n_programs, n_effects=3)
    policy = ScriptedPrograms(script=rows)
    return toy.survey_1(caps=caps, policies=(policy,), **overrides)
