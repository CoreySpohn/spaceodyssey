"""Shared cached corpus runs, including private-event probes, and the fault fixture."""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey import core
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.conformance import run_case
from spaceodyssey.core.conformance.corpus import N_STEPS, make_worlds
from spaceodyssey.core.enums import (
    I64,
    Audience,
    Disposition,
    EventKind,
    Scope,
    Stage,
)
from spaceodyssey.core.kinds import AbstractProcess
from spaceodyssey.core.snapshot import _digest
from spaceodyssey.studies import builder
from spaceodyssey.studies.conformance import CORPUS_BY_NAME

N_WORLDS = 2


def digest_ints(words, salt=0):
    """Order-sensitive 64-bit digest of an integer array, as an int64."""
    h = _digest(jnp.asarray(words), salt)
    return jax.lax.bitcast_convert_type(h, I64)


@functools.cache
def runs(name, executor):
    """The runs of one case on one executor, with the private-event probe."""
    return tuple(run_case(CORPUS_BY_NAME[name], executor, n_worlds=N_WORLDS))


# ---- fingerprints ----------------------------------------------------------------

_fingerprint_j = eqx.filter_jit(core.snapshot_fingerprint)


def _view_fingerprint(carry, study, audience):
    view = core.snapshot_view(carry, audience, study.caps)
    return int(_fingerprint_j(view, study.observer.layout(study.caps)))


def run_fingerprints(run):
    """Fingerprints of one corpus run: every decision's, and the final views'.

    `decisions` is the digest of the snapshot fingerprints the trace recorded at its
    woken steps, in order; `policy` and `observer` fingerprint the final carry's
    masked view for each audience.
    """
    trace = run.trace
    woke = np.asarray(trace.wake).astype(bool)
    fps = np.asarray(trace.snapshot_fp)[woke]
    return {
        "decisions": int(digest_ints(fps)),
        "n_decisions": int(fps.size),
        "policy": _view_fingerprint(run.carry, run.study, core.Audience.POLICY),
        "observer": _view_fingerprint(run.carry, run.study, Audience.OBSERVER),
    }


# ---- the FAULT fixture ------------------------------------------------------------

# a live fault on the first lane-0 occupancy (generation 1, admitted at tick 0, due
# to settle at 110), then a stale one naming the same retired generation
FAULTS = ((50, 0, 0, 1), (200, 0, 0, 1))


class ScriptedFaults(AbstractProcess):
    """Seeds `FAULT` rows at `(tick, lane, channel_index, gen)`, settle stage.

    With `close_lane` it also seeds a lane `Stop` at each fault's tick: a mutant
    that closes the lane on a fault, which the fault pin must catch.
    """

    rows: tuple = eqx.field(static=True)
    tie: int = eqx.field(static=True)
    channel_ids: tuple = eqx.field(static=True)
    close_lane: bool = eqx.field(static=True, default=False)

    def seed(self, carry, caps):
        """Queue every scripted fault, and its lane Stop under the mutant."""
        for tick, lane, channel_index, gen in self.rows:
            carry = core.enqueue(
                carry,
                True,
                kind=EventKind.FAULT,
                tick=tick,
                stage=Stage.SETTLE,
                lane=lane,
                channel_index=channel_index,
                channel=self.channel_ids[channel_index],
                gen=gen,
                tie=self.tie,
                caps=caps,
            )
            if self.close_lane:
                carry = core.enqueue(
                    carry,
                    True,
                    kind=EventKind.STOP,
                    tick=tick,
                    stage=Stage.SETTLE,
                    lane=lane,
                    scope=Scope.LANE,
                    disposition=Disposition.CLOSE,
                    tie=self.tie + 1,
                    caps=caps,
                )
        return carry


def fault_study(close_lane=False):
    """The toy survey-1 study with the scripted faults of `FAULTS`."""
    base = builder.survey_1()
    process = ScriptedFaults(
        rows=FAULTS,
        tie=int(core.fault_tie(base.channels, 0)),
        channel_ids=tuple(int(x) for x in np.asarray(base.channels.id)),
        close_lane=close_lane,
    )
    return eqx.tree_at(lambda s: s.process, base, process)


@functools.cache
def fault_run(executor, close_lane=False):
    """The fault fixture run on one executor, two worlds, the corpus step budget."""
    study = fault_study(close_lane)
    truths, roots = make_worlds(study, N_WORLDS)
    exe = HostExecutor() if executor == "host" else CompiledExecutor()
    return exe.run_worlds(study, truths, roots, N_STEPS)
