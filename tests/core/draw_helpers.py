import dataclasses

import jax.random as jr
import numpy as np

from spaceodyssey.core.carry import Carry, Trace
from spaceodyssey.core.enums import EventKind, RecordStatus
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from tests.core.corpus import PERTURBATIONS
from tests.core.leak_helpers import make_worlds
from tests.core.mutants import (
    MutantCompiledExecutor,
    MutantHostExecutor,
    PaddedCompiledExecutor,
    PaddedHostExecutor,
)


@dataclasses.dataclass
class World:
    carry: Carry
    trace: Trace


def fold_n_dec(key, n_dec):
    """The mutant: a measurement key that also depends on the decision count."""
    return jr.fold_in(key, n_dec)


def draw_table(world):
    """Index raw variates by subject, acquisition index, channel, revision, and status.

    Match partial records with completed records of the same identity.
    """
    c = world.carry if isinstance(world, World) else world
    log = c.log
    out = {}
    for i in np.flatnonzero(np.asarray(log.valid)):
        status = int(log.status[i])
        status = RecordStatus.COMPLETED if status == RecordStatus.PARTIAL else status
        key = (
            int(log.subject[i]),
            int(log.acquisition_index[i]),
            int(log.channel[i]),
            int(log.revision[i]),
            status,
        )
        out[key] = float(log.payload[i, 1])
    return out


def compare_draws(base, other):
    """(n_compared, n_baseline, n_changed) over the keys both runs hold, all worlds."""
    n_cmp = n_base = n_chg = 0
    for b, o in zip(base, other, strict=False):
        tb, to = draw_table(b), draw_table(o)
        shared = tb.keys() & to.keys()
        n_base += len(tb)
        n_cmp += len(shared)
        n_chg += sum(tb[k] != to[k] for k in shared)
    return n_cmp, n_base, n_chg


def _decisions_by_release(world):
    """Index policy draws by the released record that woke each decision."""
    t = world.trace
    out = {}
    for i in range(len(np.asarray(t.wake))):
        if bool(t.wake[i]) and int(t.kind[i]) == EventKind.RELEASE:
            out[(int(t.subject[i]), int(t.acquisition_index[i]), int(t.channel[i]))] = (
                float(t.pol_u[i])
            )
    return out


def policy_draws_changed(base, other):
    n = 0
    for b, o in zip(base, other, strict=False):
        db, do = _decisions_by_release(b), _decisions_by_release(o)
        n += sum(db[k] != do[k] for k in db.keys() & do.keys())
    return n


def run_study(make_study, executor, n_seeds, n_steps=40, hook=None, padded=False):
    study = make_study()
    scan = MutantCompiledExecutor() if hook is not None else CompiledExecutor()
    host = MutantHostExecutor() if hook is not None else HostExecutor()
    if padded:
        scan, host = PaddedCompiledExecutor(), PaddedHostExecutor()
    truths, roots = make_worlds(n_seeds, study.caps)
    if executor == "scan":
        cs, ts = scan.run(study, truths, roots, n_steps)
        return [World(index_world(cs, w), index_world(ts, w)) for w in range(n_seeds)]
    out = []
    for w in range(n_seeds):
        c, t = host.run(study, index_world(truths, w), roots[w], n_steps)
        out.append(World(c, t))
    return out


def run_pair(name, executor, n_seeds, mutant=False):
    """Run both variants on the same worlds, optionally with mutated draw keys."""
    make_base, make_other = PERTURBATIONS[name]
    hook = fold_n_dec if mutant else None
    return (
        run_study(make_base, executor, n_seeds, hook=hook, padded=False),
        run_study(
            make_other, executor, n_seeds, hook=hook, padded=name == "extra_kinds"
        ),
    )
