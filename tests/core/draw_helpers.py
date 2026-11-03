import dataclasses

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core.carry import Carry, Trace
from spaceodyssey.core.enums import EffectKind, RecordStatus
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import PERTURBATIONS, PLAN5
from tests.core.leak_helpers import make_worlds
from tests.core.mutants import (
    CountKeyCompiledExecutor,
    CountKeyHostExecutor,
    MutantCompiledExecutor,
    MutantHostExecutor,
    PaddedCompiledExecutor,
    PaddedHostExecutor,
)


class DrawingSpinPolicy(kinds.SpinWaitPolicy):
    """SpinWaitPolicy that reports its policy variate."""

    def decide(self, pstate, view, study, key):
        eff, ps, _ = super().decide(pstate, view, study, key)
        return eff, ps, jr.uniform(key, dtype=jnp.float64)


class WaitingReactivePolicy(kinds.ReactivePolicy):
    """ReactivePolicy that reports its variate and adds a Wait at one decision.

    The extra Wait wakes one more decision, which the reactive rule leaves idle, so the
    schedule of every other decision is unchanged. `wait_at` of -1 adds nothing.
    """

    wait_at: int = eqx.field(default=-1)
    wait_ticks: int = eqx.field(default=7)

    def decide(self, pstate, view, study, key):
        eff, ps, _ = super().decide(pstate, view, study, key)
        add = view.n_dec == self.wait_at
        eff = dataclasses.replace(
            eff,
            kind=eff.kind.at[1].set(jnp.where(add, EffectKind.WAIT, EffectKind.NONE)),
            tick=eff.tick.at[1].set(view.tick + self.wait_ticks),
        )
        return eff, ps, jr.uniform(key, dtype=jnp.float64)


def _waiting(wait_at):
    return lambda: kinds.survey_1(
        policies=(WaitingReactivePolicy(plan=PLAN5, wait_at=wait_at),)
    )


PAIRS = {**PERTURBATIONS, "insert_wait_alongside": (_waiting(-1), _waiting(0))}


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


def _compare_tables(base, other, table, min_matches):
    """(n_compared, n_baseline, n_changed); compared values must be finite.

    A variate that is NaN on both sides would otherwise count as changed, since
    NaN != NaN, so a corrupted report could pass for a key-dependence failure.
    """
    n_cmp = n_base = n_chg = 0
    for b, o in zip(base, other, strict=False):
        tb, to = table(b), table(o)
        shared = tb.keys() & to.keys()
        for k in shared:
            assert np.isfinite(tb[k]) and np.isfinite(to[k]), (
                f"draw {k} is not finite: {tb[k]} vs {to[k]}"
            )
        n_base += len(tb)
        n_cmp += len(shared)
        n_chg += sum(tb[k] != to[k] for k in shared)
    assert n_cmp >= min_matches, (
        f"only {n_cmp} matched identities were compared; need {min_matches}"
    )
    return n_cmp, n_base, n_chg


def compare_draws(base, other, min_matches=2):
    """(n_compared, n_baseline, n_changed) over the keys both runs hold, all worlds."""
    return _compare_tables(base, other, draw_table, min_matches)


def policy_draw_table(world):
    """Index policy variates by decision tick and ordinal within the tick."""
    t = world.trace
    out, seen = {}, {}
    for i in range(len(np.asarray(t.wake))):
        if bool(t.wake[i]):
            tick = int(t.tick[i])
            ordinal = seen.get(tick, 0)
            seen[tick] = ordinal + 1
            out[(tick, ordinal)] = float(t.pol_u[i])
    return out


def compare_policy_draws(base, other, min_matches=2):
    """(n_compared, n_baseline, n_changed) over decisions both runs hold, all worlds."""
    return _compare_tables(base, other, policy_draw_table, min_matches)


def policy_draws_changed(base, other):
    return compare_policy_draws(base, other)[2]


def run_study(
    make_study,
    executor,
    n_seeds,
    n_steps=40,
    hook=None,
    padded=False,
    policy_mutant=False,
):
    study = make_study()
    scan = MutantCompiledExecutor() if hook is not None else CompiledExecutor()
    host = MutantHostExecutor() if hook is not None else HostExecutor()
    if policy_mutant:
        scan, host = CountKeyCompiledExecutor(), CountKeyHostExecutor()
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


def run_pair(name, executor, n_seeds, mutant=False, policy_mutant=False):
    """Run both variants on the same worlds, optionally with mutated draw keys."""
    make_base, make_other = PAIRS[name]
    hook = fold_n_dec if mutant else None
    return (
        run_study(make_base, executor, n_seeds, hook=hook, policy_mutant=policy_mutant),
        run_study(
            make_other,
            executor,
            n_seeds,
            hook=hook,
            padded=name == "extra_kinds",
            policy_mutant=policy_mutant,
        ),
    )
