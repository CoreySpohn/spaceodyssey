"""Row-level refresh eligibility on the compiled run and keyed replay."""

import functools
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import Audience, CompiledExecutor, index_world, replay_run
from spaceodyssey.core.enums import I64
from spaceodyssey.core.records import refit_rows
from spaceodyssey.studies import builder, toy
from tests.core.helpers import worlds
from tests.core.refresh_helpers import masked


class DecliningObserver(toy.ToyRefitMean):
    """Refit measurements except those arriving on one channel."""

    declined_channel: jax.Array = eqx.field(
        kw_only=True, converter=lambda x: jnp.asarray(x, I64)
    )

    def refits_on(self, rec):
        return rec.channel_index != self.declined_channel


@functools.cache
def _compiled_toy(declined_channel):
    """Give each channel its own subject so product subjects identify refreshes."""
    study = builder.survey_refresh(
        observer=DecliningObserver(declined_channel=declined_channel),
        policies=(toy.FixedPlanPolicy(plan=((0, 10), (1, 11))),),
    )
    truths, roots = worlds(1)
    result = CompiledExecutor().run(study, truths, roots, 40, refit=masked(), chunk=7)
    return study, roots[0], index_world(result.carry, 0), index_world(result.trace, 0)


def run_toy(*, observer, executor):
    """Run once, then optionally replay the same log through the real boundary."""
    study, root, carry, trace = _compiled_toy(int(observer.declined_channel))
    belief, pbelief = carry.belief, carry.pbelief
    if executor == "replay":
        replayed = replay_run(study, root, carry, trace, refit=masked())
        belief, pbelief = replayed.belief, replayed.pbelief
    return SimpleNamespace(study=study, carry=carry, belief=belief, pbelief=pbelief)


def refreshed_channels(run):
    """Channels whose dedicated subjects received a fitted belief."""
    return set(np.flatnonzero(np.asarray(run.belief.n)))


def logged_channels(run):
    """Admission channel indices of measurement rows, excluding products."""
    log = run.carry.log
    take = np.asarray(log.valid & (log.kind == toy.RECORD_MEASUREMENT))
    return set(np.asarray(log.channel_index)[take].tolist())


@pytest.mark.medium(
    reason="compiles one refresh driver, about 7 s cold, then replays it"
)
@pytest.mark.parametrize("executor", ["compiled", "replay"])
def test_a_declined_row_marks_no_refresh_in_either_executor(executor):
    """Fails if the compiled and replay executors disagree on which rows refresh a
    subject.
    """
    run = run_toy(observer=DecliningObserver(declined_channel=1), executor=executor)
    assert refreshed_channels(run) == {0} and logged_channels(run) == {0, 1}
    np.testing.assert_array_equal(run.belief.n, [1, 0, 0, 0])
    np.testing.assert_array_equal(run.pbelief.n, run.belief.n)
    assert not bool(run.carry.overflow | run.carry.stalled | run.carry.violation)
    assert bool(run.carry.done)
    for audience in (Audience.OBSERVER, Audience.POLICY):
        rows = refit_rows(run.carry.log, run.study.caps, audience, 1)
        assert int(rows.valid.sum()) == 1
        assert int(rows.channel_index[jnp.argmax(rows.valid)]) == 1
