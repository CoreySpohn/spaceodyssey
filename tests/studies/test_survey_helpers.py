"""Action comparison controls, including requests that produced no acquisition."""

import dataclasses
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.carry import empty_trace
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.core.records import empty_log
from spaceodyssey.core.snapshot import EffectTable
from tests.studies.survey_helpers import ActionRun, same_actions

FIELDS = tuple(EffectTable.__dataclass_fields__)


def rejected_run():
    """One rejected decision with no acquisition rows to reveal its parameters."""
    effect = EffectTable(
        **{
            name: np.zeros(2, dtype=bool if name == "disruptive" else int)
            for name in FIELDS
        }
    )
    effect.kind[0] = EffectKind.OBSERVE
    caps = Capacities(n_log=0)
    trace = dataclasses.replace(
        empty_trace(caps),
        handled=np.asarray(True),
        wake=np.asarray(True),
        admit_code=np.ones(2, int),
    )

    def batched(tree):
        return jax.tree_util.tree_map(lambda x: np.asarray(x)[None], tree)

    result = SimpleNamespace(
        trace=batched(batched(trace)),
        carry=SimpleNamespace(log=batched(empty_log(caps))),
    )
    return ActionRun(result, (((None, effect),),))


@pytest.mark.parametrize("field", (*FIELDS, "trace.disruptive"))
def test_complete_effects_distinguish_rejected_requests(field):
    """Fails if any requested parameter, including disruptive, is omitted."""
    baseline = rejected_run()
    assert same_actions(baseline, baseline, 1)
    if field == "trace.disruptive":
        trace = baseline.result.trace
        flags = np.asarray(trace.eff_disruptive).copy()
        flags[0, 0, 0] = True
        result = SimpleNamespace(
            trace=dataclasses.replace(trace, eff_disruptive=flags),
            carry=baseline.result.carry,
        )
        assert not same_actions(
            baseline, dataclasses.replace(baseline, result=result), 1
        )
        return
    effect = baseline.decisions[0][0][1]
    value = getattr(effect, field).copy()
    value[0] = True if field == "disruptive" else value[0] + 1
    mutant = dataclasses.replace(effect, **{field: value})
    changed = dataclasses.replace(baseline, decisions=(((None, mutant),),))
    assert not same_actions(baseline, changed, 1)
