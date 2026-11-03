"""Relations compare the actual view handed to every policy decision."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import pytest

from spaceodyssey.core import battery, snapshot, transaction, wake_by_audience
from spaceodyssey.core.battery import metamorphic
from spaceodyssey.core.enums import I64
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from tests.core.battery_helpers import PLAN, AbsolutePlanPolicy, survey
from tests.core.corpus import frozen
from tests.core.helpers import worlds
from tests.core.mutants import _clone


def relabeled(study, mapping):
    renamed = eqx.tree_at(
        lambda s: s.channels.id,
        study,
        jnp.asarray([mapping[int(c)] for c in study.channels.id], I64),
    )
    return dataclasses.replace(
        renamed,
        agenda=dataclasses.replace(
            renamed.agenda,
            policies=(frozen(plan=tuple((s, mapping[c]) for s, c in PLAN)),),
        ),
    )


@pytest.mark.parametrize("relation", ("relabel", "latency"))
def test_relation_accepts_masked_views_from_host(relation):
    truths, roots = worlds(1)
    if relation == "relabel":
        study = survey()
        mapping = {10: 30, 11: 20, 12: 10}
        result = battery.relabel_ids(
            study,
            relabeled(study, mapping),
            mapping,
            truths,
            roots,
            40,
            executor=HostExecutor(),
        )
    else:
        result = battery.latency_shift(
            survey(AbsolutePlanPolicy()),
            truths,
            roots,
            40,
            delay_s=2,
            executor=HostExecutor(),
        )
    assert result, result.diagnostic


def changed_view_transition(content):
    def wrong_rows(log, audience):
        hidden = (
            (log.channel >= 30)
            | (log.start_tick >= 1000)
            | (log.available_tick - log.end_tick > 20)
        )
        return snapshot.visible_rows(log, audience) & ~hidden

    def changed_payload(carry, audience, caps):
        original = snapshot.snapshot_view(carry, audience, caps)
        log = original.log
        changed = log.valid & (
            (log.channel >= 30)
            | (log.start_tick >= 1000)
            | (log.available_tick - log.end_tick > 20)
        )
        return dataclasses.replace(
            original,
            log=dataclasses.replace(log, payload=log.payload.at[:, 0].add(changed)),
        )

    view = (
        changed_payload
        if content
        else _clone(snapshot.snapshot_view, visible_rows=wrong_rows)
    )
    changed = _clone(
        transaction.transition,
        decide=_clone(transaction.decide, snapshot_view=view),
        decision_fingerprint=_clone(
            transaction.decision_fingerprint, snapshot_view=view
        ),
    )
    changed.__kwdefaults__ = transaction.transition.__kwdefaults__
    return changed


@pytest.mark.parametrize("content", (False, True), ids=("rows", "payload"))
@pytest.mark.parametrize("relation", ("relabel", "epoch", "latency"))
def test_relation_rejects_changed_policy_view(relation, content, monkeypatch):
    changed = changed_view_transition(content)

    class HiddenRows(CompiledExecutor):
        def transition(self, carry, i, root, study, truth, *, wake_fn=wake_by_audience):
            return changed(carry, i, root, study, truth, wake_fn=wake_fn)

    truths, roots = worlds(1)
    study = survey()
    if relation == "relabel":
        mapping = {10: 30, 11: 20, 12: 10}
        result = battery.relabel_ids(
            study,
            relabeled(study, mapping),
            mapping,
            truths,
            roots,
            40,
            executor=HiddenRows(),
        )
    elif relation == "epoch":
        monkeypatch.setattr(metamorphic, "transition", changed)
        result = battery.shift_epoch(study, truths, roots, 40, 1000)
    else:
        result = battery.latency_shift(
            survey(AbsolutePlanPolicy()),
            truths,
            roots,
            40,
            delay_s=2,
            executor=HiddenRows(),
        )
    assert not result
    assert "view" in result.diagnostic
    assert "view was not captured" not in result.diagnostic
    if not content:
        assert "n_rows changed" in result.diagnostic
    elif relation == "relabel":
        assert "payload differs from its masked records" in result.diagnostic
    else:
        assert "log payload changed" in result.diagnostic
