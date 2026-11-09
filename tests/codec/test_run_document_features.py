"""A run document of each enabled feature reloads to the same study and history.

Each case builds a small study with one Slice 3 feature on, dumps it as a host run
document, loads it back and requires the dump, the study and the executed history to
match the original.
"""

import dataclasses

import equinox as eqx
import pytest

from spaceodyssey.codec import dump_run, load_run
from spaceodyssey.core import HostExecutor, preflight
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.studies import toy
from tests.codec.run_cases import bindings
from tests.codec.test_document import TOY_KINDS, assert_same_pytree
from tests.core.preflight_cases import DOC_KINDS

N_FEATURE_STEPS = 20

# channel 12 sits on lane 2, which is on asset 1, so its draws carry a nonzero asset
# coordinate under key version 2 and the history depends on the key version
FEATURE_PLAN = ((0, 10), (1, 11), (0, 12), (1, 12))
PLAN_DOC_KIND = ("policy", "test.doc_plan_policy", 1)


class DocPolicy(toy.FixedPlanPolicy):
    """A plan policy whose declared programs, awards and disruptive flag are data."""

    programs: tuple = eqx.field(static=True, default=(), kw_only=True)
    awards: tuple = eqx.field(static=True, default=(), kw_only=True)
    disruptive: bool = eqx.field(static=True, default=False, kw_only=True)

    def declare(self):
        return dataclasses.replace(
            super().declare(),
            programs=self.programs,
            awards=self.awards,
            disruptive=self.disruptive,
        )


@pytest.mark.medium(
    reason="one study per feature run twice on the host, 3 to 5 s cold each"
)
@pytest.mark.parametrize(
    "feature",
    ["programs", "refunds", "awards", "commitments", "balances", "key_version"],
)
def test_an_enabled_feature_run_recipe_redumps_exactly(feature):
    """Fails if a valid enabled feature changes its study, registry or bindings on a
    run document round trip, if the loaded study differs from the original in any
    leaf or static field, or if the loaded recipe executes to another history.
    """
    kinds = {**TOY_KINDS, **DOC_KINDS, PLAN_DOC_KIND: DocPolicy}
    caps = {
        "programs": Capacities(n_programs=2),
        "refunds": Capacities(refunds=True),
        "awards": Capacities(awards=True),
        "commitments": Capacities(n_commitments=2),
        "balances": Capacities(snapshot_balances=True, balance_quantum=0.5),
        "key_version": Capacities(),
    }[feature]
    policy = DocPolicy(
        plan=FEATURE_PLAN,
        programs=(0, 1) if feature == "programs" else (),
        awards=((2, 10, 80),) if feature == "awards" else (),
        disruptive=feature == "commitments",
    )
    study = toy.survey_1(
        caps=caps, policies=(policy,), key_version=2 if feature == "key_version" else 1
    )
    if feature == "refunds":
        study = dataclasses.replace(
            study,
            channels=dataclasses.replace(
                study.channels, charge=study.channels.cost.at[1, 2].set(7.0)
            ),
        )
    if feature == "commitments":
        study = dataclasses.replace(
            study,
            assets=dataclasses.replace(
                study.assets,
                freeze_ticks=study.assets.freeze_ticks.at[1].set(5),
                disruptive_resource=study.assets.disruptive_resource.at[1].set(1),
            ),
        )
    assert preflight(study) == []
    table = bindings()
    doc = dump_run(
        study, toy.toy_registry(), executor="host", bindings=table, kinds=kinds
    )
    loaded = load_run(doc.to_json(), table, kinds)
    again = dump_run(
        loaded.study,
        loaded.registry,
        executor=loaded.executor,
        bindings=table,
        service=loaded.make_service(),
        refit=loaded.refit or None,
        kinds=kinds,
    )
    assert again.to_json() == doc.to_json()
    assert loaded.executor == "host"
    assert loaded.service is None
    assert not loaded.refit
    assert_same_pytree(loaded.study, study)
    truths, roots = make_worlds(study, 2)
    mine = HostExecutor().run_worlds(study, truths, roots, N_FEATURE_STEPS)
    theirs = loaded.execute(truths, roots, N_FEATURE_STEPS)
    for w in range(2):
        assert histories_equal(mine.history(w), theirs.history(w)).ok
