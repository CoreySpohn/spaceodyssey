"""Seeded-mistake and correct studies for the pre-flight checks (not a test file).

Each seeded study is `survey_1` (or the catalog study) with one thing wrong, and the
single code it must raise. Components state their declarations through a static
`Declaration`, the way a real component defines `declare`.
"""

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import (
    DRAW_COORDS,
    EXECUTORS,
    HOST,
    SCORER,
    TRUTH,
    Declaration,
    Need,
)
from spaceodyssey.core.enums import (
    I64,
    Disposition,
    EffectKind,
    Scope,
)
from spaceodyssey.core.preflight import FindingCode as C
from spaceodyssey.core.preflight.checks import PreflightContext
from spaceodyssey.core.refresh import HostLoopRefit, MaskedBatchRefit
from spaceodyssey.studies import catalog, toy
from spaceodyssey.studies.premission import PreMissionImporter, authored_table
from spaceodyssey.studies.prior_observer import prior_study
from tests.core.corpus import CORPUS, frozen
from tests.core.prior_helpers import EXTERNAL, refit_study, summaries


class StatedObserver(toy.ToyRunningMean):
    stated: Declaration = eqx.field(static=True, default=Declaration())

    def declare(self):
        return self.stated


class StatedPolicy(toy.NullPolicy):
    stated: Declaration = eqx.field(static=True, default=Declaration())

    def declare(self):
        return self.stated


class StatedChannelModel(toy.ToyPhotometry):
    stated: Declaration = eqx.field(static=True, default=Declaration())

    def declare(self):
        return self.stated


class StatedWorld(toy.ToyWorldSource):
    stated: Declaration = eqx.field(static=True, default=Declaration())

    def declare(self):
        return self.stated


def _field(default):
    return eqx.field(static=True, default=default, kw_only=True)


class DocObserver(toy.ToyRunningMean):
    """An observer whose declaration is plain data, so a document can state it."""

    reads: tuple = _field(())

    def declare(self):
        return Declaration(reads=self.reads)


class DocPolicy(toy.NullPolicy):
    """A policy whose declaration is plain data: `needs` holds (channel id, start
    tick, decide tick) triples, `stops` (scope, disposition) pairs and `effects`
    effect-kind numbers.
    """

    reads: tuple = _field(())
    needs: tuple = _field(())
    stops: tuple = _field(())
    effects: tuple = _field(())
    requires: tuple = _field(())
    executors: tuple = _field(tuple(sorted(EXECUTORS)))
    disruptive: bool = _field(False)
    programs: tuple = _field(())

    def declare(self):
        return Declaration(
            programs=self.programs,
            reads=self.reads,
            needs=tuple(Need(*n) for n in self.needs),
            stops=tuple((Scope(a), Disposition(b)) for a, b in self.stops),
            effects=frozenset(EffectKind(e) for e in self.effects),
            requires=frozenset(self.requires),
            executors=frozenset(self.executors),
            disruptive=self.disruptive,
        )


class DocChannelModel(toy.ToyPhotometry):
    """A channel model whose key coordinates a document states."""

    key_coords: tuple = _field(DRAW_COORDS)
    effects: tuple = _field(())

    def declare(self):
        return Declaration(
            key_coords=self.key_coords,
            effects=frozenset(EffectKind(e) for e in self.effects),
        )


class DocObjective(toy.ToyObjective):
    """An objective whose declared effects a document states."""

    effects: tuple = _field(())

    def declare(self):
        return Declaration(effects=frozenset(EffectKind(e) for e in self.effects))


# The document kinds, to add to a resolver's mapping.
DOC_KINDS = {
    ("observer", "test.doc_observer", 1): DocObserver,
    ("policy", "test.doc_policy", 1): DocPolicy,
    ("channel_model", "test.doc_photometry", 1): DocChannelModel,
    ("objective", "test.doc_objective", 1): DocObjective,
}


def _replace(study, **fields):
    return dataclasses.replace(study, **fields)


def _with_policy(study, policy):
    agenda = dataclasses.replace(study.agenda, policies=(policy,))
    return _replace(study, agenda=agenda)


def _with_model(**stated):
    base = toy.survey_1()
    model = StatedChannelModel(
        noise=base.channel_model.noise, stated=Declaration(**stated)
    )
    return _replace(base, channel_model=model)


def _coords_without(name):
    return tuple(c for c in Declaration().key_coords if c != name)


def _observer(**stated):
    return _replace(
        toy.survey_1(), observer=StatedObserver(stated=Declaration(**stated))
    )


def _policy(**stated):
    return _with_policy(toy.survey_1(), StatedPolicy(stated=Declaration(**stated)))


def _assets(**columns):
    study = toy.survey_1()
    return _replace(study, assets=dataclasses.replace(study.assets, **columns))


def _channel_edit(**columns):
    study = toy.survey_1()
    return _replace(study, channels=dataclasses.replace(study.channels, **columns))


def _unsupported_kind():
    return _channel_edit(kind=toy.survey_1().channels.kind.at[2].set(9))


def _dropped_order():
    return _channel_edit(coincident_order=None)


def import_study(mistake=None):
    """`survey_1` importing two public measurement rows, with one seeded mistake.

    `mistake` is `None` (a correct table), `"upstream"` (a declared upstream hash,
    unverified without a document), `"pedigree"` (an in-table pedigree entry naming
    an absent source), `"index"` (a local index repeated on one channel) or
    `"offset"` (a row that ends after its release).
    """
    pedigree = [(), ()]
    if mistake == "pedigree":
        pedigree = [(), ((0, 60, 0, 7, False),)]
    table = authored_table(
        subject=[0, 1],
        local_index=[0, 0 if mistake == "index" else 1],
        channel=[60, 60],
        kind=[toy.RECORD_MEASUREMENT] * 2,
        audience=[3, 3],
        payload=[[1.0], [2.0]],
        start_offset=[-5, -5],
        end_offset=[3 if mistake == "offset" else -1, -1],
        pedigree=pedigree,
        source_index=[0, 1],
        upstream_hash="ab" * 32 if mistake == "upstream" else "",
    )
    return toy.survey_1(process=PreMissionImporter(table=table))


# name -> (the one code the study must raise, builder, executor selected)
SEEDED = {
    "observer_reads_truth": (C.F001, lambda: _observer(reads=(TRUTH,)), None),
    "policy_reads_truth": (C.F001, lambda: _policy(reads=(TRUTH,)), None),
    "observer_reads_scorer": (C.F002, lambda: _observer(reads=(SCORER,)), None),
    "decision_before_release": (
        C.C001,
        lambda: _policy(needs=(Need(10, 0, 19),)),
        None,
    ),
    "decision_reads_observer_only_channel": (
        C.C001,
        lambda: _with_policy(
            catalog.catalog_study(),
            StatedPolicy(
                stated=Declaration(needs=(Need(catalog.CATALOG_CHANNEL_ID, 0, 5000),))
            ),
        ),
        None,
    ),
    "draws_without_acquisition_index": (
        C.K001,
        lambda: _with_model(key_coords=_coords_without("acquisition_index")),
        None,
    ),
    "world_draw_without_world": (
        C.K001,
        lambda: _replace(
            toy.survey_1(),
            world_source=StatedWorld(stated=Declaration(key_coords=("subject",))),
        ),
        None,
    ),
    "duplicate_channel_ids": (
        C.K002,
        lambda: _channel_edit(id=jnp.asarray([10, 10, 12], I64)),
        None,
    ),
    "allocate": (
        C.X001,
        lambda: _policy(effects=frozenset({EffectKind.ALLOCATE})),
        None,
    ),
    "gradients_on_host": (
        C.X001,
        lambda: _policy(requires=frozenset({"gradients"})),
        HOST,
    ),
    "host_only_policy_on_compiled": (
        C.X001,
        lambda: _policy(executors=frozenset({HOST})),
        "compiled",
    ),
    "charge_differs_from_cost": (
        C.X002,
        lambda: _channel_edit(charge=2.0 * jnp.ones((3, 3))),
        None,
    ),
    "frozen_horizon": (
        C.X002,
        lambda: _assets(freeze_ticks=jnp.asarray([5, 0], I64)),
        None,
    ),
    "disruptive_resource": (
        C.X002,
        lambda: _assets(disruptive_resource=jnp.asarray([0, -1], I64)),
        None,
    ),
    "disruptive_effect": (C.X002, lambda: _policy(disruptive=True), None),
    "drain_below_campaign": (
        C.T001,
        lambda: _with_policy(
            toy.survey_1(),
            frozen(stop_at=2, stop_scope=Scope.LANE, stop_disp=Disposition.DRAIN),
        ),
        None,
    ),
    "program_out_of_range": (C.P001, lambda: _policy(programs=(1,)), None),
    "channel_without_order": (C.T002, _dropped_order, None),
    "channel_order_out_of_range": (
        C.T002,
        lambda: _channel_edit(coincident_order=jnp.asarray([0, 1, 7], I64)),
        None,
    ),
    "channel_kind_unregistered": (C.R001, _unsupported_kind, None),
    "import_upstream_unverified": (C.D004, lambda: import_study("upstream"), None),
    "import_pedigree_absent": (C.D005, lambda: import_study("pedigree"), None),
    "import_index_repeated": (C.D006, lambda: import_study("index"), None),
    "import_offset_positive": (C.D007, lambda: import_study("offset"), None),
    "priors_summarize_one_source": (
        C.D008,
        lambda: prior_study(summaries(EXTERNAL)),
        None,
    ),
    "prior_refit_boundary_unverified": (C.D010, refit_study, None),
}

# Mistakes that only a run's context reveals: name -> (code, builder, context).
SEEDED_IN_CONTEXT = {
    "prior_audiences_on_the_observer_init_boundary": (
        C.D009,
        refit_study,
        lambda: PreflightContext(boundary=MaskedBatchRefit()),
    ),
    "prior_with_a_boundary_that_ignores_it": (
        C.D010,
        refit_study,
        lambda: PreflightContext(boundary=HostLoopRefit(toy.toy_refit_fit)),
    ),
}

# Correct studies and the codes they must raise (warnings only).
_DRAIN_BELOW = {f"stop_{scope}_drain" for scope in ("lane", "asset", "program")}
CLEAN = {
    "survey_1": (toy.survey_1, ()),
    "catalog_study": (catalog.catalog_study, ()),
    "d2_study": (toy.d2_study, ()),
    "import_study": (import_study, ()),
    "prior_study": (lambda: prior_study(summaries((7, 90, 0, 1))), ()),
    "stated_clean_components": (
        lambda: _policy(
            reads=(),
            needs=(Need(10, 0, 500),),
            requires=frozenset({"gradients"}),
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),),
        ),
        (),
    ),
    "policy_names_held_programs": (
        lambda: _with_policy(
            toy.survey_1(caps=Capacities(n_programs=2)),
            StatedPolicy(stated=Declaration(programs=(0, 1))),
        ),
        (),
    ),
    "policy_asks_query": (
        lambda: _policy(effects=frozenset({EffectKind.QUERY})),
        (),
    ),
}
for case in CORPUS:
    CLEAN[f"corpus_{case.name}"] = (
        case.make_study,
        (C.T001,) if case.name in _DRAIN_BELOW else (),
    )
